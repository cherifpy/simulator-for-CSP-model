"""
Shared "save everything" machinery for every experiment script.

Principle: a run writes the RAW data (every solve's decisions, the solver's own console output, the
exact files exchanged with the Java model, the infrastructure and the parameters), and statistics
are computed AFTERWARDS from that -- so a new statistic never needs a re-run.

Two entry points:

* start_recording(results_dir, params=..., nodes_config=...)  -- call once per run, right after the
  results directory exists. It (1) writes run_params.json (+ nodes_config.csv when given), (2) tees
  everything printed from now on into <results_dir>/solver_stdout.log, and (3) turns on the
  per-solve archive below. Safe to call again for the next run of a multi-run script: it just
  switches to the new directory.

* the per-solve archive -- once recording is on, EVERY call to utils.modelCSP.schedulingUsingJavaCSP
  (the single entry point to the Java CSP model, used by all the experiment scripts) saves, under
  <results_dir>/solver_archive/solve_00001/, solve_00002/, ...:
      meta.json        when it was called (simulation time), which Java model / objective / epsilon
                       settings, the solver budget, the jobs handed to it, the node free times,
                       wall-clock time, solver stats (search time, optimality, phase bests)
      solver.log       the solver's complete console output for that solve
      tasks.csv / transfers.csv (with the energy split) / deletions.csv   the decisions, absolute times
      trajectory.csv   every improving solution found, with its solve-time stamp and phase
      raw_io/          the raw Java input and output files of that solve
"""
import atexit
import contextlib
import csv
import io
import json
import os
import re
import shutil
import sys
import time

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_DIR = os.path.dirname(UTILS_DIR)

TASKS_FIELDS = ["job_id", "task_id", "node", "start", "end", "duration", "origin"]
TRANSFERS_FIELDS = ["job_id", "node", "start", "end", "duration", "sender_energy", "receiver_energy",
                    "network_energy", "energy", "origin"]
DELETIONS_FIELDS = ["job_id", "node", "deletion_time", "origin"]
TRAJECTORY_FIELDS = ["index", "phase", "t", "sum_flow_time", "max_flow_time", "new_job_flow_time", "energy", "nb_data"]


class _Tee(io.TextIOBase):
    """Writes to every given stream -- output is shown live AND kept."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
        return len(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def run_captured(fn, *fargs):
    """Runs fn(*fargs), returning (result, everything it printed, wall-clock seconds)."""
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        result = fn(*fargs)
    return result, buf.getvalue(), time.time() - t0


def solver_model_dir():
    """utils/model directory the Java solver actually reads/writes for THIS process (honours the
    SIMULATOR_RUN_CWD isolation used when several runs share one checkout, and -- taking
    precedence over it -- a per-thread override for isolating concurrent solves WITHIN one
    process; see utils.modelCSP.run_cwd_override)."""
    from utils.modelCSP import get_run_cwd_override
    run_cwd = get_run_cwd_override() or os.environ.get("SIMULATOR_RUN_CWD", SIMULATOR_DIR)
    return os.path.join(run_cwd, "utils", "model")


def snapshot_solver_io(run_dir):
    """Copies the raw Java exchange files of the solve that JUST ended (inputs/ = what the solver was
    given, outputs/ = raw works/transfers/deletions it produced) into <run_dir>/raw_io/."""
    for sub in ("inputs", "outputs"):
        src = os.path.join(solver_model_dir(), sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(run_dir, "raw_io", sub), dirs_exist_ok=True)


def write_rows(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plan_rows(master, plan, origin):
    """Flattens a solve's {transfers, works, deletions} (dicts keyed by node) into row lists."""
    sender = master._config.get('master_energy_consumption', 0.0)
    network = master._config.get('network_energy_per_transfer', 0.0)
    tasks, transfers, deletions = [], [], []
    for entries in (plan.get("works") or {}).values():
        for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
            tasks.append({"job_id": job_id, "task_id": task_index, "node": node_index, "start": start_abs,
                          "end": end_abs, "duration": duration, "origin": origin})
    for entries in (plan.get("transfers") or {}).values():
        for job_id, node_index, start_abs, end_abs, duration in entries:
            receiver = master.compute_nodes[node_index].energy_consumption * duration
            transfers.append({"job_id": job_id, "node": node_index, "start": start_abs, "end": end_abs,
                              "duration": duration, "sender_energy": sender, "receiver_energy": receiver,
                              "network_energy": network, "energy": sender + receiver + network, "origin": origin})
    for key, entries in (plan.get("deletions") or {}).items():
        for job_id, deletion_time in entries:
            deletions.append({"job_id": job_id, "node": int(key.split("_")[-1]),
                              "deletion_time": deletion_time, "origin": origin})
    return {"tasks": tasks, "transfers": transfers, "deletions": deletions}


_SOL_RE = re.compile(r"### DIAG solution found: sumFlowTime=(\S+) maxFlowTime=(\S+)(?: newJobFlowTime=(\S+))?"
                     r"(?: energy=(\S+))? nb_data=(\S+)(?: t=(\S+))?")
_END_RE = re.compile(r"### DIAG search end: timeCount=(\S+)s\s+timeLimitWas=(\S+)s\s+objectiveOptimal=(\S+)\s+solutionCount=(\S+)")
_P1_RE = re.compile(r"phase1 best maxFlowTime=(\S+)\s+phase2 cap=maxFlowTime<=(\S+)")
_P2_RE = re.compile(r"phase2 best energy=(\S+)\s+\(maxFlowTime=(\S+)\)")


def parse_solver_log(log_text):
    """Turns the Java solver's console output into (trajectory rows, stats dict): every improving
    solution it found in order (with its solve-time stamp and objective values), which epsilon phase it
    belongs to, and the final search stats. The raw log itself is saved too (solver.log)."""
    trajectory, stats, phase = [], {}, 1
    for line in log_text.splitlines():
        m = _SOL_RE.search(line)
        if m:
            f = lambda v: (float(v) if v is not None else None)
            trajectory.append({"index": len(trajectory), "phase": phase, "t": f(m.group(6)),
                               "sum_flow_time": f(m.group(1)), "max_flow_time": f(m.group(2)),
                               "new_job_flow_time": f(m.group(3)), "energy": f(m.group(4)),
                               "nb_data": int(m.group(5))})
            continue
        m = _P1_RE.search(line)
        if m:
            stats["phase1_best_max_flow_time"] = float(m.group(1)); stats["phase2_cap"] = float(m.group(2)); phase = 2
            continue
        m = _P2_RE.search(line)
        if m:
            stats["phase2_best_energy"] = float(m.group(1)); stats["phase2_best_max_flow_time"] = float(m.group(2))
            continue
        m = _END_RE.search(line)
        if m:
            stats.update({"search_time_s": float(m.group(1)), "time_limit_s": float(m.group(2)),
                          "objective_optimal": m.group(3) == "true", "solution_count_reported": int(m.group(4))})
        if "warm start HELD" in line:
            stats["warm_start_held"] = True
    stats["solutions_logged"] = len(trajectory)
    stats["solutions_phase1"] = sum(1 for r in trajectory if r["phase"] == 1)
    stats["solutions_phase2"] = sum(1 for r in trajectory if r["phase"] == 2)
    return trajectory, stats


# ------------------------------------------------------------------------------ per-solve archive
def archive_solve(impl, master, jobs, replicas_locations, nodes_free_time, scheduling_start_time, archive_dir):
    """Runs the real solver call `impl` and archives it under archive_dir/solve_NNNNN/ (see module doc).
    Never lets an archiving problem break the experiment itself: the solve's result is always returned."""
    result, log_text, wall_s = run_captured(impl, master, jobs, replicas_locations, nodes_free_time, scheduling_start_time)
    try:
        os.makedirs(archive_dir, exist_ok=True)
        seq = sum(1 for d in os.listdir(archive_dir) if d.startswith("solve_")) + 1
        solve_dir = os.path.join(archive_dir, f"solve_{seq:05d}")
        os.makedirs(solve_dir, exist_ok=True)
        transfers_, works_, deletions_ = result
        rows = plan_rows(master, {"transfers": transfers_, "works": works_, "deletions": deletions_}, "solve")
        trajectory, stats = parse_solver_log(log_text)
        write_rows(os.path.join(solve_dir, "tasks.csv"), TASKS_FIELDS, rows["tasks"])
        write_rows(os.path.join(solve_dir, "transfers.csv"), TRANSFERS_FIELDS, rows["transfers"])
        write_rows(os.path.join(solve_dir, "deletions.csv"), DELETIONS_FIELDS, rows["deletions"])
        write_rows(os.path.join(solve_dir, "trajectory.csv"), TRAJECTORY_FIELDS, trajectory)
        with open(os.path.join(solve_dir, "solver.log"), "w") as f:
            f.write(log_text)
        snapshot_solver_io(solve_dir)
        meta = {
            "seq": seq, "env_now": master.env.now, "scheduling_start_time": scheduling_start_time,
            "java_main_class": getattr(master, "java_main_class", None),
            "objective_choice": getattr(master, "objective_choice", None),
            "multi_objective": getattr(master, "multi_objective", None),
            "epsilon_fraction": getattr(master, "epsilon_fraction", None),
            "epsilon_phase1_fraction": getattr(master, "epsilon_phase1_fraction", None),
            "epsilon_max_cap": getattr(master, "epsilon_max_cap", None),
            "solver_time_limit_s": master._config.get("solver_time_limit_s"),
            "wall_time_s": wall_s, "solver": stats,
            "n_tasks": len(rows["tasks"]), "n_transfers": len(rows["transfers"]), "n_deletions": len(rows["deletions"]),
            "jobs": [{"job_id": j.job_id, "dataset_size": j.dataset_size, "nb_tasks": len(j.tasks),
                      "nb_tasks_not_started": sum(1 for t in j.tasks if t.status == "NotStarted"),
                      "arriving_time": getattr(j, "arriving_time", None)} for j in jobs],
            "nodes_free_time": {str(k): v for k, v in dict(nodes_free_time).items()},
        }
        with open(os.path.join(solve_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
    except Exception as exc:  # archiving must never take the experiment down
        print(f"### WARNING: solve archive failed ({type(exc).__name__}: {exc}); the run continues ###", flush=True)
    return result


# ------------------------------------------------------------------------------ run recording
class _Recorder:
    def __init__(self):
        self.original_stdout = None
        self.log_file = None
        self.installed = False

    def switch_to(self, results_dir):
        if self.log_file is not None:
            self.log_file.flush(); self.log_file.close()
        self.log_file = open(os.path.join(results_dir, "solver_stdout.log"), "a")
        if not self.installed:
            self.original_stdout = sys.stdout
            sys.stdout = _Tee(self.original_stdout, _LogProxy(self))
            self.installed = True
            atexit.register(self.close)

    def close(self):
        if self.log_file is not None:
            try:
                self.log_file.flush(); self.log_file.close()
            except Exception:
                pass
            self.log_file = None


class _LogProxy(io.TextIOBase):
    """Forwards to whichever log file the recorder currently points at (switchable between runs)."""
    def __init__(self, recorder):
        self.recorder = recorder

    def write(self, text):
        if self.recorder.log_file is not None:
            self.recorder.log_file.write(text)
        return len(text)

    def flush(self):
        if self.recorder.log_file is not None:
            self.recorder.log_file.flush()


_recorder = _Recorder()


def start_recording(results_dir, params=None, config=None, nodes_config=None, archive=True):
    """See the module doc. `params` (dict or argparse Namespace), `config` and `nodes_config` are saved
    as-is (run_params.json / nodes_config.csv) when given; archive=False skips the per-solve archive."""
    os.makedirs(results_dir, exist_ok=True)
    if params is not None:
        with open(os.path.join(results_dir, "run_params.json"), "w") as f:
            json.dump({"params": vars(params) if hasattr(params, "__dict__") else params,
                       "config_used": config, "argv": sys.argv}, f, indent=2, default=str)
    if nodes_config is not None:
        fields = ["node_id", "bandwidth", "computation_nodes", "energy_consumption", "storage_capacity"]
        write_rows(os.path.join(results_dir, "nodes_config.csv"), fields,
                   [{"node_id": i, **{k: cfg.get(k) for k in fields[1:]}} for i, cfg in enumerate(nodes_config)])
    _recorder.switch_to(results_dir)
    if archive:
        os.environ["SOLVER_ARCHIVE_DIR"] = os.path.join(results_dir, "solver_archive")
    else:
        os.environ.pop("SOLVER_ARCHIVE_DIR", None)
