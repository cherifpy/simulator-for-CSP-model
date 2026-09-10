#!/usr/bin/env python3
"""
All-in-one launcher for Grid5000 (or any fresh machine): sets up everything needed (Python
venv, pinned dependencies, optionally a modern JDK, compiles the Java CSP model) and then runs
the experiment -- a single command instead of setup_grid5000.sh + exps/xp_online_grid5000.py
run separately.

Can be launched with the system python3 (no venv needed yet): it bootstraps the venv itself,
installs dependencies into it, then re-executes itself under that venv's interpreter to
actually run the experiment.

Examples:
    python3 launcher.py --approach online --instance-dir workloads/.../inst-10J-20N \\
        --nb-jobs 10 --nb-nodes 20 --solver-time-limit 10 --lambda-rate 60

    # Grid5000 node with an old default JDK (Choco fails with a low-level JVM startup error,
    # e.g. "graal_create_isolate error", on anything too old):
    python3 launcher.py --approach online --instance-dir ... --nb-jobs 10 --nb-nodes 20 \\
        --solver-time-limit 10 --lambda-rate 60 --java25

    # Already set up (venv present, Java compiled) and just want to run again without
    # re-checking/re-installing everything:
    python3 launcher.py --approach incremental --instance-dir ... --nb-jobs 50 --nb-nodes 50 \\
        --solver-time-limit 20 --lambda-rate 100 --skip-setup
"""

import argparse
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../simulator-for-CSP-model/simulator
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                    # .../simulator-for-CSP-model
VENV_DIR = os.path.join(PROJECT_ROOT, "env")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
MODEL_DIR = os.path.join(SCRIPT_DIR, "utils", "model")

JDK_URL = "https://download.oracle.com/java/25/latest/jdk-25_linux-x64_bin.deb"
JDK_FILE = "/tmp/jdk-25_linux-x64_bin.deb"


def run(cmd, check=True, **kwargs):
    print(">>>", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        print(f"### Command failed ({result.returncode}): {' '.join(cmd)}", file=sys.stderr)
        sys.exit(1)
    return result


def venv_python():
    return os.path.join(VENV_DIR, "bin", "python3")


def venv_is_usable():
    """A venv is NOT portable across machines: bin/python3's shebang/binary is specific to the
    machine that created it. If this whole project folder was copied from elsewhere (scp/rsync
    of the full tree including a pre-existing env/), that venv won't run here -- detect that
    instead of trusting the directory's mere presence."""
    py = venv_python()
    if not os.path.exists(py) or not os.access(py, os.X_OK):
        return False
    try:
        subprocess.run([py, "-c", ""], check=True, capture_output=True)
        return True
    except Exception:
        return False


def setup_venv():
    if venv_is_usable():
        print(f"### Venv already exists at {VENV_DIR} and works here, reusing it.")
        return
    if os.path.isdir(VENV_DIR):
        print(f"### Venv at {VENV_DIR} isn't usable on this machine (likely copied from "
              f"another machine -- venvs aren't portable). Recreating it.")
        shutil.rmtree(VENV_DIR)
    print(f"### Creating venv at {VENV_DIR}")
    run([sys.executable, "-m", "venv", VENV_DIR])


def install_requirements():
    if not os.path.isfile(REQUIREMENTS_FILE):
        print(f"### ERROR: requirements.txt not found at {REQUIREMENTS_FILE}", file=sys.stderr)
        sys.exit(1)
    print(f"### Installing Python dependencies from {REQUIREMENTS_FILE}")
    run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"], capture_output=True)
    run([venv_python(), "-m", "pip", "install", "-r", REQUIREMENTS_FILE])


def java_25_installed():
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return "25" in result.stderr
    except FileNotFoundError:
        return False


def install_java_25():
    print("### Downloading Java 25...")
    run(["wget", JDK_URL, "-O", JDK_FILE])
    print("### Installing Java 25 (sudo-g5k dpkg -i)...")
    run(["sudo-g5k", "dpkg", "-i", JDK_FILE])
    print("### Fixing dependencies if needed (sudo-g5k apt -f install)...")
    run(["sudo-g5k", "apt", "-f", "install", "-y"], check=False)
    print("### Verifying installation...")
    run(["java", "-version"])
    run(["javac", "-version"])


def compile_java():
    print(f"### Compiling Java CSP model in {MODEL_DIR}")
    bin_dir = os.path.join(MODEL_DIR, "bin", "main")
    os.makedirs(bin_dir, exist_ok=True)
    lib_glob = os.path.join(MODEL_DIR, "lib", "*")
    for main_class in ("MainOnline", "MainIncremental"):
        src = os.path.join(MODEL_DIR, "src", "main", f"{main_class}.java")
        run(["javac", "-d", bin_dir, "-cp", lib_glob, src])


def do_setup(args):
    setup_venv()
    install_requirements()

    if args.java25:
        if java_25_installed():
            print("### Java 25 already installed, skipping.")
        else:
            install_java_25()

    if shutil.which("java") is None or shutil.which("javac") is None:
        print("### ERROR: java/javac not found on PATH. Either 'module load' a JDK, or "
              "re-run with --java25 to install one.", file=sys.stderr)
        sys.exit(1)
    version_lines = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()
    print(f"### {version_lines[0] if version_lines else '(could not read java -version output)'}")

    compile_java()


def reexec_under_venv():
    """Re-run this exact command under the venv's python (which now has simpy/pandas/etc.
    installed), now that setup is done. Only needed the first time -- if the caller already
    used the venv's python (or --skip-setup), this is skipped."""
    py = venv_python()
    print(f"### Re-launching under {py}", flush=True)
    os.execv(py, [py, os.path.abspath(__file__)] + sys.argv[1:] + ["--skip-setup"])


def run_experiment(args):
    # Only import after the venv's interpreter (with simpy/pandas/etc.) is the one running --
    # importing at module load time would fail under the bootstrap system python.
    exps_dir = os.path.join(SCRIPT_DIR, "exps")
    if exps_dir not in sys.path:
        sys.path.append(exps_dir)
    import exps.xp_online_grid5000 as xp

    ns = argparse.Namespace(
        approach=args.approach,
        instance_dir=args.instance_dir,
        nb_jobs=args.nb_jobs,
        nb_nodes=args.nb_nodes,
        solver_time_limit=args.solver_time_limit,
        lambda_rate=args.lambda_rate,
        config=args.config or os.path.join(SCRIPT_DIR, "config.json"),
        results_dir=args.results_dir,
        seed=args.seed,
        skip_gantt=args.skip_gantt,
    )

    xp.configure_logging(xp.logging.WARNING)
    results_dir = xp.run(ns)

    violations = xp.verify_storage(results_dir, ns.instance_dir)
    print()
    print("=" * 70)
    if not violations:
        print("STORAGE CHECK: OK -- no violation at any instant, on any node.")
    else:
        print(f"STORAGE CHECK: {len(violations)} violation(s) found:")
        for node, t, occupied, cap, label in violations:
            print(f"  node_{node} at t={t:.2f}: occupied={occupied:.1f} > capacity={cap:.1f} ({label})")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", required=True, choices=["online", "incremental", "incremental_free_nodes_only"])
    parser.add_argument("--instance-dir", required=True,
                         help="Directory containing this instance's jobs.json and infrastructure.csv.")
    parser.add_argument("--nb-jobs", required=True, type=int)
    parser.add_argument("--nb-nodes", required=True, type=int)
    parser.add_argument("--solver-time-limit", required=True, type=int, help="Seconds per CSP solve.")
    parser.add_argument("--lambda-rate", type=int, default=60)
    parser.add_argument("--config", default=None, help="Defaults to <simulator>/config.json.")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-gantt", action="store_true")
    parser.add_argument("--java25", action="store_true",
                         help="Install Oracle JDK 25 via sudo-g5k if not already present (Grid5000).")
    parser.add_argument("--skip-setup", action="store_true",
                         help="Skip venv/deps/Java setup entirely and go straight to running the "
                              "experiment (assumes it was already done in a previous invocation).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_setup:
        do_setup(args)

    # Whether or not setup just ran, the experiment itself needs the venv's interpreter (it's
    # the only one with simpy/pandas/etc. installed). --skip-setup only skips venv/deps/Java
    # setup -- it must NOT also skip switching to that interpreter, or a bare
    # `--skip-setup` run under the system python fails with "No module named 'simpy'".
    #
    # A venv's bin/python3 is typically just a SYMLINK to the system python3 it was created
    # from, so comparing realpath(sys.executable) to realpath(venv_python()) is wrong -- both
    # resolve to the exact same underlying binary, making them look "already equal" even when
    # not actually running inside the venv (its site-packages wouldn't be on sys.path at all).
    # sys.prefix is what Python itself sets to the venv's root when genuinely launched through
    # it, regardless of what the executable symlinks to -- that's the correct check.
    if os.path.realpath(sys.prefix) != os.path.realpath(VENV_DIR):
        reexec_under_venv()  # process replaced -- nothing below runs in this invocation
        return

    run_experiment(args)


if __name__ == "__main__":
    main()
