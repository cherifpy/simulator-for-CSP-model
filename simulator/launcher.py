#!/usr/bin/env python3
"""
All-in-one launcher for Grid5000 (or any fresh machine): sets up everything needed (Python
venv, pinned dependencies, optionally a modern JDK, compiles the Java CSP model) and then runs
one of the experiment scripts under exps/ -- a single command instead of setup_grid5000.sh +
the experiment script run separately.

Can be launched with the system python3 (no venv needed yet): it bootstraps the venv itself,
installs dependencies into it, then re-executes itself under that venv's interpreter to
actually run the experiment.

--experiment selects which script under exps/ actually runs (default: online). Every other
flag is that script's OWN command-line interface, passed through unchanged -- see each script's
own --help for its full flag set (they differ: xp_online_grid5000.py wants --nb-jobs/--skip-gantt,
xp_single_decision_grid5000.py wants --n-existing/--new-job-index/--state-a-time-limit instead).

Examples:
    # Default experiment (xp_online_grid5000.py): full simulation run, one approach at a time.
    python3 launcher.py --approach online --instance-dir workloads/.../inst-10J-20N \\
        --nb-jobs 10 --nb-nodes 20 --solver-time-limit 10 --lambda-rate 60

    # Controlled single-decision-point test (xp_single_decision_grid5000.py): e.g. a 2h Online
    # solve for the one new job arriving on top of N=15 already-in-progress jobs.
    python3 launcher.py --experiment single_decision --approach online \\
        --instance-dir workloads/.../inst-20J-50N --nb-nodes 50 --n-existing 15 \\
        --solver-time-limit 7200 --lambda-rate 100

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


EXPERIMENT_MODULES = {
    "online": "exps.xp_online_grid5000",
    "single_decision": "exps.xp_single_decision_grid5000",
}


def run_experiment(experiment, extra_argv):
    """Delegates entirely to the chosen script's own argparse + main() -- each exps/xp_*.py is
    a complete, independently-runnable CLI tool (see their own --help), so the launcher doesn't
    re-declare or mirror their flags; it just forwards whatever wasn't consumed by the launcher's
    own --experiment/--java25/--skip-setup."""
    exps_dir = os.path.join(SCRIPT_DIR, "exps")
    if exps_dir not in sys.path:
        sys.path.append(exps_dir)
    import importlib
    xp = importlib.import_module(EXPERIMENT_MODULES[experiment])

    sys.argv = [sys.argv[0]] + extra_argv
    xp.main()


def parse_launcher_args():
    """Only the launcher's OWN flags are declared here (add_help=False so -h/--help falls
    through, unconsumed, to the chosen experiment script's own parser instead of this one).
    Everything else (--approach, --instance-dir, ...) is experiment-specific and is left in
    `extra_argv` for that script's own parse_args() to handle."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_MODULES.keys()), default="online",
                         help="Which exps/xp_*.py script to run (default: online).")
    parser.add_argument("--java25", action="store_true",
                         help="Install Oracle JDK 25 via sudo-g5k if not already present (Grid5000).")
    parser.add_argument("--skip-setup", action="store_true",
                         help="Skip venv/deps/Java setup entirely and go straight to running the "
                              "experiment (assumes it was already done in a previous invocation).")
    return parser.parse_known_args()


def main():
    launcher_args, extra_argv = parse_launcher_args()

    if not launcher_args.skip_setup:
        do_setup(launcher_args)

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

    run_experiment(launcher_args.experiment, extra_argv)


if __name__ == "__main__":
    main()
