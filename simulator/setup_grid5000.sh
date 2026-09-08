#!/usr/bin/env bash
# One-shot environment setup for running the experiments on Grid5000 (or any fresh machine):
# creates/activates a Python venv, installs the pinned dependencies, optionally installs a
# modern JDK, and compiles the Java CSP model. Safe to re-run -- each step is skipped if
# already done.
#
# Usage:
#   cd simulator-for-CSP-model
#   ./simulator/setup_grid5000.sh              # uses whatever java/javac is already on PATH
#   ./simulator/setup_grid5000.sh --java25      # also installs Oracle JDK 25 via sudo-g5k first
#
# Grid5000's default nodes often ship an old JDK, which fails Choco Solver with a
# "graal_create_isolate error" (or similar low-level JVM startup failure) -- not a GraalVM
# problem specifically, just too old a JVM. --java25 installs a current JDK the same way a
# previous offline version of this pipeline did (wget the .deb, install with sudo-g5k).
#
# After this finishes, activate the venv in your own shell before running experiments:
#   source env/bin/activate
#   cd simulator
#   python3 exps/xp_online_grid5000.py --approach online --instance-dir ... --nb-jobs ... \
#       --nb-nodes ... --solver-time-limit ... --lambda-rate ...

set -euo pipefail

INSTALL_JAVA25=false
for arg in "$@"; do
    case "$arg" in
        --java25) INSTALL_JAVA25=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../simulator-for-CSP-model/simulator
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"                      # .../simulator-for-CSP-model
VENV_DIR="$PROJECT_ROOT/env"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

echo "### Project root: $PROJECT_ROOT"
echo "### Simulator dir: $SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH. Load/module-add a Python toolchain first." >&2
    exit 1
fi

# --- Java 25 (optional) --------------------------------------------------------------
JDK_URL="https://download.oracle.com/java/25/latest/jdk-25_linux-x64_bin.deb"
JDK_FILE="/tmp/jdk-25_linux-x64_bin.deb"

java_25_installed() {
    command -v java >/dev/null 2>&1 && java -version 2>&1 | grep -q "25"
}

install_java_25() {
    echo "### Downloading Java 25..."
    wget "$JDK_URL" -O "$JDK_FILE"

    echo "### Installing Java 25 (sudo-g5k dpkg -i)..."
    sudo-g5k dpkg -i "$JDK_FILE"

    echo "### Fixing dependencies if needed (sudo-g5k apt -f install)..."
    sudo-g5k apt -f install -y

    echo "### Verifying installation..."
    java -version
    javac -version
}

if [ "$INSTALL_JAVA25" = true ]; then
    if java_25_installed; then
        echo "### Java 25 already installed, skipping."
    else
        install_java_25
    fi
fi

if ! command -v java >/dev/null 2>&1 || ! command -v javac >/dev/null 2>&1; then
    echo "ERROR: java/javac not found on PATH. Either 'module load' a JDK, or re-run this" >&2
    echo "       script with --java25 to install one." >&2
    exit 1
fi
echo "### $(java -version 2>&1 | head -1)"

# --- Python venv -------------------------------------------------------------------
# A venv is NOT portable across machines: its bin/pip and bin/python3 scripts have the
# interpreter's absolute path baked into their shebang line at creation time. If this whole
# project folder was copied from another machine (e.g. scp/rsync/git of the full tree including
# a pre-existing env/), that venv's shebangs point to a path that doesn't exist here, and
# "cannot execute: required file not found" is the result. Detect that and recreate instead of
# trusting the directory's mere presence.
venv_is_usable() {
    [ -x "$VENV_DIR/bin/python3" ] && "$VENV_DIR/bin/python3" -c "" >/dev/null 2>&1
}

if [ -d "$VENV_DIR" ] && venv_is_usable; then
    echo "### Venv already exists at $VENV_DIR and works here, reusing it."
else
    if [ -d "$VENV_DIR" ]; then
        echo "### Venv at $VENV_DIR exists but isn't usable on this machine (likely copied from"
        echo "### another machine -- venvs aren't portable). Recreating it."
        rm -rf "$VENV_DIR"
    fi
    echo "### Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "### Using $(python3 --version) from $(command -v python3)"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: requirements.txt not found at $REQUIREMENTS_FILE" >&2
    exit 1
fi

echo "### Installing Python dependencies from $REQUIREMENTS_FILE"
pip install --upgrade pip >/dev/null
pip install -r "$REQUIREMENTS_FILE"

# --- Java CSP model ------------------------------------------------------------------
MODEL_DIR="$SCRIPT_DIR/utils/model"
echo "### Compiling Java CSP model in $MODEL_DIR"
mkdir -p "$MODEL_DIR/bin/main"
javac -d "$MODEL_DIR/bin/main" -cp "$MODEL_DIR/lib/*" "$MODEL_DIR/src/main/MainOnline.java"
javac -d "$MODEL_DIR/bin/main" -cp "$MODEL_DIR/lib/*" "$MODEL_DIR/src/main/MainIncremental.java"

echo
echo "### Setup complete."
echo "### To run experiments in this shell:"
echo "###   source $VENV_DIR/bin/activate"
echo "###   cd $SCRIPT_DIR"
echo "###   python3 exps/xp_online_grid5000.py --approach online --instance-dir <dir> \\"
echo "###       --nb-jobs <N> --nb-nodes <M> --solver-time-limit <S> --lambda-rate <L>"
