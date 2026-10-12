
#!/usr/bin/env python3
"""Install PhyloCNN dependencies in the current working directory."""

from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import sys

# Everything is relative to where you RUN this script.
ROOT = Path.cwd().resolve()

ENV_DIR = ROOT / ".envs"
CACHE_DIR = ROOT / ".cache"
TMP_DIR = ROOT / ".tmp"

PY_PREFIX = ENV_DIR / "phylocnn"
R_PREFIX = ENV_DIR / "phylo-r"

SOURCE = ROOT / "phyloCNN" / "environment.yml"
REQ_FILE = ROOT / "Step2-PhyloCNN-Linux-requirements.txt"


def section(title):
    print("\n" + "=" * 65, flush=True)
    print(title, flush=True)
    print("=" * 65, flush=True)


def run(command, env):
    args = [str(x) for x in command]
    print("$ " + shlex.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def configure_local_storage():
    """Store caches, temporary files and environments locally."""

    directories = [
        ENV_DIR,
        CACHE_DIR / "uv",
        CACHE_DIR / "pip",
        CACHE_DIR / "conda-pkgs",
        CACHE_DIR / "xdg",
        TMP_DIR,
        ROOT / ".config",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    env.update({
        "UV_CACHE_DIR": str(CACHE_DIR / "uv"),
        "PIP_CACHE_DIR": str(CACHE_DIR / "pip"),
        "CONDA_PKGS_DIRS": str(CACHE_DIR / "conda-pkgs"),
        "XDG_CACHE_HOME": str(CACHE_DIR / "xdg"),
        "XDG_CONFIG_HOME": str(ROOT / ".config"),
        "TMPDIR": str(TMP_DIR),
        "TMP": str(TMP_DIR),
        "TEMP": str(TMP_DIR),
        "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    })

    return env


def read_requirements():
    """Read the ORIGINAL phyloCNN/environment.yml file."""

    if not SOURCE.is_file():
        sys.exit("ERROR: Missing phyloCNN/environment.yml")

    requirements = []
    inside_pip = False

    for line in SOURCE.read_text().splitlines():

        if re.match(r"^\s+-\s+pip:\s*$", line):
            inside_pip = True
            continue

        if not inside_pip:
            continue

        match = re.match(r"^\s{6,}-\s+(\S.*?)\s*$", line)

        if match:
            requirements.append(match.group(1))
        elif line.strip() and not line.lstrip().startswith("#"):
            break

    if len(requirements) < 30:
        sys.exit("ERROR: Could not read PhyloCNN dependencies.")

    excluded = {"tensorflow-macos", "appnope"}
    linux_requirements = []

    for requirement in requirements:
        name = re.split(
            r"[=<>!~\[]", requirement, maxsplit=1
        )[0].lower().replace("_", "-")

        if name not in excluded:
            linux_requirements.append(requirement)
        else:
            print("[SKIP macOS package]", requirement)

    REQ_FILE.write_text("\n".join(linux_requirements) + "\n")

    print("Original packages:", len(requirements))
    print("Linux packages:", len(linux_requirements))
    print("Saved requirements:", REQ_FILE)


def create_environment(manager, env, prefix, executable, packages):
    """Create environment in current working directory."""

    if executable.is_file():
        print("[REUSE] Existing environment:", prefix, flush=True)
        return

    if prefix.exists():
        sys.exit("ERROR: Incomplete environment: " + str(prefix))

    run([
        manager, "create", "-y",
        "--prefix", prefix,
        "--override-channels",
        "-c", "conda-forge",
        *packages,
    ], env)

    if not executable.is_file():
        sys.exit("ERROR: Environment creation failed.")


def main():

    section("STEP 2: INSTALL PHYLOCNN PACKAGES")

    print("CURRENT DIRECTORY:", ROOT)

    # Verify all three repositories.
    for repo in ("phyloCNN", "phylodeep", "deeptimelearning"):
        if not (ROOT / repo).is_dir():
            sys.exit("ERROR: Missing repository: " + repo)

        print("[FOUND]", repo)

    if not sys.platform.startswith("linux"):
        sys.exit("ERROR: This installer is for Linux HPC.")

    manager = next(
        (shutil.which(x) for x in ("mamba", "micromamba", "conda")
         if shutil.which(x)),
        None
    )

    if not manager:
        sys.exit("ERROR: Mamba or Conda not available.")

    env = configure_local_storage()

    section("1/5 - READ ORIGINAL REQUIREMENTS")

    read_requirements()

    section("2/5 - CREATE LOCAL PYTHON ENVIRONMENT")

    python = PY_PREFIX / "bin" / "python"

    create_environment(
        manager, env, PY_PREFIX, python,
        ["python=3.8.20", "pip=24.2"]
    )

    section("3/5 - INSTALL ALL PYTHON PACKAGES")

    uv = shutil.which("uv")

    if uv:
        run([
            uv, "pip", "install",
            "--python", python,
            "-r", REQ_FILE
        ], env)
    else:
        run([
            python, "-m", "pip", "install",
            "-r", REQ_FILE
        ], env)

    section("4/5 - VERIFY PYTHON PACKAGES")

    run([python, "-m", "pip", "check"], env)

    check = (
    "import importlib.metadata as m; "
    "names=['tensorflow','keras','numpy','pandas',"
    "'scipy','scikit-learn','ete3','phylodeep','spektral']; "
    "[(print(n,m.version(n))) for n in names]; "
    "import spektral; "
    "from spektral.layers import "
    "GCNConv, GATConv, GINConv, ECCConv, GraphSageConv; "
    "print('Spektral GNN layers OK')"
    )

    run([python, "-c", check], env)

    section("5/5 - INSTALL R PACKAGES FOR BISSE")

    rscript = R_PREFIX / "bin" / "Rscript"

    create_environment(
        manager, env, R_PREFIX, rscript,
        ["r-base=4.4", "r-castor", "r-treesim"]
    )

    run([
        rscript, "-e",
        'library(castor); library(TreeSim); '
        'cat("BiSSE R packages OK\\n")'
    ], env)

    section("STEP 2 COMPLETED SUCCESSFULLY")

    print("Working directory:", ROOT)
    print("Python environment:", PY_PREFIX)
    print("R environment:", R_PREFIX)
    print("Package caches:", CACHE_DIR)
    print("Temporary files:", TMP_DIR)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(
            f"INSTALLATION FAILED (exit code {error.returncode})"
        )
