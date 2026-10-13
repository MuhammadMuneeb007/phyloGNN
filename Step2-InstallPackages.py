
#!/usr/bin/env python3
"""
Step2-InstallPackages.py

Install original PhyloCNN dependencies plus:
    - TensorFlow / Keras
    - Spektral
    - PyTorch
    - PyTorch Geometric

All Python frameworks share ONE environment:
    .envs/phylocnn

R dependencies are installed separately:
    .envs/phylo-r

Designed for Linux HPC and Python 3.8.
Existing environments are reused.
"""

from pathlib import Path
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path.cwd().resolve()

ENV_DIR = ROOT / ".envs"
CACHE_DIR = ROOT / ".cache"
TMP_DIR = ROOT / ".tmp"

PY_PREFIX = ENV_DIR / "phylocnn"
R_PREFIX = ENV_DIR / "phylo-r"

SOURCE = ROOT / "phyloCNN" / "environment.yml"
REQ_FILE = ROOT / "Step2-PhyloCNN-Linux-requirements.txt"

PYTHON_VERSION = "3.8.20"

# Python 3.8-compatible framework versions.
# Retain original TensorFlow/Keras/NumPy pins.
EXTRA_PACKAGES = [
    "spektral==1.3.1",
    "torch==2.4.1",
    "torch-geometric==2.6.1",
]

EXCLUDED_PACKAGES = {
    "tensorflow-macos",
    "tensorflow-metal",
    "appnope",
}


def section(title):
    print("\n" + "=" * 70, flush=True)
    print(title, flush=True)
    print("=" * 70, flush=True)


def run(command, env):
    args = [str(x) for x in command]
    print("$ " + shlex.join(args), flush=True)
    subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=True,
    )


def configure_local_storage():
    """Keep environments, caches, and temporary files local."""

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


def package_name(requirement):
    """Extract normalized package name."""

    name = re.split(
        r"[=<>!~\[\s]",
        requirement,
        maxsplit=1,
    )[0]

    return name.lower().replace("_", "-")


def read_requirements():
    """Read original PhyloCNN requirements and add GNN frameworks."""

    if not SOURCE.is_file():
        sys.exit(
            "ERROR: Missing original environment file: "
            + str(SOURCE)
        )

    requirements = []
    inside_pip = False

    for line in SOURCE.read_text().splitlines():

        if re.match(r"^\s+-\s+pip:\s*$", line):
            inside_pip = True
            continue

        if not inside_pip:
            continue

        match = re.match(
            r"^\s{6,}-\s+(\S.*?)\s*$",
            line,
        )

        if match:
            requirements.append(match.group(1))

        elif line.strip() and not line.lstrip().startswith("#"):
            break

    if len(requirements) < 30:
        sys.exit(
            "ERROR: Could not read original PhyloCNN dependencies."
        )

    linux_requirements = []

    for requirement in requirements:
        name = package_name(requirement)

        if name in EXCLUDED_PACKAGES:
            print("[SKIP macOS]", requirement)
            continue

        linux_requirements.append(requirement)

    # Replace existing specifications for extra frameworks
    # to avoid duplicate or conflicting direct requirements.
    extra_names = {
        package_name(req)
        for req in EXTRA_PACKAGES
    }

    linux_requirements = [
        req for req in linux_requirements
        if package_name(req) not in extra_names
    ]

    linux_requirements.extend(EXTRA_PACKAGES)

    REQ_FILE.write_text(
        "\n".join(linux_requirements) + "\n"
    )

    print("Original packages:", len(requirements))
    print("Linux packages:", len(linux_requirements))
    print("Additional GNN packages:")

    for package in EXTRA_PACKAGES:
        print("  +", package)

    print("Saved requirements:", REQ_FILE)


def create_environment(
    manager,
    env,
    prefix,
    executable,
    packages,
):
    """Create or reuse a local Conda environment."""

    if executable.is_file():
        print(
            "[REUSE] Existing environment:",
            prefix,
            flush=True,
        )
        return

    if prefix.exists():
        sys.exit(
            "ERROR: Incomplete environment exists: "
            + str(prefix)
        )

    command = [
        manager,
        "create",
        "-y",
        "--prefix",
        prefix,
        "--override-channels",
        "-c",
        "conda-forge",
        *packages,
    ]

    run(command, env)

    if not executable.is_file():
        sys.exit(
            "ERROR: Environment creation failed: "
            + str(prefix)
        )


def install_python_packages(python, env):
    """Install all packages into the same PhyloCNN environment."""

    uv = shutil.which("uv")

    if uv:
        print("[INSTALLER] uv detected")

        command = [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "-r",
            REQ_FILE,
        ]

    else:
        print("[INSTALLER] Using pip")

        command = [
            python,
            "-m",
            "pip",
            "install",
            "-r",
            REQ_FILE,
        ]

    run(command, env)


def verify_python_packages(python, env):
    """Verify package versions, imports, and basic GNN execution."""

    section("VERIFY PYTHON DEPENDENCIES")

    run([
        python,
        "-m",
        "pip",
        "check",
    ], env)

    verification = r'''
import importlib.metadata as metadata

packages = [
    "tensorflow",
    "keras",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "ete3",
    "phylodeep",
    "spektral",
    "torch",
    "torch-geometric",
]

print("\nINSTALLED PACKAGE VERSIONS")
print("-" * 55)

for package in packages:
    print(
        f"{package:20s}",
        metadata.version(package)
    )

print("\nVERIFYING TENSORFLOW")

import tensorflow as tf

print("TensorFlow:", tf.__version__)

print("\nVERIFYING SPEKTRAL")

import spektral

from spektral.layers import (
    GCNConv,
    GATConv,
    GINConv,
    ECCConv,
    GraphSageConv,
)

print("Spektral:", spektral.__version__)
print("Spektral GNN layers imported successfully")

print("\nVERIFYING PYTORCH")

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("\nVERIFYING PYTORCH GEOMETRIC")

import torch_geometric

from torch_geometric.nn import (
    GCNConv as PyG_GCNConv,
    GATConv as PyG_GATConv,
    GINConv as PyG_GINConv,
    SAGEConv,
    GINEConv,
)

print("PyG:", torch_geometric.__version__)

# Small CPU graph test
x = torch.randn(4, 8)

edge_index = torch.tensor(
    [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
    ],
    dtype=torch.long,
)

model = PyG_GCNConv(8, 16)

output = model(x, edge_index)

assert output.shape == (4, 16)

print("PyTorch Geometric graph test PASSED")

print("\n" + "=" * 55)
print("ALL PYTHON FRAMEWORKS VERIFIED")
print("=" * 55)
'''

    run([
        python,
        "-c",
        verification,
    ], env)


def install_r_packages(manager, env):
    """Install R packages needed for BiSSE."""

    section("INSTALL R PACKAGES FOR BISSE")

    rscript = R_PREFIX / "bin" / "Rscript"

    create_environment(
        manager,
        env,
        R_PREFIX,
        rscript,
        [
            "r-base=4.4",
            "r-castor",
            "r-treesim",
        ],
    )

    run([
        rscript,
        "-e",
        (
            'library(castor); '
            'library(TreeSim); '
            'cat("BiSSE R packages OK\\n")'
        ),
    ], env)


def main():

    section("STEP 2: INSTALL PHYLOCNN + SPEKTRAL + PYTORCH GEOMETRIC")

    print("CURRENT DIRECTORY:", ROOT)

    if not sys.platform.startswith("linux"):
        sys.exit(
            "ERROR: This installer is designed for Linux HPC."
        )

    # Verify repositories.
    for repo in (
        "phyloCNN",
        "phylodeep",
        "deeptimelearning",
    ):
        path = ROOT / repo

        if not path.is_dir():
            sys.exit(
                "ERROR: Missing repository: "
                + str(path)
            )

        print("[FOUND]", repo)

    manager = next(
        (
            shutil.which(name)
            for name in (
                "mamba",
                "micromamba",
                "conda",
            )
            if shutil.which(name)
        ),
        None,
    )

    if not manager:
        sys.exit(
            "ERROR: Mamba, Micromamba, or Conda is required."
        )

    env = configure_local_storage()

    section("1/5 - READ ORIGINAL REQUIREMENTS")

    read_requirements()

    section("2/5 - CREATE OR REUSE PYTHON ENVIRONMENT")

    python = PY_PREFIX / "bin" / "python"

    create_environment(
        manager,
        env,
        PY_PREFIX,
        python,
        [
            f"python={PYTHON_VERSION}",
            "pip=24.2",
        ],
    )

    section("3/5 - INSTALL ALL PYTHON FRAMEWORKS")

    install_python_packages(python, env)

    section("4/5 - VERIFY ALL PYTHON FRAMEWORKS")

    verify_python_packages(python, env)

    section("5/5 - INSTALL R PACKAGES")

    install_r_packages(manager, env)

    section("STEP 2 COMPLETED SUCCESSFULLY")

    print("Working directory:", ROOT)
    print("Python environment:", PY_PREFIX)
    print("R environment:", R_PREFIX)
    print("Package caches:", CACHE_DIR)
    print("Temporary files:", TMP_DIR)

    print("\nInstalled frameworks:")
    print("  TensorFlow / Keras")
    print("  Spektral")
    print("  PyTorch")
    print("  PyTorch Geometric")

    print("\nTo use the environment:")
    print(f"  conda activate {PY_PREFIX}")

    print("\nOr execute Python directly:")
    print(f"  {PY_PREFIX}/bin/python")


if __name__ == "__main__":
    try:
        main()

    except subprocess.CalledProcessError as error:
        sys.exit(
            "INSTALLATION FAILED "
            f"(exit code {error.returncode})"
        )
