 

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


# =============================================================================
# SETTINGS
# =============================================================================

MODELS = ("BD", "BDEI", "BDSS", "BiSSE")

N_PER_MODEL = 100_000
CHUNK_SIZE = 1_000
MAX_TIME = 500

PARAM_SEED = 20261009
SIM_SEED = 20262009

MODEL_OFFSET = {
    "BD": 101,
    "BDEI": 202,
    "BDSS": 303,
    "BiSSE": 404,
}

BISSE_SEED = 12345
BISSE_STEP = 10
BISSE_RETRIALS = 100


# =============================================================================
# GENERAL
# =============================================================================

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def banner(x):
    print()
    print("=" * 110)
    print(x)
    print("=" * 110)


def count_lines(path):
    path = Path(path)
    if not path.exists():
        return 0

    with path.open("rb") as f:
        return sum(1 for _ in f)


def run(cmd, cwd=None, stdout=None, capture=False):
    print(
        "[CMD]",
        " ".join(shlex.quote(str(x)) for x in cmd),
        flush=True,
    )

    return subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE if capture else stdout,
        stderr=subprocess.PIPE if capture else None,
        text=capture,
        check=True,
    )


# =============================================================================
# WORKSPACE
# =============================================================================

class Workspace:

    def __init__(self, root):

        self.root = Path(root).resolve()

        self.phylognn = self.root / "PhyloGNN"
        self.phylodeep = self.root / "phylodeep"
        self.deeptime = self.root / "deeptimelearning"

        self.param_generator = (
            self.phylognn
            / "Simulations"
            / "generate_parameters.py"
        )

        self.sim_original = {
            "BD":
                self.phylodeep
                / "simulators"
                / "bd_models"
                / "TreeGen_BD_refactored.py",

            "BDEI":
                self.phylodeep
                / "simulators"
                / "bd_models"
                / "TreeGen_BDEI_refactored.py",

            "BDSS":
                self.phylodeep
                / "simulators"
                / "bd_models"
                / "TreeGen_BDSS_refactored.py",

            "BiSSE":
                self.deeptime
                / "simulators"
                / "BiSSE"
                / "BiSSE_simulator.R",
        }

        self.work = self.root / ".step1_generate_data"

        self.logs = self.work / "logs"

        self.chunks = self.work / "chunks"

        self.patched = self.work / "patched_simulators"

        self.task_map = self.work / "task_map.tsv"

        self.jobs = self.work / "jobs.json"

        self.data = self.phylognn / "data"

    def params(self, model):

        names = {
            "BD": "parameters_BD.txt",
            "BDEI": "parameters_BDEI.txt",
            "BDSS": "parameters_BDSS.txt",
            "BiSSE": "parameters_BiSSE_prior_draws.txt",
        }

        return self.data / model / names[model]

    def patched_simulator(self, model):

        if model == "BiSSE":
            return self.sim_original[model]

        return self.patched / self.sim_original[model].name


def get_workspace():

    cwd = Path.cwd()

    if cwd.name == "PhyloGNN":
        cwd = cwd.parent

    return Workspace(cwd)


# =============================================================================
# ENVIRONMENT
# =============================================================================

def check_environment(ws):

    banner("CHECK ENVIRONMENT")

    for d in [
        ws.phylognn,
        ws.phylodeep,
        ws.deeptime,
    ]:

        if not d.exists():
            raise SystemExit(
                f"[ERROR] Missing directory: {d}"
            )

    if not ws.param_generator.exists():
        raise SystemExit(
            f"[ERROR] Missing parameter generator: "
            f"{ws.param_generator}"
        )

    for model, script in ws.sim_original.items():

        if not script.exists():

            raise SystemExit(
                f"[ERROR] Missing {model} simulator: {script}"
            )

    import numpy
    import pandas
    import ete3
    import pyDOE

    print("[OK] Python :", sys.executable)
    print("[OK] Python version:", sys.version)
    print("[OK] NumPy :", numpy.__version__)
    print("[OK] pandas:", pandas.__version__)
    print("[OK] ete3  :", ete3.__version__)
    print("[OK] pyDOE :", pyDOE.__version__)

    if not shutil.which("sbatch"):
        raise SystemExit("[ERROR] sbatch not found")

    if not shutil.which("Rscript"):
        raise SystemExit("[ERROR] Rscript not found")


# =============================================================================
# PATCH OLD NUMPY CODE
# =============================================================================

def patch_one_file(src, dst):

    text = src.read_text()

    original = text

    # -------------------------------------------------------------------------
    # IMPORTANT FIX
    #
    # OLD:
    #
    # int(np.floor(np.random.uniform(0, X, 1)))
    #
    # NumPy 2.x returns a one-element array.
    # Python 3.14 refuses:
    #
    # int(array([x]))
    #
    # NEW:
    #
    # int(np.floor(np.random.uniform(0, X)))
    #
    # which produces a scalar.
    # -------------------------------------------------------------------------

    pattern = re.compile(
        r"""
        int
        \(
            \s*
            np\.floor
            \(
                \s*
                np\.random\.uniform
                \(
                    (?P<args>.*?)
                    ,
                    \s*1\s*
                \)
                \s*
            \)
            \s*
        \)
        """,
        re.VERBOSE | re.DOTALL,
    )

    def replace(match):

        args = match.group("args").rstrip()

        return (
            "int(np.floor("
            "np.random.uniform("
            f"{args}"
            ")"
            "))"
        )

    text, number = pattern.subn(replace, text)

    # -------------------------------------------------------------------------
    # Additional modern NumPy compatibility.
    #
    # Catch simple:
    #
    # int(np.random.uniform(..., 1))
    #
    # -------------------------------------------------------------------------

    pattern2 = re.compile(
        r"""
        int
        \(
            \s*
            np\.random\.uniform
            \(
                (?P<args>.*?)
                ,
                \s*1\s*
            \)
            \s*
        \)
        """,
        re.VERBOSE | re.DOTALL,
    )

    def replace2(match):

        args = match.group("args").rstrip()

        return (
            "int("
            "np.random.uniform("
            f"{args}"
            ")"
            ")"
        )

    text, number2 = pattern2.subn(
        replace2,
        text,
    )

    dst.write_text(text)

    print()
    print("SOURCE :", src)
    print("PATCHED:", dst)

    print(
        "PATCHES:",
        number + number2,
    )

    # Print the specific previously-broken line(s).
    print()
    print("Relevant patched lines:")

    for i, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        if (
            "number_infectious_leaves"
            in line
            and "np.random.uniform"
            in line
        ):

            print(f"{i}: {line}")

    return number + number2


def patch_simulators(ws):

    banner(
        "PATCH PHYLODEEP FOR NUMPY 2.x / PYTHON 3.14"
    )

    ws.patched.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0

    for model in [
        "BD",
        "BDEI",
        "BDSS",
    ]:

        total += patch_one_file(
            ws.sim_original[model],
            ws.patched_simulator(model),
        )

    print()
    print(
        "[OK] Total compatibility patches:",
        total,
    )

    if total == 0:

        raise SystemExit(
            "[ERROR] No NumPy compatibility "
            "patterns were found."
        )


# =============================================================================
# PARAMETER GENERATION
# =============================================================================

def seeded_python(
    script,
    seed,
    args,
):

    code = (
        "import numpy as np,runpy,sys;"
        "np.random.seed(int(sys.argv[1]));"
        "s=sys.argv[2];"
        "sys.argv=[s]+sys.argv[3:];"
        "runpy.run_path(s,run_name=\"__main__\")"
    )

    return [
        sys.executable,
        "-c",
        code,
        str(seed),
        str(script),
        *[str(x) for x in args],
    ]


def parameter_args(
    model,
    n,
    output,
):

    if model == "BD":

        return [
            "-m", "BD",
            "-r", "1,5",
            "-i", "1,10",
            "-s", "200,500",
            "-p", "0.01,1",
            "-n", str(n),
            "-o", str(output),
        ]

    if model == "BDEI":

        return [
            "-m", "BDEI",
            "-r", "1,5",
            "-i", "1,10",
            "-e", "0.2,5",
            "-s", "200,500",
            "-p", "0.01,1",
            "-n", str(n),
            "-o", str(output),
        ]

    if model == "BDSS":

        return [
            "-m", "BDSS",
            "-r", "1,5",
            "-i", "1,10",
            "-x", "3,10",
            "-f", "0.05,0.2",
            "-s", "200,500",
            "-p", "0.01,1",
            "-n", str(n),
            "-o", str(output),
        ]

    if model == "BiSSE":

        return [
            "-m", "BISSE",
            "-l0", "0.01,1",
            "-t", "0,1",
            "-l1", "0.1,1",
            "-q", "0.01,0.1",
            "-s", "200,500",
            "-p", "0.01,1",
            "-n", str(n),
            "-o", str(output),
        ]

    raise ValueError(model)


def create_parameters(
    ws,
    n,
):

    banner(
        "GENERATE PARAMETER DESIGNS"
    )

    for model in MODELS:

        out = ws.params(model)

        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        seed = (
            PARAM_SEED
            + MODEL_OFFSET[model]
        )

        print()
        print(
            f"[{model}] "
            f"Generating {n:,} parameters"
        )

        run(
            seeded_python(
                ws.param_generator,
                seed,
                parameter_args(
                    model,
                    n,
                    out,
                ),
            ),
            cwd=out.parent,
        )

        rows = count_lines(out)

        expected = (
            n
            if model == "BiSSE"
            else n + 1
        )

        if rows != expected:

            raise SystemExit(
                f"[ERROR] {model} "
                f"parameter rows = {rows}, "
                f"expected = {expected}"
            )

        print(
            f"[PASS] {model}: "
            f"{n:,} parameter sets"
        )


# =============================================================================
# SINGLE SIMULATION TEST
# =============================================================================

def test_one(
    ws,
    model,
):

    check_environment(ws)

    ws.work.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_simulators(ws)

    banner(
        f"TEST EXACTLY ONE {model} SIMULATION"
    )

    with tempfile.TemporaryDirectory(
        prefix=f"test_{model}_",
        dir=str(ws.work),
    ) as td:

        td = Path(td)

        params = (
            td
            / "params.tsv"
        )

        seed = (
            PARAM_SEED
            + MODEL_OFFSET[model]
        )

        run(
            seeded_python(
                ws.param_generator,
                seed,
                parameter_args(
                    model,
                    1,
                    params,
                ),
            ),
            cwd=td,
        )

        print()
        print(
            "Parameter file:"
        )

        print(
            params.read_text()
        )

        # =====================================================================
        # BiSSE
        # =====================================================================

        if model == "BiSSE":

            trees = (
                td
                / "trees.nwk"
            )

            stats = (
                td
                / "stats.txt"
            )

            final_params = (
                td
                / "params_final.txt"
            )

            run(
                [
                    "Rscript",
                    str(
                        ws.sim_original[
                            "BiSSE"
                        ]
                    ),
                    str(params),
                    "1",
                    str(BISSE_SEED),
                    str(BISSE_STEP),
                    str(BISSE_RETRIALS),
                    str(trees),
                    str(stats),
                    str(final_params),
                ],
                cwd=td,
            )

            generated = min(
                count_lines(trees),
                count_lines(stats),
                count_lines(final_params),
            )

        # =====================================================================
        # BD / BDEI / BDSS
        # =====================================================================

        else:

            trees = (
                td
                / "trees.tsv"
            )

            seed = (
                SIM_SEED
                + MODEL_OFFSET[model]
            )

            with trees.open("w") as fout:

                run(
                    seeded_python(
                        ws.patched_simulator(
                            model
                        ),
                        seed,
                        [
                            params,
                            MAX_TIME,
                        ],
                    ),
                    cwd=td,
                    stdout=fout,
                )

            stats = (
                td
                / "subpopulations.txt"
            )

            print()
            print(
                "tree file lines:",
                count_lines(trees),
            )

            print(
                "stats file lines:",
                count_lines(stats),
            )

            # simulator has headers
            tree_rows = max(
                0,
                count_lines(trees) - 1,
            )

            stat_rows = max(
                0,
                count_lines(stats) - 1,
            )

            generated = min(
                tree_rows,
                stat_rows,
            )

        print()

        if generated == 1:

            banner(
                f"SUCCESS — ONE {model} DATASET GENERATED"
            )

            print(
                trees.read_text(
                    errors="replace"
                )[:3000]
            )

        else:

            raise SystemExit(
                f"[FAILED] {model}: "
                f"expected 1 generated dataset, "
                f"got {generated}"
            )


# =============================================================================
# SPLIT PARAMETERS
# =============================================================================

def split_parameter_file(
    model,
    src,
    destination,
):

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for x in destination.glob(
        "params_*.tsv"
    ):

        x.unlink()

    header = (
        model != "BiSSE"
    )

    chunks = []

    with src.open() as fin:

        first = (
            fin.readline()
            if header
            else None
        )

        if not header:
            fin.seek(0)

        out = None

        count = 0

        try:

            for line in fin:

                if (
                    count
                    % CHUNK_SIZE
                    == 0
                ):

                    if out:
                        out.close()

                    idx = (
                        count
                        // CHUNK_SIZE
                    )

                    path = (
                        destination
                        / f"params_{idx:05d}.tsv"
                    )

                    out = (
                        path.open("w")
                    )

                    if first:
                        out.write(first)

                    chunks.append(path)

                out.write(line)

                count += 1

        finally:

            if out:
                out.close()

    return chunks


def build_tasks(ws):

    banner(
        "BUILD SIMULATION TASKS"
    )

    tasks = []

    task_id = 0

    for model in MODELS:

        chunks = split_parameter_file(
            model,
            ws.params(model),
            ws.chunks / model,
        )

        print(
            f"{model}: "
            f"{len(chunks)} chunks"
        )

        for i, path in enumerate(
            chunks
        ):

            tasks.append(
                (
                    task_id,
                    model,
                    i,
                    path.resolve(),
                )
            )

            task_id += 1

    ws.task_map.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ws.task_map.open("w") as f:

        f.write(
            "task_id\tmodel\tchunk\tparameter_file\n"
        )

        for (
            task_id,
            model,
            chunk,
            path,
        ) in tasks:

            f.write(
                f"{task_id}\t"
                f"{model}\t"
                f"{chunk}\t"
                f"{path}\n"
            )

    print()
    print(
        "[OK] total tasks:",
        len(tasks),
    )


def read_tasks(ws):

    if not ws.task_map.exists():
        return []

    out = []

    with ws.task_map.open() as f:

        next(f)

        for line in f:

            if not line.strip():
                continue

            task, model, chunk, path = (
                line.rstrip().split(
                    "\t",
                    3,
                )
            )

            out.append(
                (
                    int(task),
                    model,
                    int(chunk),
                    Path(path),
                )
            )

    return out


def expected_rows(
    model,
    parameter_file,
):

    n = count_lines(
        parameter_file
    )

    if model == "BiSSE":
        return n

    return max(
        0,
        n - 1,
    )


# =============================================================================
# OUTPUT PATHS
# =============================================================================

def output_dir(
    ws,
    model,
    chunk,
):

    return (
        ws.chunks
        / model
        / f"out_{chunk:05d}"
    )


def verified_rows(
    ws,
    model,
    chunk,
    parameter_file,
):

    d = output_dir(
        ws,
        model,
        chunk,
    )

    done = (
        d
        / "DONE.json"
    )

    if not done.exists():
        return 0

    expected = expected_rows(
        model,
        parameter_file,
    )

    try:
        metadata = json.loads(
            done.read_text()
        )

    except Exception:
        return 0

    if (
        metadata.get(
            "verified_rows"
        )
        != expected
    ):

        return 0

    return expected


# =============================================================================
# WORKER
# =============================================================================

def worker(
    ws,
    task_id,
):

    task_dict = {
        x[0]: x
        for x in read_tasks(ws)
    }

    (
        _,
        model,
        chunk,
        parameter_file,
    ) = task_dict[task_id]

    expected = expected_rows(
        model,
        parameter_file,
    )

    d = output_dir(
        ws,
        model,
        chunk,
    )

    d.mkdir(
        parents=True,
        exist_ok=True,
    )

    done = d / "DONE.json"

    failed = d / "FAILED.json"

    # Already verified
    if (
        verified_rows(
            ws,
            model,
            chunk,
            parameter_file,
        )
        == expected
    ):

        print(
            "[SKIP]",
            model,
            chunk,
        )

        return

    # Remove partial files
    for x in d.iterdir():

        if x.is_file():
            x.unlink()

    try:

        # =====================================================================
        # BISSE
        # =====================================================================

        if model == "BiSSE":

            trees = (
                d
                / "trees.nwk"
            )

            stats = (
                d
                / "stats.txt"
            )

            final_params = (
                d
                / "params_final.txt"
            )

            run(
                [
                    "Rscript",
                    str(
                        ws.sim_original[
                            "BiSSE"
                        ]
                    ),
                    str(parameter_file),
                    str(chunk + 1),
                    str(BISSE_SEED),
                    str(BISSE_STEP),
                    str(BISSE_RETRIALS),
                    str(trees),
                    str(stats),
                    str(final_params),
                ],
                cwd=d,
            )

            actual = min(
                count_lines(trees),
                count_lines(stats),
                count_lines(final_params),
            )

        # =====================================================================
        # BD / BDEI / BDSS
        # =====================================================================

        else:

            trees = (
                d
                / "trees.tsv"
            )

            seed = (
                SIM_SEED
                + MODEL_OFFSET[
                    model
                ]
                * 100_000
                + chunk
            )

            with trees.open(
                "w"
            ) as fout:

                run(
                    seeded_python(
                        ws.patched_simulator(
                            model
                        ),
                        seed,
                        [
                            parameter_file,
                            MAX_TIME,
                        ],
                    ),
                    cwd=d,
                    stdout=fout,
                )

            stats = (
                d
                / "subpopulations.txt"
            )

            actual = min(
                max(
                    0,
                    count_lines(
                        trees
                    )
                    - 1,
                ),
                max(
                    0,
                    count_lines(
                        stats
                    )
                    - 1,
                ),
            )

        if actual != expected:

            raise RuntimeError(
                f"{model} chunk {chunk}: "
                f"expected {expected}, "
                f"generated {actual}"
            )

        metadata = {
            "task_id": task_id,
            "model": model,
            "chunk": chunk,
            "verified_rows": actual,
            "completed": now(),
        }

        done.write_text(
            json.dumps(
                metadata,
                indent=2,
            )
        )

        print(
            f"[PASS] "
            f"{model} chunk {chunk}: "
            f"{actual:,} trees"
        )

    except Exception as e:

        metadata = {
            "task_id": task_id,
            "model": model,
            "chunk": chunk,
            "exception": str(e),
            "traceback":
                traceback.format_exc(),
        }

        failed.write_text(
            json.dumps(
                metadata,
                indent=2,
            )
        )

        raise


# =============================================================================
# STATUS
# =============================================================================

def print_bar(
    model,
    done,
    total,
):

    try:

        from tqdm import tqdm

        bar = tqdm(
            total=total,
            initial=done,
            desc=f"{model:6s}",
            unit="tree",
            dynamic_ncols=True,
        )

        bar.close()

    except ImportError:

        pct = (
            done
            / total
            * 100
            if total
            else 0
        )

        print(
            f"{model:6s}: "
            f"{done:,}/{total:,} "
            f"({pct:.2f}%)"
        )


def status(ws):

    banner(
        "REAL VERIFIED DATA GENERATION STATUS"
    )

    tasks = read_tasks(ws)

    if not tasks:

        print(
            "No task map found."
        )

        return

    done_count = {
        x: 0
        for x in MODELS
    }

    targets = {
        x: 0
        for x in MODELS
    }

    chunks_done = {
        x: 0
        for x in MODELS
    }

    failed = {
        x: 0
        for x in MODELS
    }

    for (
        task_id,
        model,
        chunk,
        parameter_file,
    ) in tasks:

        expected = expected_rows(
            model,
            parameter_file,
        )

        verified = verified_rows(
            ws,
            model,
            chunk,
            parameter_file,
        )

        targets[model] += expected

        done_count[model] += verified

        if (
            verified
            == expected
            and expected > 0
        ):

            chunks_done[
                model
            ] += 1

        if (
            output_dir(
                ws,
                model,
                chunk,
            )
            / "FAILED.json"
        ).exists():

            failed[
                model
            ] += 1

    print()
    print(
        "A tree counts ONLY after the "
        "whole chunk is verified."
    )
    print()

    for model in MODELS:

        print_bar(
            model,
            done_count[model],
            targets[model],
        )

    print()

    print(
        f"{'MODEL':7s}"
        f"{'VERIFIED':>14s}"
        f"{'TARGET':>14s}"
        f"{'CHUNKS':>14s}"
        f"{'FAILED':>10s}"
    )

    print(
        "-" * 70
    )

    for model in MODELS:

        print(
            f"{model:7s}"
            f"{done_count[model]:14,d}"
            f"{targets[model]:14,d}"
            f"{chunks_done[model]:7d}/100"
            f"{failed[model]:10d}"
        )

    total_done = sum(
        done_count.values()
    )

    total_target = sum(
        targets.values()
    )

    print(
        "-" * 70
    )

    print(
        f"{'TOTAL':7s}"
        f"{total_done:14,d}"
        f"{total_target:14,d}"
    )

    print()

    print_bar(
        "TOTAL",
        total_done,
        total_target,
    )


# =============================================================================
# CANCEL / RESET
# =============================================================================

def cancel_jobs():

    banner(
        "CANCEL PHYLOGNN STEP1 JOBS"
    )

    user = os.environ.get(
        "USER"
    )

    result = subprocess.run(
        [
            "squeue",
            "-h",
            "-u",
            user,
            "-o",
            "%A|%j",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )

    ids = []

    for line in (
        result.stdout
        .splitlines()
    ):

        if "|" not in line:
            continue

        job_id, name = (
            line.split(
                "|",
                1,
            )
        )

        if name in [
            "PhyloGNN_S01",
            "PhyloGNN_S01_AGG",
        ]:

            if (
                job_id
                not in ids
            ):

                ids.append(
                    job_id
                )

    if ids:

        print(
            "Cancelling:",
            " ".join(ids),
        )

        subprocess.run(
            [
                "scancel",
                *ids,
            ],
            check=False,
        )

    else:

        print(
            "No active Step1 jobs."
        )


def reset(ws):

    cancel_jobs()

    banner(
        "CLEAR STEP1 CACHE"
    )

    if ws.work.exists():

        print(
            "Removing:",
            ws.work,
        )

        shutil.rmtree(
            ws.work
        )

    for model in MODELS:

        d = (
            ws.data
            / model
        )

        if d.exists():

            print(
                "Removing:",
                d,
            )

            shutil.rmtree(
                d
            )

    print()
    print(
        "[DONE] Step1 cache/data cleared."
    )


# =============================================================================
# SUBMIT
# =============================================================================

def incomplete_tasks(ws):

    ids = []

    for (
        task_id,
        model,
        chunk,
        parameter_file,
    ) in read_tasks(ws):

        expected = expected_rows(
            model,
            parameter_file,
        )

        got = verified_rows(
            ws,
            model,
            chunk,
            parameter_file,
        )

        if got != expected:

            ids.append(
                task_id
            )

    return ids


def compact(ids):

    if not ids:
        return ""

    ids = sorted(
        set(ids)
    )

    out = []

    start = ids[0]

    previous = ids[0]

    for x in ids[1:]:

        if x == previous + 1:

            previous = x

            continue

        if start == previous:

            out.append(
                str(start)
            )

        else:

            out.append(
                f"{start}-{previous}"
            )

        start = x

        previous = x

    if start == previous:

        out.append(
            str(start)
        )

    else:

        out.append(
            f"{start}-{previous}"
        )

    return ",".join(out)


def submit(ws):

    ids = incomplete_tasks(
        ws
    )

    if not ids:

        print(
            "Everything already complete."
        )

        return

    array = compact(
        ids
    )

    banner(
        "SUBMIT FULL SIMULATION ARRAY"
    )

    print(
        "Tasks:",
        len(ids),
    )

    print(
        "Array:",
        array,
    )

    print(
        "NO % concurrency restriction "
        "is added by this script."
    )

    ws.logs.mkdir(
        parents=True,
        exist_ok=True,
    )

    script = (
        Path(__file__)
        .resolve()
    )

    python = (
        Path(sys.executable)
        .resolve()
    )

    command = (
        f"{shlex.quote(str(python))} "
        f"{shlex.quote(str(script))} "
        "--worker "
        "$SLURM_ARRAY_TASK_ID"
    )

    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--job-name",
            "PhyloGNN_S01",
            "--array",
            array,
            "--partition",
            "general",
            "--cpus-per-task",
            "1",
            "--mem",
            "8G",
            "--time",
            "12:00:00",
            "--chdir",
            str(ws.root),
            "--output",
            str(
                ws.logs
                / "%x.%A_%a.out"
            ),
            "--error",
            str(
                ws.logs
                / "%x.%A_%a.err"
            ),
            "--wrap",
            command,
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )

    job = (
        result.stdout
        .strip()
        .split(";")[0]
    )

    print()
    print(
        "[SUBMITTED]",
        job,
    )

    print()
    print(
        "Check REAL progress with:"
    )

    print(
        "python Step1-GenerateData.py --status"
    )


# =============================================================================
# PREPARE FULL RUN
# =============================================================================

def prepare(ws):

    check_environment(
        ws
    )

    if ws.task_map.exists():

        raise SystemExit(
            "\nExisting Step1 run detected.\n"
            "Use:\n"
            "  python Step1-GenerateData.py --resume\n"
            "or reset:\n"
            "  python Step1-GenerateData.py --reset"
        )

    ws.work.mkdir(
        parents=True,
        exist_ok=True,
    )

    ws.logs.mkdir(
        parents=True,
        exist_ok=True,
    )

    ws.chunks.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_simulators(
        ws
    )

    create_parameters(
        ws,
        N_PER_MODEL,
    )

    build_tasks(
        ws
    )

    banner(
        "PRE-SUBMISSION ONE-TREE BD TEST"
    )

    test_one(
        ws,
        "BD",
    )

    banner(
        "ONE-TREE TEST PASSED"
    )

    submit(
        ws
    )


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reset",
        action="store_true",
    )

    parser.add_argument(
        "--status",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--test-one",
        choices=MODELS,
    )

    parser.add_argument(
        "--worker",
        type=int,
    )

    args = parser.parse_args()

    ws = get_workspace()

    if args.reset:

        reset(ws)

        return

    if args.status:

        status(ws)

        return

    if args.test_one:

        test_one(
            ws,
            args.test_one,
        )

        return

    if args.worker is not None:

        # patched simulators must already exist
        worker(
            ws,
            args.worker,
        )

        return

    if args.resume:

        check_environment(
            ws
        )

        patch_simulators(
            ws
        )

        submit(
            ws
        )

        return

    prepare(
        ws
    )


if __name__ == "__main__":
    main()
 