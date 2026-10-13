
#!/usr/bin/env python3
"""
STEP 3 — PhyloCNN data generation, encoding and validation.

Sources:
  phyloCNN/
  phylodeep/
  deeptimelearning/

New runs:
  - Generate 100,000 training/validation trees per model.
  - Extract a nested 10,000-tree subset.
  - Use the authors' original independent test archives.
  - Encode with original PhyloCNN functions.
  - Validate trees, parameters and encodings.

Existing runs:
  - Preserve Data/_step3/tasks.json and task IDs.
  - Completed chunks are verified and skipped.
  - Legacy generated-test tasks are ignored.

This script never calls sbatch. The SLURM array is Step3-GenerateData.sh;
each array element runs ONE work item:

  python Step3-GenerateData.py --task-id N

  work items 0 .. T-1   training/validation simulation chunks (T = 800 by default)
  work items T .. T+3   encoding of the authors' test sets (BD, BDEI, BDSS, BiSSE)

The first array task prepares the run (parameter designs, chunks, task list)
under a lock; the others wait for it. Preparation can also be run by hand.

Commands (run from the project root):
  python Step3-GenerateData.py --prepare          # optional; the array does it too
  python Step3-GenerateData.py --pilot            # one tree per model, before sbatch
  python Step3-GenerateData.py --array-range      # e.g. 0-803
  sbatch Step3-GenerateData.sh
  python Step3-GenerateData.py --status
  python Step3-GenerateData.py --pending-array    # unfinished items, for a resume
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback

from contextlib import ExitStack
from pathlib import Path


# ============================================================
# 1. CONFIGURATION — NO HARDCODED ABSOLUTE PATHS
# ============================================================

ROOT = Path.cwd().resolve()
SCRIPT = Path(__file__).resolve()

DATA = ROOT / "Data"
CONTROL = DATA / "_step3"

CONFIG = CONTROL / "config.json"
TASKS = CONTROL / "tasks.json"
PREPARE_LOCK = CONTROL / "prepare.lock"
PREPARE_FAILED = CONTROL / "PREPARE_FAILED.json"

# Bunya: array indices must be 0..999.
MAX_ARRAY_INDEX = 999

# A parameter row whose tree fails validation is re-simulated with new seeds.
# Attempts 1-5 use the same seeds as before, so earlier rescues are unchanged.
RESCUE_ATTEMPTS = 20

# Rows that still fail are excluded (recorded in DONE.json). More than this
# share of a chunk failing means something is systematically wrong.
MAX_EXCLUDED_FRACTION = 0.02

LOGS = CONTROL / "logs"
SCRATCH = CONTROL / "scratch"

MODELS = ("BD", "BDEI", "BDSS", "BiSSE")

# Keep original ordering for compatibility with running jobs.
SPLITS = ("trainval", "test")

DIMS = {
    "BD": 18,
    "BDEI": 18,
    "BDSS": 18,
    "BiSSE": 19,
}

BISSE_COLS = (
    "index", "lambda1", "lambda2", "turnover",
    "sampling_frac", "tree_size", "mu1", "mu2",
    "net_rate1", "net_rate2", "q01", "q10",
    "lambda2_ratio", "q01_ratio"
)

BISSE_STATS = (
    "index", "tip_state_1", "tip_state_2",
    "total_tips", "root_state", "tree_height",
    "simulation_retries"
)

TEST_ARCHIVES = {
    "BD": (
        "BD.csv.gz",
        "BD_1.nwk.gz",
        "BD_2.nwk.gz",
    ),
    "BDEI": (
        "BDEI.csv.gz",
        "BDEI_1.nwk.gz",
        "BDEI_2.nwk.gz",
    ),
    "BDSS": (
        "BDSS.csv.gz",
        "BDSS_1.nwk.gz",
        "BDSS_2.nwk.gz",
    ),
    "BiSSE": (
        "BiSSE.csv.gz",
        "BiSSE.nwk.gz",
    ),
}

SEEDED = (
    "import numpy as np,runpy,sys;"
    "np.random.seed(int(sys.argv[1]));"
    "p=sys.argv[2];sys.argv=[p]+sys.argv[3:];"
    "runpy.run_path(p,run_name='__main__')"
)


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def log(*args):
    timestamp = dt.datetime.now().strftime("%H:%M:%S")
    print(timestamp, *args, flush=True)


def shell(command, cwd=None, stdout=None):
    command = [str(x) for x in command]

    log("$", shlex.join(command))

    subprocess.run(
        command,
        cwd=str(cwd or ROOT),
        stdout=stdout,
        check=True,
    )


def json_save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".writing")

    temporary.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n"
    )

    temporary.replace(path)


def json_read(path):
    return json.loads(path.read_text())


def hash_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)

    return digest.hexdigest()


def idx(value):
    number = float(value)

    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid index: {value}")

    return int(number)


# ============================================================
# 3. ORIGINAL REPOSITORY PATHS
# ============================================================

def source_paths():
    return {
        "params":
            ROOT / "phyloCNN/Simulations/generate_parameters.py",

        "BD":
            ROOT / "phylodeep/simulators/bd_models/TreeGen_BD_refactored.py",

        "BDEI":
            ROOT / "phylodeep/simulators/bd_models/TreeGen_BDEI_refactored.py",

        "BDSS":
            ROOT / "phylodeep/simulators/bd_models/TreeGen_BDSS_refactored.py",

        "BiSSE":
            ROOT / "deeptimelearning/simulators/BiSSE/BiSSE_simulator.R",

        "encode_epi":
            ROOT / "phyloCNN/Encoding/PhyloCNN_Encoding_PhyloDyn.py",

        "encode_bisse":
            ROOT / "phyloCNN/Encoding/PhyloCNN_Encoding_BiSSE.py",
    }


def py():
    executable = ROOT / ".envs/phylocnn/bin/python"

    if not executable.is_file():
        raise RuntimeError(
            "Missing .envs/phylocnn/bin/python. Run Step 2."
        )

    return executable


def rscript():
    executable = ROOT / ".envs/phylo-r/bin/Rscript"

    if not executable.is_file():
        raise RuntimeError(
            "Missing .envs/phylo-r/bin/Rscript. Run Step 2."
        )

    return executable


def check_sources(conf=None):
    py()
    rscript()

    for name, path in source_paths().items():
        if not path.is_file():
            raise RuntimeError(f"Missing {name}: {path}")

        if conf:
            expected = conf["source_sha256"][name]

            if hash_file(path) != expected:
                raise RuntimeError(
                    f"Repository source changed: {path}"
                )

    if conf:
        patch = CONTROL / "code/BiSSE_simulator_per_row_seed.R"

        if hash_file(patch) != conf["bisse_patch_sha256"]:
            raise RuntimeError(
                "BiSSE compatibility source changed."
            )


def check_author_archives():
    missing = []

    for model, names in TEST_ARCHIVES.items():
        for name in names:
            path = ROOT / "phyloCNN/Test_Sets" / model / name

            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))

    if missing:
        raise RuntimeError(
            "Missing original test archives:\n"
            + "\n".join(missing)
        )


# ============================================================
# 4. ORIGINAL PARAMETER RANGES
# ============================================================

def param_options(model, n, output):
    options = {
        "BD": [
            "-m", "BD",
            "-r", "1,5",
            "-i", "1,10",
            "-s", "200,500",
            "-p", "0.01,1",
        ],

        "BDEI": [
            "-m", "BDEI",
            "-r", "1,5",
            "-i", "1,10",
            "-e", "0.2,5",
            "-s", "200,500",
            "-p", "0.01,1",
        ],

        "BDSS": [
            "-m", "BDSS",
            "-r", "1,5",
            "-i", "1,10",
            "-x", "3,10",
            "-f", "0.05,0.2",
            "-s", "200,500",
            "-p", "0.01,1",
        ],

        "BiSSE": [
            "-m", "BISSE",
            "-l0", "0.01,1",
            "-t", "0,1",
            "-l1", "0.1,1",
            "-q", "0.01,0.1",
            "-s", "200,500",
            "-p", "0.01,1",
        ],
    }

    return options[model] + [
        "-n", str(n),
        "-o", str(output),
    ]


def seeded_cmd(path, seed, argv):
    return [
        py(),
        "-c",
        SEEDED,
        str(seed),
        path,
        *argv,
    ]


def patch_bisse_seed_copy():
    """
    Copy the original R simulator.

    Change only its per-row random-seed calculation.
    Never modify the original Git repository.
    """

    source = source_paths()["BiSSE"]

    destination = (
        CONTROL / "code/BiSSE_simulator_per_row_seed.R"
    )

    text = source.read_text()

    pattern = (
        r"set\.seed\(\s*seed_base\s*\+\s*indice\s*\*"
        r"\s*step\s*\+\s*3\s*\+\s*j\s*\)"
    )

    replacement = (
        "set.seed(seed_base + "
        "as.integer(params[i,1]) * step + 3 + j)"
    )

    updated, count = re.subn(
        pattern, replacement, text
    )

    if count != 2:
        raise RuntimeError(
            "Unexpected original BiSSE seed implementation."
        )

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    destination.write_text(updated)

    return destination


# ============================================================
# 5. PARAMETER FILE HANDLING
# ============================================================

def design_rows(path, model):
    with path.open(newline="") as f:
        if model == "BiSSE":
            reader = csv.DictReader(
                f,
                delimiter="\t",
                fieldnames=BISSE_COLS,
            )
        else:
            reader = csv.DictReader(
                f,
                delimiter="\t",
            )

        rows = list(reader)

    if any("index" not in row for row in rows):
        raise RuntimeError(
            f"Missing parameter index: {path}"
        )

    return rows


def write_design_chunk(path, rows, model):
    if not rows:
        raise RuntimeError("Empty parameter chunk.")

    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open("w", newline="") as f:
        if model == "BiSSE":
            writer = csv.writer(
                f,
                delimiter="\t",
                lineterminator="\n",
            )

            for row in rows:
                writer.writerow(
                    [row[column] for column in BISSE_COLS]
                )

        else:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0]),
                delimiter="\t",
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(rows)


def load_tasks():
    if not TASKS.is_file():
        raise RuntimeError(
            "Missing task manifest. Run --prepare first."
        )

    return json_read(TASKS)


# ============================================================
# 6. OUTPUT DIRECTORIES
# ============================================================

def outdir(task, conf):
    if task["split"] == "trainval":
        label = f"full_{conf['n_full']}"
    else:
        # Only for compatibility with old manifests.
        label = f"test_{conf['n_test']}"

    return (
        DATA
        / task["model"]
        / label
        / "chunks"
        / f"{task['chunk']:04d}"
    )


def subdir(task, conf):
    return (
        DATA
        / task["model"]
        / f"subset_{conf['n_subset']}"
        / "chunks"
        / f"{task['chunk']:04d}"
    )


def complete(task, conf):
    """
    Count a batch only when its validated output
    and all required files exist.
    """

    folder = outdir(task, conf)
    marker = folder / "DONE.json"

    if not marker.is_file():
        return False

    try:
        state = json_read(marker)

        if (
            state["task_id"] != task["id"]
            or state["count"] != task["count"]
        ):
            return False

        required = (
            "raw/trees.nwk.gz",
            "raw/trees.tsv.gz",
            "raw/parameters_prior.tsv.gz",
            "raw/parameters_final.tsv.gz",
            "raw/stats.tsv.gz",
            "encoded/encoded.tsv.gz",
        )

        if not all(
            (folder / name).is_file()
            and (folder / name).stat().st_size > 0
            for name in required
        ):
            return False

        if task["split"] == "trainval":
            subset = subdir(task, conf)
            sub_marker = subset / "DONE.json"

            if not sub_marker.is_file():
                return False

            if (
                json_read(sub_marker).get("count")
                != state["subset_count"]
            ):
                return False

            if not all(
                (subset / name).is_file()
                and (subset / name).stat().st_size > 0
                for name in required
            ):
                return False

        return True

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ):
        return False


# ============================================================
# 7. PREPARE FULL DATASET
# ============================================================

def prepare(args):
    check_sources()
    check_author_archives()

    if (
        args.n_full <= 0
        or args.n_subset <= 0
        or args.chunk_size <= 0
        or args.n_subset > args.n_full
    ):
        raise RuntimeError(
            "Require 0 < subset <= full and chunk size > 0."
        )

    if not 0 < args.train_fraction < 1:
        raise RuntimeError(
            "Training fraction must be between 0 and 1."
        )

    # Preserve existing run and task IDs.
    if CONFIG.is_file():
        conf = json_read(CONFIG)

        settings = (
            "n_full",
            "n_subset",
            "n_test",
            "train_fraction",
            "chunk_size",
            "seed",
            "max_time",
        )

        for key in settings:
            if conf[key] != getattr(args, key):
                raise RuntimeError(
                    f"Existing run uses different {key}. "
                    "Keep the existing settings, or move "
                    "Data/_step3 aside to start a new run."
                )

        check_sources(conf)

        log("[REUSE] Existing task manifest unchanged.")

        return conf

    LOGS.mkdir(parents=True, exist_ok=True)

    adapted = patch_bisse_seed_copy()

    conf = {
        "n_full": args.n_full,
        "n_subset": args.n_subset,
        "n_test": args.n_test,
        "train_fraction": args.train_fraction,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "max_time": args.max_time,
        "test_source": "author",
        "source_sha256": {
            name: hash_file(path)
            for name, path in source_paths().items()
        },
        "bisse_patch_sha256": hash_file(adapted),
        "created_at": dt.datetime.now().isoformat(),
    }

    n_train = round(
        args.n_full * args.train_fraction
    )

    n_subset_train = round(
        args.n_subset * args.train_fraction
    )

    if (
        n_subset_train > n_train
        or args.n_subset - n_subset_train
        > args.n_full - n_train
    ):
        raise RuntimeError(
            "Subset cannot fit within full train/validation split."
        )

    tasks = []

    for model_number, model in enumerate(MODELS):
        folder = DATA / model
        folder.mkdir(parents=True, exist_ok=True)

        rng = random.Random(
            args.seed + 76543 + model_number
        )

        shuffled = list(range(args.n_full))
        rng.shuffle(shuffled)

        train_ids = set(shuffled[:n_train])
        validation_ids = set(shuffled[n_train:])

        subset_train = set(
            rng.sample(
                sorted(train_ids),
                n_subset_train,
            )
        )

        subset_validation = set(
            rng.sample(
                sorted(validation_ids),
                args.n_subset - n_subset_train,
            )
        )

        selected = sorted(
            subset_train | subset_validation
        )

        # Save IDs selected for nested 10K subset.
        subset_file = (
            CONTROL / "subset_ids" / f"{model}.txt"
        )

        subset_file.parent.mkdir(
            parents=True, exist_ok=True
        )

        subset_file.write_text(
            "\n".join(map(str, selected)) + "\n"
        )

        # Save reproducible full split.
        full_split = (
            folder / "full_trainval_assignments.tsv"
        )

        with full_split.open("w") as f:
            f.write("index\tpartition\n")

            for i in range(args.n_full):
                partition = (
                    "train"
                    if i in train_ids
                    else "validation"
                )

                f.write(f"{i}\t{partition}\n")

        # Save subset split.
        sub_split = (
            folder / "subset_trainval_assignments.tsv"
        )

        with sub_split.open("w") as f:
            f.write("index\tpartition\n")

            for i in selected:
                partition = (
                    "train"
                    if i in subset_train
                    else "validation"
                )

                f.write(f"{i}\t{partition}\n")

        # Original PhyloCNN Latin Hypercube generator.
        parameter_file = (
            folder / "parameter_design_trainval.tsv"
        )

        seed = (
            args.seed
            + (model_number + 1) * 10000
        )

        shell(
            seeded_cmd(
                source_paths()["params"],
                seed,
                param_options(
                    model,
                    args.n_full,
                    parameter_file,
                ),
            )
        )

        rows = design_rows(
            parameter_file, model
        )

        if len(rows) != args.n_full:
            raise RuntimeError(
                f"{model}: incorrect parameter count."
            )

        indices = [
            idx(row["index"]) for row in rows
        ]

        if indices != list(range(args.n_full)):
            raise RuntimeError(
                f"{model}: incorrect parameter indices."
            )

        # Create chunks for SLURM.
        for start in range(
            0, args.n_full, args.chunk_size
        ):
            number = start // args.chunk_size

            chunk_file = (
                CONTROL
                / "params"
                / model
                / "trainval"
                / f"{number:04d}.tsv"
            )

            batch = rows[
                start:start + args.chunk_size
            ]

            write_design_chunk(
                chunk_file,
                batch,
                model,
            )

            tasks.append({
                "id": len(tasks),
                "model": model,
                "split": "trainval",
                "chunk": number,
                "param_file": str(
                    chunk_file.relative_to(ROOT)
                ),
                "start": start,
                "count": len(batch),
            })

    json_save(TASKS, tasks)
    json_save(CONFIG, conf)

    log(
        "[PREPARED]",
        len(tasks),
        "training/validation tasks."
    )

    log(
        "[FULL]",
        n_train,
        "training,",
        args.n_full - n_train,
        "validation per model."
    )

    log(
        "[SUBSET]",
        n_subset_train,
        "training,",
        args.n_subset - n_subset_train,
        "validation per model."
    )

    log(
        "[TEST]",
        "Original PhyloCNN test archives."
    )

    return conf


# ============================================================
# 8. SIMULATION
# ============================================================

def sim_seed(task, conf):
    # Preserve original seed calculations for running jobs.
    return (
        conf["seed"]
        + (MODELS.index(task["model"]) + 1) * 1000000
        + (SPLITS.index(task["split"]) + 1) * 100000
        + task["chunk"] * 100
    )


def simulate(task, parameter_file, scratch, seed, conf):
    model = task["model"]

    scratch.mkdir(parents=True, exist_ok=True)

    if model == "BiSSE":
        tree_file = scratch / "trees.nwk"
        stat_file = scratch / "stats.tsv"
        final_file = scratch / "final.tsv"

        shell(
            [
                rscript(),
                CONTROL / "code/BiSSE_simulator_per_row_seed.R",
                parameter_file,
                1,
                seed,
                10,
                100,
                tree_file,
                stat_file,
                final_file,
            ],
            cwd=scratch,
        )

        with tree_file.open() as f:
            trees = [
                line.strip()
                for line in f
                if line.strip()
            ]

        with stat_file.open(newline="") as f:
            stats = [
                dict(zip(BISSE_STATS, row))
                for row in csv.reader(
                    f, delimiter="\t"
                )
                if row
            ]

        with final_file.open(newline="") as f:
            final = [
                dict(zip(BISSE_COLS, row))
                for row in csv.reader(
                    f, delimiter="\t"
                )
                if row
            ]

        return trees, stats, final

    # Original PhyloDeep epidemic simulators.
    tree_file = scratch / "trees.tsv"

    with tree_file.open("w") as output:
        shell(
            seeded_cmd(
                source_paths()[model],
                seed,
                [
                    parameter_file,
                    conf["max_time"],
                ],
            ),
            cwd=scratch,
            stdout=output,
        )

    with tree_file.open(newline="") as f:
        records = list(
            csv.DictReader(
                f, delimiter="\t"
            )
        )

    design = design_rows(
        parameter_file, model
    )

    # Verify that the simulator retained parameter IDs.
    if len(records) != len(design):
        raise RuntimeError(
            "Simulator returned incorrect tree count."
        )

    for i, record in enumerate(records):
        if (
            idx(record["index"])
            != idx(design[i]["index"])
        ):
            raise RuntimeError(
                "Tree-to-parameter index mismatch."
            )

    with (
        scratch / "subpopulations.txt"
    ).open(newline="") as f:
        stats = list(
            csv.DictReader(
                f, delimiter="\t"
            )
        )

    return [
        record["tree"] for record in records
    ], stats, None


# ============================================================
# 9. TREE VALIDATION
# ============================================================

def valid_tree(newick, model, target_size=None):
    from ete3 import Tree

    if (
        not newick
        or newick.strip().lower()
        in ("na", "nan", "none")
    ):
        raise ValueError("Missing Newick tree.")

    tree = Tree(newick, format=1)

    leaves = tree.get_leaf_names()
    count = len(leaves)

    if not 200 <= count <= 500:
        raise ValueError(
            f"Expected 200–500 leaves; got {count}."
        )

    if (
        target_size is not None
        and model != "BiSSE"
        and abs(count - int(target_size)) > 1
    ):
        raise ValueError(
            "Tree size differs from parameter design."
        )

    if len(set(leaves)) != count:
        raise ValueError(
            "Duplicate leaf identifiers."
        )

    for node in tree.traverse():
        length = float(node.dist)

        if (
            not math.isfinite(length)
            or length < -1e-8
        ):
            raise ValueError(
                "Invalid branch length."
            )

    if model == "BiSSE":
        if (
            "t_s=" not in newick
            and "t_s-" not in newick
        ):
            raise ValueError(
                "Missing BiSSE tip-state annotations."
            )

    return count


# ============================================================
# 10. RECOVER DIFFICULT SIMULATION ROWS
# ============================================================

def rescue(task, row, conf, scratch, attempt):
    destination = (
        scratch
        / f"rescue_{idx(row['index'])}_{attempt}"
    )

    destination.mkdir(
        parents=True, exist_ok=True
    )

    params = destination / "one.tsv"

    write_design_chunk(
        params,
        [row],
        task["model"],
    )

    seed = (
        sim_seed(task, conf)
        + attempt * 10000000
    )

    trees, stats, finals = simulate(
        task,
        params,
        destination,
        seed,
        conf,
    )

    if (
        len(trees) != 1
        or len(stats) != 1
        or (
            finals is not None
            and len(finals) != 1
        )
    ):
        raise RuntimeError(
            "Single-row simulator returned wrong count."
        )

    if idx(stats[0]["index"]) != idx(row["index"]):
        raise RuntimeError(
            "Statistics index mismatch."
        )

    if (
        finals
        and idx(finals[0]["index"])
        != idx(row["index"])
    ):
        raise RuntimeError(
            "Final parameters index mismatch."
        )

    valid_tree(
        trees[0],
        task["model"],
        idx(row["tree_size"]),
    )

    return (
        trees[0],
        stats[0],
        finals[0] if finals else row,
    )


# ============================================================
# 11. ORIGINAL PHYLOCNN ENCODERS
# ============================================================

def encoder(model):
    name = (
        "encode_bisse"
        if model == "BiSSE"
        else "encode_epi"
    )

    source = source_paths()[name]

    spec = importlib.util.spec_from_file_location(
        "original_phylo_encoder_" + model,
        source,
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    if model == "BiSSE":
        # Support both representations of NHX tip states.
        def read_tip_state(tree):
            for leaf in tree.iter_leaves():
                name = str(leaf.name)

                if (
                    "&&NHX-t_s=1" in name
                    or "&&NHX:t_s=1" in name
                ):
                    state = 1

                elif (
                    "&&NHX-t_s=2" in name
                    or "&&NHX:t_s=2" in name
                ):
                    state = 2

                else:
                    value = str(
                        getattr(leaf, "t_s", "")
                    )

                    state = (
                        int(value)
                        if value in ("1", "2")
                        else 0
                    )

                leaf.add_feature("t_s", state)

        module.assign_t_s_attributes = read_tip_state

    return module


# ============================================================
# 12. COMPRESSED OUTPUT WRITERS
# ============================================================

def gzout(path):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    return gzip.open(
        path,
        "wt",
        encoding="utf-8",
        newline="",
        compresslevel=1,
    )


def writers(root, stack):
    paths = {
        "nwk": "raw/trees.nwk.gz",
        "trees": "raw/trees.tsv.gz",
        "prior": "raw/parameters_prior.tsv.gz",
        "final": "raw/parameters_final.tsv.gz",
        "stats": "raw/stats.tsv.gz",
        "encoded": "encoded/encoded.tsv.gz",
    }

    return {
        name: stack.enter_context(
            gzout(root / path)
        )
        for name, path in paths.items()
    }


# ============================================================
# 13. ENCODE AND VALIDATE SIMULATED TREES
# ============================================================

def save_batch(
    task,
    conf,
    rows,
    trees,
    stats,
    finals,
    stage,
    subsets,
    progress,
):
    import numpy as np

    model = task["model"]
    enc = encoder(model)

    selected = 0

    with ExitStack() as stack:
        full = writers(
            stage / "full", stack
        )

        subset = (
            writers(stage / "subset", stack)
            if subsets is not None
            else None
        )

        columns = list(rows[0])
        stat_cols = list(stats[0])

        destinations = (
            (full, subset)
            if subset is not None
            else (full,)
        )

        for dst in destinations:
            for label, cols in (
                ("prior", columns),
                ("final", columns),
                ("stats", stat_cols),
            ):
                dst[label].write(
                    "\t".join(cols) + "\n"
                )

            dst["trees"].write(
                "index\ttree\n"
            )

        for j, (row, tree, st, fin) in enumerate(
            zip(rows, trees, stats, finals)
        ):
            ident = idx(row["index"])

            if (
                idx(st["index"]) != ident
                or idx(fin["index"]) != ident
            ):
                raise RuntimeError(
                    f"Tree/label mismatch: row {j}"
                )

            tips = valid_tree(
                tree,
                model,
                idx(row["tree_size"]),
            )

            if model == "BiSSE":
                state_total = (
                    idx(st["tip_state_1"])
                    + idx(st["tip_state_2"])
                )

                if (
                    idx(st["total_tips"]) != tips
                    or state_total != tips
                ):
                    raise RuntimeError(
                        "BiSSE tip states do not match tree."
                    )

            elif "sampled_leaves" in st:
                if idx(st["sampled_leaves"]) != tips:
                    raise RuntimeError(
                        "Epidemic sampled tips mismatch."
                    )

            # The actual original PhyloCNN encoder.
            df = enc.process_tree(tree)

            expected = (1000, DIMS[model])

            if df.shape != expected:
                raise RuntimeError(
                    f"Encoder produced {df.shape}, "
                    f"expected {expected}."
                )

            if not np.isfinite(
                df.to_numpy(dtype=float)
            ).all():
                raise RuntimeError(
                    f"Nonfinite encoding: {ident}"
                )

            if model == "BiSSE":
                maximum = float(
                    np.max(df.iloc[:999, 4])
                )

                if maximum <= 0:
                    raise RuntimeError(
                        "BiSSE tip states lost in encoding."
                    )

            else:
                df = df.astype("float32")

            chosen = (
                subset is not None
                and ident in subsets
            )

            current = (
                (full, subset)
                if chosen
                else (full,)
            )

            for dst in current:
                dst["nwk"].write(
                    tree.rstrip() + "\n"
                )

                csv.writer(
                    dst["trees"],
                    delimiter="\t",
                    lineterminator="\n",
                ).writerow([ident, tree])

                for label, record in (
                    ("prior", row),
                    ("final", fin),
                    ("stats", st),
                ):
                    fields = (
                        stat_cols
                        if label == "stats"
                        else columns
                    )

                    dst[label].write(
                        "\t".join(
                            str(record[k])
                            for k in fields
                        ) + "\n"
                    )

                header = (
                    j == 0
                    if dst is full
                    else selected == 0
                )

                df.to_csv(
                    dst["encoded"],
                    sep="\t",
                    index=True,
                    index_label=(
                        "Index" if header else None
                    ),
                    header=header,
                )

            if chosen:
                selected += 1

            if (
                (j + 1) % 25 == 0
                or j + 1 == len(rows)
            ):
                log(
                    "[ENCODE]",
                    model,
                    task["split"],
                    "chunk",
                    task["chunk"],
                    f"{j + 1}/{len(rows)}",
                )

                if progress:
                    json_save(progress, {
                        "encoded": j + 1,
                        "expected": len(rows),
                        "updated": (
                            dt.datetime.now().isoformat()
                        ),
                    })

    return selected


# ============================================================
# 14. SLURM SIMULATION WORKER
# ============================================================

def worker(task, conf, pilot=False):
    model = task["model"]

    # Old queued generated-test tasks are unnecessary.
    if task["split"] == "test" and not pilot:
        log(
            "[SKIP] Generated test task",
            task["id"],
            "- using original author test files."
        )
        return

    if not pilot and complete(task, conf):
        log(
            "[SKIP] Already verified:",
            model,
            task["chunk"],
        )
        return

    rows = design_rows(
        ROOT / task["param_file"],
        model,
    )

    if len(rows) != task["count"]:
        raise RuntimeError(
            "Parameter chunk row count mismatch."
        )

    base = (
        CONTROL / "pilot" / model
        if pilot
        else SCRATCH / f"task_{task['id']:04d}"
    )

    if base.exists():
        shutil.rmtree(base)

    base.mkdir(
        parents=True, exist_ok=True
    )

    progress = (
        None
        if pilot
        else outdir(task, conf) / "PROGRESS.json"
    )

    if progress:
        json_save(progress, {
            "encoded": 0,
            "expected": len(rows),
        })

    try:
        excluded = []

        # First attempt a bulk simulation.
        try:
            trees, stats, finals = simulate(
                task,
                ROOT / task["param_file"],
                base / "simulation",
                sim_seed(task, conf),
                conf,
            )

            if (
                len(trees) != len(rows)
                or len(stats) != len(rows)
                or (
                    finals is not None
                    and len(finals) != len(rows)
                )
            ):
                raise RuntimeError(
                    "Bulk simulation row counts mismatch."
                )

            if finals is None:
                finals = list(rows)

            rescued = 0

        except (
            RuntimeError,
            subprocess.CalledProcessError,
            ValueError,
            OSError,
        ) as error:
            log(
                "[BULK FAILED]",
                str(error)[:260]
            )

            log(
                "[RECOVERY]",
                "Retrying original simulator per row."
            )

            trees, stats, finals = [], [], []

            for row in rows:
                last_error = ""

                for retry in range(1, RESCUE_ATTEMPTS + 1):
                    try:
                        tree, st, fin = rescue(
                            task,
                            row,
                            conf,
                            base,
                            retry,
                        )

                        trees.append(tree)
                        stats.append(st)
                        finals.append(fin)

                        break

                    except Exception as exc:
                        last_error = str(exc)[:300]
                        log(
                            "[RETRY]",
                            row["index"],
                            retry,
                            str(exc)[:160],
                        )

                else:
                    # Keep the lists aligned; this row is excluded below.
                    trees.append(None)
                    stats.append(None)
                    finals.append(None)
                    excluded.append({
                        "index": idx(row["index"]),
                        "reason": "unable to simulate: " + last_error,
                    })
                    log("[EXCLUDED]", model, row["index"], last_error[:150])

            rescued = len(rows) - len(excluded)

        # Independently validate each tree and its labels.
        excluded_ids = {item["index"] for item in excluded}

        for i, row in enumerate(rows):
            if idx(row["index"]) in excluded_ids:
                continue

            try:
                valid_tree(
                    trees[i],
                    model,
                    idx(row["tree_size"]),
                )

                if (
                    idx(stats[i]["index"])
                    != idx(row["index"])
                    or idx(finals[i]["index"])
                    != idx(row["index"])
                ):
                    raise ValueError(
                        "Parameter index mismatch."
                    )

            except Exception as error:
                log(
                    "[RESCUE]",
                    model,
                    row["index"],
                    str(error)[:150],
                )

                last_error = str(error)[:300]

                for retry in range(1, RESCUE_ATTEMPTS + 1):
                    try:
                        (
                            trees[i],
                            stats[i],
                            finals[i],
                        ) = rescue(
                            task,
                            row,
                            conf,
                            base,
                            retry,
                        )

                        rescued += 1
                        break

                    except Exception as exc:
                        last_error = str(exc)[:300]
                        log(
                            "[RETRY]",
                            row["index"],
                            retry,
                            str(exc)[:150],
                        )

                else:
                    excluded.append({
                        "index": idx(row["index"]),
                        "reason": "failed validation after "
                                  f"{RESCUE_ATTEMPTS} seeded attempts: "
                                  + last_error,
                    })
                    excluded_ids.add(idx(row["index"]))
                    log("[EXCLUDED]", model, row["index"], last_error[:150])

        # A few unsimulatable parameter sets are dropped (and recorded);
        # many would mean something is systematically wrong.
        limit = int(MAX_EXCLUDED_FRACTION * len(rows))

        if len(excluded) > limit:
            raise RuntimeError(
                f"{len(excluded)} of {len(rows)} rows could not be "
                f"simulated (limit {limit}); first: "
                + "; ".join(
                    f"{e['index']}: {e['reason'][:80]}"
                    for e in excluded[:3]
                )
            )

        if excluded:
            keep = [
                i for i, row in enumerate(rows)
                if idx(row["index"]) not in excluded_ids
            ]
            rows = [rows[i] for i in keep]
            trees = [trees[i] for i in keep]
            stats = [stats[i] for i in keep]
            finals = [finals[i] for i in keep]

        # Nested subset only from train/validation.
        subsets = None

        if not pilot:
            path = (
                CONTROL
                / "subset_ids"
                / f"{model}.txt"
            )

            subsets = {
                int(x)
                for x in path.read_text().split()
            }

        selected = save_batch(
            task,
            conf,
            rows,
            trees,
            stats,
            finals,
            base / "staging",
            subsets,
            progress,
        )

        staging = base / "staging/full"

        metadata = {
            "task_id": task["id"],
            "model": model,
            "split": task["split"],
            "chunk": task["chunk"],
            # count = parameter rows attempted (what complete() checks);
            # kept = trees written; excluded = unsimulatable rows + reason.
            "count": task["count"],
            "kept": len(rows),
            "excluded": excluded,
            "subset_count": selected,
            "rescued_trees": rescued,
            "seed": sim_seed(task, conf),
            "validated_at": (
                dt.datetime.now().isoformat()
            ),
            "original_simulator": str(
                source_paths()[model].relative_to(ROOT)
            ),
            "original_encoder": str(
                source_paths()[
                    "encode_bisse"
                    if model == "BiSSE"
                    else "encode_epi"
                ].relative_to(ROOT)
            ),
            "encoded_shape_per_tree": [
                1000,
                DIMS[model],
            ],
        }

        json_save(
            staging / "DONE.json",
            metadata,
        )

        if not pilot:
            json_save(
                base / "staging/subset/DONE.json",
                {
                    **metadata,
                    "count": selected,
                },
            )

        destination = (
            CONTROL / "pilot_verified" / model
            if pilot
            else outdir(task, conf)
        )

        if destination.exists():
            shutil.rmtree(destination)

        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        shutil.move(
            str(staging),
            str(destination),
        )

        if not pilot:
            subset_destination = subdir(
                task, conf
            )

            if subset_destination.exists():
                shutil.rmtree(
                    subset_destination
                )

            subset_destination.parent.mkdir(
                parents=True, exist_ok=True
            )

            shutil.move(
                str(base / "staging/subset"),
                str(subset_destination),
            )

        log(
            "[COMPLETE]",
            model,
            task["split"],
            "chunk",
            task["chunk"],
            "trees",
            len(rows),
            "subset",
            selected,
            "rescued",
            rescued,
        )

    except Exception as error:
        json_save(
            base / "FAILED.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )

        raise

    finally:
        if (
            base.exists()
            and not (base / "FAILED.json").exists()
        ):
            shutil.rmtree(base)


# ============================================================
# 15. ONE-TREE PILOT
# ============================================================

def pilot(model):
    conf = json_read(CONFIG)

    check_sources(conf)

    source = next(
        task
        for task in load_tasks()
        if (
            task["model"] == model
            and task["split"] == "trainval"
        )
    )

    rows = design_rows(
        ROOT / source["param_file"],
        model,
    )[:1]

    destination = (
        CONTROL / "pilot" / f"{model}_one.tsv"
    )

    write_design_chunk(
        destination, rows, model
    )

    single = {
        **source,
        "param_file": str(
            destination.relative_to(ROOT)
        ),
        "id": -1,
        "count": 1,
    }

    worker(
        single,
        conf,
        pilot=True,
    )


# ============================================================
# 16. ORIGINAL AUTHOR TEST DATA
# ============================================================

def author_paths(model):
    folder = (
        ROOT / "phyloCNN/Test_Sets" / model
    )

    labels = folder / f"{model}.csv.gz"

    trees = [
        folder / name
        for name in TEST_ARCHIVES[model][1:]
    ]

    return labels, trees


def author_newicks(paths):
    """
    Read semicolon-terminated Newick trees in archive order.

    Preserves the original author files unchanged.
    """

    for path in paths:
        buffer = ""

        with gzip.open(
            path, "rt", encoding="utf-8"
        ) as f:
            for line in f:
                buffer += line.strip()

                while ";" in buffer:
                    tree, buffer = buffer.split(
                        ";", 1
                    )

                    if tree.strip():
                        yield tree.strip() + ";"

        if buffer.strip():
            raise RuntimeError(
                f"Unterminated Newick: {path}"
            )


def author_labels(path):
    with gzip.open(
        path,
        "rt",
        newline="",
        encoding="utf-8",
    ) as f:
        sample = f.read(4096)
        f.seek(0)

        delimiter = (
            "\t"
            if sample.count("\t") > sample.count(",")
            else ","
        )

        reader = csv.DictReader(
            f, delimiter=delimiter
        )

        rows = list(reader)

    if not rows or not reader.fieldnames:
        raise RuntimeError(
            f"Empty author parameter table: {path}"
        )

    return rows


def author_test_done(model):
    folder = DATA / model / "author_test"
    marker = folder / "DONE.json"

    if not marker.is_file():
        return None

    try:
        state = json_read(marker)

        required = (
            "raw/trees.nwk.gz",
            "raw/trees.tsv.gz",
            "raw/parameters_original.tsv.gz",
            "encoded/encoded.tsv.gz",
        )

        for name in required:
            path = folder / name

            if (
                not path.is_file()
                or path.stat().st_size == 0
            ):
                return None

        return state

    except (OSError, ValueError):
        return None


# ============================================================
# 17. ENCODE AUTHOR-PROVIDED TEST DATA
# ============================================================

def author_worker(model):
    import numpy as np

    check_author_archives()

    if author_test_done(model):
        log(
            "[SKIP]",
            model,
            "author test already encoded."
        )
        return

    label_file, tree_files = author_paths(model)

    labels = author_labels(label_file)

    enc = encoder(model)

    staging = (
        SCRATCH / f"author_test_{model}"
    )

    if staging.exists():
        shutil.rmtree(staging)

    staging.mkdir(
        parents=True, exist_ok=True
    )

    try:
        with ExitStack() as stack:
            tree_out = stack.enter_context(
                gzout(staging / "raw/trees.nwk.gz")
            )

            indexed = stack.enter_context(
                gzout(staging / "raw/trees.tsv.gz")
            )

            params = stack.enter_context(
                gzout(
                    staging
                    / "raw/parameters_original.tsv.gz"
                )
            )

            encoded = stack.enter_context(
                gzout(
                    staging / "encoded/encoded.tsv.gz"
                )
            )

            columns = [
                "tree_row",
                *list(labels[0]),
            ]

            params.write(
                "\t".join(columns) + "\n"
            )

            indexed.write(
                "index\ttree\n"
            )

            count = 0

            for i, tree in enumerate(
                author_newicks(tree_files)
            ):
                if i >= len(labels):
                    raise RuntimeError(
                        f"{model}: more trees than labels."
                    )

                valid_tree(tree, model)

                df = enc.process_tree(tree)

                if df.shape != (
                    1000, DIMS[model]
                ):
                    raise RuntimeError(
                        f"{model}: incorrect test encoding shape."
                    )

                if not np.isfinite(
                    df.to_numpy(dtype=float)
                ).all():
                    raise RuntimeError(
                        f"{model}: nonfinite test encoding."
                    )

                if model == "BiSSE":
                    maximum = float(
                        np.max(df.iloc[:999, 4])
                    )

                    if maximum <= 0:
                        raise RuntimeError(
                            "BiSSE tip states lost."
                        )

                else:
                    df = df.astype("float32")

                tree_out.write(
                    tree + "\n"
                )

                csv.writer(
                    indexed,
                    delimiter="\t",
                    lineterminator="\n",
                ).writerow([i, tree])

                csv.writer(
                    params,
                    delimiter="\t",
                    lineterminator="\n",
                ).writerow(
                    [i] + [
                        labels[i][column]
                        for column in labels[i]
                    ]
                )

                df.to_csv(
                    encoded,
                    sep="\t",
                    index=True,
                    index_label=(
                        "Index" if i == 0 else None
                    ),
                    header=(i == 0),
                )

                count += 1

                if count % 25 == 0:
                    log(
                        "[AUTHOR TEST]",
                        model,
                        f"{count}/{len(labels)}",
                        "encoded."
                    )

            if count != len(labels):
                raise RuntimeError(
                    f"{model}: {count} trees but "
                    f"{len(labels)} parameter rows."
                )

        metadata = {
            "model": model,
            "trees": count,
            "label_rows": len(labels),
            "encoded_shape_per_tree": [
                1000,
                DIMS[model],
            ],
            "source": "phyloCNN/Test_Sets",
            "tree_files": [
                str(path.relative_to(ROOT))
                for path in tree_files
            ],
            "labels_file": str(
                label_file.relative_to(ROOT)
            ),
            "alignment": (
                "positional archive ordering; "
                "Newick records do not contain "
                "tree-specific parameter IDs"
            ),
            "validated_at": (
                dt.datetime.now().isoformat()
            ),
        }

        json_save(
            staging / "DONE.json",
            metadata,
        )

        destination = (
            DATA / model / "author_test"
        )

        if destination.exists():
            raise RuntimeError(
                "Incomplete author-test output exists. "
                f"Inspect before replacing: {destination}"
            )

        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        shutil.move(
            str(staging),
            str(destination),
        )

        log(
            "[AUTHOR TEST READY]",
            model,
            count,
            "trees and matching parameter rows."
        )

    except Exception as error:
        json_save(
            staging / "FAILED.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )

        raise


# ============================================================
# 18. AUDIT ORIGINAL TEST ARCHIVES
# ============================================================

def audit_author():
    print()
    print("=" * 75)
    print("ORIGINAL PHYLOCNN INDEPENDENT TEST ARCHIVES")
    print("=" * 75)

    for model, names in TEST_ARCHIVES.items():
        folder = (
            ROOT / "phyloCNN/Test_Sets" / model
        )

        missing = [
            name
            for name in names
            if not (folder / name).is_file()
        ]

        if missing:
            print(
                model,
                "MISSING:",
                missing,
            )
            continue

        labels_file, tree_files = author_paths(model)

        label_count = len(
            author_labels(labels_file)
        )

        tree_count = sum(
            1
            for _ in author_newicks(tree_files)
        )

        result = (
            "COUNT MATCH"
            if tree_count == label_count
            else "COUNT MISMATCH"
        )

        print(
            f"{model:<6} "
            f"trees={tree_count:>8,} "
            f"labels={label_count:>8,} "
            f"{result}"
        )

    print()
    print(
        "Archive row counts are checked; "
        "tree-specific parameter correspondence "
        "cannot be proven from counts alone."
    )


# ============================================================
# 19. FIXED STATUS REPORTER
# ============================================================

def status():
    print()
    print("=" * 88)
    print(
        "STEP 3: VERIFIED TRAINING/VALIDATION "
        "AND AUTHOR TEST STATUS"
    )
    print("=" * 88)

    if not CONFIG.is_file() or not TASKS.is_file():
        print(
            "No prepared run found. "
            "Run --prepare, or submit Step3-GenerateData.sh."
        )
        return

    conf = json_read(CONFIG)
    tasks = load_tasks()

    print(
        f"{'MODEL':<7} "
        f"{'DATASET':<15} "
        f"{'VERIFIED/TARGET':>20} "
        f"{'ENCODING':>12} "
        f"{'CHUNKS':>12} "
        f"{'FAILED':>8}"
    )

    print("-" * 88)

    total = 0
    total_goal = 0

    for model in MODELS:
        # Only count training/validation simulation tasks.
        items = [
            task
            for task in tasks
            if (
                task["model"] == model
                and task["split"] == "trainval"
            )
        ]

        verified = [
            task
            for task in items
            if complete(task, conf)
        ]

        # Trees actually kept (excluded unsimulatable rows are not counted).
        finished = 0
        dropped = 0

        for task in verified:
            state = json_read(outdir(task, conf) / "DONE.json")
            finished += state.get("kept", state["count"])
            dropped += len(state.get("excluded", []))

        goal = sum(
            task["count"]
            for task in items
        )

        partial = 0

        for task in items:
            if task in verified:
                continue

            path = (
                outdir(task, conf)
                / "PROGRESS.json"
            )

            if path.is_file():
                try:
                    partial += int(
                        json_read(path).get(
                            "encoded", 0
                        )
                    )
                except (
                    OSError,
                    ValueError,
                    TypeError,
                ):
                    pass

        failed = sum(
            (
                SCRATCH
                / f"task_{task['id']:04d}"
                / "FAILED.json"
            ).is_file()
            for task in items
        )

        # FIX: Valid Python f-string number formatting.
        print(
            f"{model:<7} "
            f"{'full':<15} "
            f"{finished:>8,}/{goal:<8,} "
            f"{partial:>12,} "
            f"{len(verified):>4}/{len(items):<7} "
            f"{failed:>8}"
            + (f"   ({dropped} unsimulatable rows excluded)" if dropped else "")
        )

        subset_count = sum(
            json_read(
                subdir(task, conf) / "DONE.json"
            )["count"]
            for task in verified
        )

        print(
            f"{model:<7} "
            f"{'nested_subset':<15} "
            f"{subset_count:>8,}/"
            f"{conf['n_subset']:<8,}"
        )

        author = author_test_done(model)
        author_count = (
            author["trees"] if author else 0
        )

        names = TEST_ARCHIVES[model]

        present = sum(
            (
                ROOT
                / "phyloCNN/Test_Sets"
                / model
                / name
            ).is_file()
            for name in names
        )

        print(
            f"{model:<7} "
            f"{'author_test':<15} "
            f"{author_count:>8,} encoded; "
            f"original archives {present}/{len(names)}"
        )

        total += finished
        total_goal += goal

    print("-" * 88)

    print(
        f"TOTAL FULL TRAIN/VAL: "
        f"{total:,}/{total_goal:,} verified"
    )

    print(
        "ENCODING is the last reported progress "
        "of unfinished batches; it may be stale."
    )

    print()

    for model in MODELS:
        marker = (
            CONTROL
            / "pilot_verified"
            / model
            / "DONE.json"
        )

        print(
            "PILOT",
            model,
            "PASS" if marker.is_file() else "PENDING",
        )

    pending = pending_items()

    print(
        f"\nWork items: {len(work_items(tasks))} "
        f"(array {array_range(tasks)}); "
        f"unfinished: {len(pending)}"
    )

    if pending:
        print(
            "Resume only unfinished items:\n"
            "  sbatch --array="
            f"{compact(pending)}%100 Step3-GenerateData.sh"
        )

    print(
        "SLURM queue: squeue -u $USER"
    )

    print(
        "\nLegacy generated-test tasks are ignored. "
        "Original PhyloCNN test archives are used."
    )


# ============================================================
# 20. WORK ITEMS (ONE PER SLURM ARRAY ELEMENT)
# ============================================================

def compact(values):
    values = sorted(set(values))

    if not values:
        return ""

    spans = []
    start = previous = values[0]

    for value in values[1:]:
        if value == previous + 1:
            previous = value
        else:
            spans.append(
                str(start)
                if start == previous
                else f"{start}-{previous}"
            )

            start = previous = value

    spans.append(
        str(start)
        if start == previous
        else f"{start}-{previous}"
    )

    return ",".join(spans)


def work_items(tasks):
    """
    Array index -> work item.

    Training/validation chunks first (manifest order, task IDs
    unchanged), then one author-test encoding item per model.
    Legacy generated-test tasks are not work items.
    """

    trainval = [
        ("trainval", task)
        for task in tasks
        if task["split"] == "trainval"
    ]

    author = [
        ("author_test", model)
        for model in MODELS
    ]

    return trainval + author


def expected_item_count(args):
    """Number of work items, without preparing anything."""

    if TASKS.is_file():
        return len(work_items(load_tasks()))

    chunks = math.ceil(args.n_full / args.chunk_size)

    return len(MODELS) * chunks + len(MODELS)


def array_range(tasks=None, args=None):
    count = (
        len(work_items(tasks))
        if tasks is not None
        else expected_item_count(args)
    )

    return f"0-{count - 1}"


def item_done(kind, item, conf):
    if kind == "trainval":
        return complete(item, conf)

    return author_test_done(item) is not None


def pending_items():
    conf = json_read(CONFIG)

    return [
        number
        for number, (kind, item) in enumerate(
            work_items(load_tasks())
        )
        if not item_done(kind, item, conf)
    ]


# ============================================================
# 21. ONE-TIME PREPARATION SHARED BY ALL ARRAY TASKS
# ============================================================

def ensure_prepared(args):
    """
    Return the run configuration, preparing it exactly once.

    The first array task creates Data/_step3/prepare.lock
    (mkdir is atomic on cluster file systems) and prepares;
    all others wait until config.json and tasks.json exist.
    An existing run is always reused as it is.
    """

    deadline = time.time() + args.prepare_wait * 60

    while True:
        if CONFIG.is_file() and TASKS.is_file():
            return json_read(CONFIG)

        if PREPARE_FAILED.is_file():
            raise RuntimeError(
                "Preparation failed in another task: "
                f"{PREPARE_FAILED}. Fix the cause, delete "
                "that file, and submit again."
            )

        CONTROL.mkdir(parents=True, exist_ok=True)

        try:
            PREPARE_LOCK.mkdir()

        except FileExistsError:
            if time.time() > deadline:
                raise RuntimeError(
                    "Timed out waiting for preparation. If no "
                    "task is preparing, delete "
                    f"{PREPARE_LOCK} and submit again."
                )

            time.sleep(20 + random.random() * 10)
            continue

        try:
            json_save(
                PREPARE_LOCK / "owner.json",
                {
                    "host": socket.gethostname(),
                    "slurm_job": os.environ.get(
                        "SLURM_JOB_ID"
                    ),
                    "started": dt.datetime.now().isoformat(),
                },
            )

            if not (CONFIG.is_file() and TASKS.is_file()):
                log("[PREPARE] This task prepares the run.")

                try:
                    prepare(args)

                except Exception as error:
                    json_save(
                        PREPARE_FAILED,
                        {
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )

                    raise

            return json_read(CONFIG)

        finally:
            shutil.rmtree(PREPARE_LOCK, ignore_errors=True)


def manual_prepare(args):
    if PREPARE_FAILED.is_file():
        PREPARE_FAILED.unlink()

    conf = ensure_prepared(args)
    tasks = load_tasks()

    print()
    print(
        "Prepared:",
        len(work_items(tasks)),
        "work items. Submit with:"
    )
    print(
        f"  sbatch --array={array_range(tasks)}%100 "
        "Step3-GenerateData.sh"
    )

    return conf


# ============================================================
# 22. ARRAY TASK ENTRY POINT
# ============================================================

def run_task(args):
    number = args.task_id + args.task_offset

    if number < 0:
        raise RuntimeError(f"Invalid task ID {number}.")

    conf = ensure_prepared(args)
    items = work_items(load_tasks())

    if number >= len(items):
        log(
            "[NOTHING TO DO]",
            f"task {number}; this run has "
            f"{len(items)} work items "
            f"(array {array_range(load_tasks())})."
        )
        return

    kind, item = items[number]

    log(
        "[TASK]",
        number,
        "of",
        len(items) - 1,
        "|",
        kind,
        item["model"] if kind == "trainval" else item,
        f"chunk {item['chunk']}" if kind == "trainval" else "",
    )

    if kind == "trainval":
        check_sources(conf)
        worker(item, conf)

    else:
        author_worker(item)


def run_pilots(args):
    """One tree per model, run here (no SLURM). Use before sbatch."""

    ensure_prepared(args)

    failed = []

    for model in MODELS:
        log("[PILOT]", model)

        try:
            pilot(model)
            log("[PILOT PASS]", model)

        except Exception as error:
            log("[PILOT FAILED]", model, repr(error))
            failed.append(model)

    if failed:
        raise RuntimeError(
            "Pilot failed for: " + ", ".join(failed)
        )

    print()
    print("All pilots passed. Submit with:")
    print(
        f"  sbatch --array={array_range(load_tasks())}%100 "
        "Step3-GenerateData.sh"
    )


# ============================================================
# 23. COMMAND-LINE INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )

    actions = parser.add_mutually_exclusive_group(
        required=True
    )

    actions.add_argument(
        "--task-id",
        type=int,
        help="Run one work item (SLURM_ARRAY_TASK_ID).",
    )

    for name, text in (
        ("prepare", "Prepare the run (parameters, chunks, task list)."),
        ("pilot", "Simulate and encode one tree per model here."),
        ("status", "Show verified progress."),
        ("array-range", "Print the --array range for sbatch."),
        ("pending-array", "Print unfinished work items for sbatch --array."),
        ("audit-author-tests", "Check the authors' test archives."),
    ):
        actions.add_argument(
            "--" + name,
            action="store_true",
            help=text,
        )

    parser.add_argument(
        "--task-offset",
        type=int,
        default=int(
            os.environ.get("TASK_OFFSET", "0")
        ),
        help=(
            "Added to --task-id (env TASK_OFFSET). Needed only "
            "when a run has more than 1000 work items."
        ),
    )

    parser.add_argument(
        "--prepare-wait",
        type=int,
        default=180,
        help="Minutes an array task waits for preparation.",
    )

    # Configurable sizes and parameters (used only when preparing).
    parser.add_argument(
        "--n-full",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--n-subset",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help=(
            "Old manifest compatibility only. "
            "Actual test sizes come from author archives."
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261012,
    )

    parser.add_argument(
        "--max-time",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    if args.task_id is not None:
        run_task(args)

    elif args.prepare:
        manual_prepare(args)

    elif args.pilot:
        run_pilots(args)

    elif args.status:
        status()

    elif args.array_range:
        count = expected_item_count(args)

        if count - 1 > MAX_ARRAY_INDEX:
            print(
                f"[WARNING] {count} work items exceed the "
                "1000-index limit; submit in blocks with "
                "TASK_OFFSET.",
                file=sys.stderr,
            )

        print(array_range(args=args))

    elif args.pending_array:
        if not CONFIG.is_file():
            raise RuntimeError(
                "No prepared run. Run --prepare first."
            )

        pending = pending_items()

        if not pending:
            print(
                "[DONE] Every work item is verified.",
                file=sys.stderr,
            )

        # stdout carries only the range, for: sbatch --array=$(...)
        print(compact(pending))

    elif args.audit_author_tests:
        audit_author()


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log(
            "[ERROR]",
            repr(error),
        )

        traceback.print_exc()

        sys.exit(1)
