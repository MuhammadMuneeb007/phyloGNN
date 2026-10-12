 
"""Automatically prepare Step 3 trees for the Step 4 benchmark."""

import csv
import gzip
import itertools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Data"
REPO = ROOT / "phyloCNN"
DEST = REPO / "Training_Data"

DATASETS = ("BD", "BDEI", "BDSS", "BiSSE")


def log(message):
    print(message, flush=True)


def assemble(dataset):
    base = SOURCE / dataset

    chunks = sorted(
        marker.parent
        for marker in base.glob("full_*/chunks/*/DONE.json")
    )

    if not chunks:
        raise FileNotFoundError(
            f"No completed Step 3 training chunks: {base}"
        )

    destination = DEST / dataset
    destination.mkdir(parents=True, exist_ok=True)

    trees_file = destination / f"{dataset}_trees.nwk"
    params_file = destination / f"parameters_{dataset}.txt"

    temp_trees = trees_file.with_suffix(".nwk.tmp")
    temp_params = params_file.with_suffix(".txt.tmp")

    total = 0
    indices = set()
    header = None

    log(f"\n{'=' * 60}")
    log(f"DATASET: {dataset}")
    log(f"Completed chunks found: {len(chunks)}")

    try:
        with (
            temp_trees.open("w", encoding="utf-8") as tree_out,
            temp_params.open("w", encoding="utf-8", newline="") as param_out,
        ):
            writer = None

            for chunk in chunks:
                raw = chunk / "raw"

                tree_input = raw / "trees.nwk.gz"
                param_input = raw / "parameters_prior.tsv.gz"

                if not tree_input.is_file():
                    raise FileNotFoundError(tree_input)

                if not param_input.is_file():
                    raise FileNotFoundError(param_input)

                chunk_count = 0

                with (
                    gzip.open(tree_input, "rt", encoding="utf-8") as trees,
                    gzip.open(
                        param_input,
                        "rt",
                        encoding="utf-8",
                        newline="",
                    ) as params,
                ):
                    reader = csv.DictReader(
                        params,
                        delimiter="\t",
                    )

                    if not reader.fieldnames:
                        raise ValueError(
                            f"Missing parameter header: {param_input}"
                        )

                    if header is None:
                        header = list(reader.fieldnames)
                        writer = csv.DictWriter(
                            param_out,
                            fieldnames=header,
                            delimiter="\t",
                        )
                        writer.writeheader()

                    elif list(reader.fieldnames) != header:
                        raise ValueError(
                            f"Parameter columns differ in {chunk}"
                        )

                    for row, tree in itertools.zip_longest(
                        reader,
                        trees,
                    ):
                        if row is None or tree is None:
                            raise ValueError(
                                f"Tree/parameter mismatch in {chunk}"
                            )

                        tree = tree.strip()

                        if not tree.endswith(";"):
                            raise ValueError(
                                f"Invalid Newick tree in {chunk}"
                            )

                        if "index" not in row:
                            raise ValueError(
                                f"Missing index column in {param_input}"
                            )

                        index = int(float(row["index"]))

                        if index in indices:
                            raise ValueError(
                                f"Duplicate tree index: {index}"
                            )

                        indices.add(index)

                        writer.writerow(row)
                        tree_out.write(tree + "\n")

                        chunk_count += 1
                        total += 1

                marker = chunk / "DONE.json"

                try:
                    metadata = json.loads(marker.read_text())
                    expected = metadata.get("count")
                    if expected is not None and int(expected) != chunk_count:
                        raise ValueError(
                            f"{chunk}: expected {expected} trees, "
                            f"found {chunk_count}"
                        )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid DONE.json: {marker}"
                    ) from exc

                log(
                    f"  {chunk.name}: {chunk_count} trees "
                    f"(total={total})"
                )

        if total == 0:
            raise RuntimeError(
                f"No training trees assembled for {dataset}"
            )

        os.replace(temp_trees, trees_file)
        os.replace(temp_params, params_file)

    finally:
        temp_trees.unlink(missing_ok=True)
        temp_params.unlink(missing_ok=True)

    log(f"SUCCESS: {dataset} — {total} aligned trees")
    log(f"Trees:      {trees_file}")
    log(f"Parameters: {params_file}")

    return total


def main():
    log("=" * 60)
    log("STEP 4 DATA PREPARATION")
    log("Automatic source discovery: DISABLED")
    log("Using known Step 3 paths")
    log("=" * 60)

    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"Step 3 Data/ directory missing: {SOURCE}"
        )

    if not REPO.is_dir():
        raise FileNotFoundError(
            f"PhyloCNN repository missing: {REPO}"
        )

    if not (REPO / "Encoding").is_dir():
        raise FileNotFoundError(
            f"Original encoding directory missing: {REPO / 'Encoding'}"
        )

    counts = {}

    for dataset in DATASETS:
        counts[dataset] = assemble(dataset)

    log("\n" + "=" * 60)
    log("PREPARATION COMPLETED")
    log("=" * 60)

    for dataset, n in counts.items():
        log(f"{dataset:8s}: {n:,} trees")

    log(f"TOTAL: {sum(counts.values()):,} trees")

    if (REPO / "Test_Sets").is_dir():
        log("Original author-test directory: FOUND")
    else:
        log("WARNING: Original author-test directory not found.")
        log("Step 4 requires phyloCNN/Test_Sets for independent testing.")

    log("\nNo --source argument is required.")
    log("Step 4 will find the training files automatically.")


if __name__ == "__main__":
    main()
 