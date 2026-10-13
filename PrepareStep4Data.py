#!/usr/bin/env python3
"""Collect the verified Step 3 trees into phyloCNN/Training_Data/<NAME>/ (one file per dataset).

Output per dataset:
  <NAME>_trees.nwk          one Newick tree per line
  parameters_<NAME>.txt     matching parameter rows (tab-separated, with header)
  excluded_rows.tsv         parameter rows Step 3 could not simulate (left out by Step 3)
  skipped_chunks.tsv        chunks left out here because they are incomplete or inconsistent
  PREPARED.json             counts and the list of chunks used

Rules:
  - Only chunks with a DONE.json (verified by Step 3) are read; unfinished chunks are ignored.
  - A chunk whose trees match DONE.json is used. Trees Step 3 excluded (DONE.json "excluded")
    are expected to be missing.
  - A chunk with missing files, a different tree count, or tree/parameter IDs that do not
    match is skipped with a warning (it never stops the other chunks).
  - Targets come from raw/parameters_final.tsv.gz: for BiSSE these are the simulator's
    state-aligned values; for BD/BDEI/BDSS they equal the parameter design.

Run from the project root:  python PrepareStep4Data.py
Python 3.8 compatible.
"""

import csv
import gzip
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Data"
REPO = ROOT / "phyloCNN"
DEST = REPO / "Training_Data"

DATASETS = ("BD", "BDEI", "BDSS", "BiSSE")

# Newick strings of 500-tip trees can exceed csv's default field limit.
csv.field_size_limit(2 ** 31 - 1)


def log(message):
    print(message, flush=True)


def tree_id(value):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"invalid tree index {value!r}")
    return int(number)


def read_tsv(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"missing header in {path.name}")
        return list(reader.fieldnames), list(reader)


def read_chunk(chunk):
    """Return (header, rows, trees, excluded) of one verified chunk, or raise ValueError."""
    done = json.loads((chunk / "DONE.json").read_text())
    excluded = done.get("excluded", [])
    expected = done.get("kept", done.get("count"))

    trees_file = chunk / "raw" / "trees.tsv.gz"
    params_file = chunk / "raw" / "parameters_final.tsv.gz"
    for path in (trees_file, params_file):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty {path.relative_to(chunk)}")

    _, tree_rows = read_tsv(trees_file)
    header, rows = read_tsv(params_file)

    if "index" not in header:
        raise ValueError("parameter file has no 'index' column")
    if len(tree_rows) != len(rows):
        raise ValueError(f"{len(tree_rows)} trees but {len(rows)} parameter rows")
    if expected is not None and len(rows) != int(expected):
        raise ValueError(f"DONE.json expects {expected} trees, found {len(rows)}")

    excluded_ids = {int(e["index"]) for e in excluded}
    trees = []
    for k, (t, row) in enumerate(zip(tree_rows, rows)):
        tid, pid = tree_id(t["index"]), tree_id(row["index"])
        if tid != pid:
            raise ValueError(f"tree id {tid} != parameter id {pid} at row {k}")
        if tid in excluded_ids:
            raise ValueError(f"tree {tid} is listed as excluded but present")
        newick = t["tree"].strip()
        if not newick.endswith(";"):
            raise ValueError(f"invalid Newick tree for index {tid}")
        trees.append(newick)
    return header, rows, trees, excluded


def assemble(dataset):
    base = SOURCE / dataset
    chunks = sorted(marker.parent for marker in base.glob("full_*/chunks/*/DONE.json"))
    all_chunk_dirs = sorted(p for p in base.glob("full_*/chunks/*") if p.is_dir())
    if not chunks:
        raise FileNotFoundError(f"No verified Step 3 training chunks: {base}")

    destination = DEST / dataset
    destination.mkdir(parents=True, exist_ok=True)
    trees_file = destination / f"{dataset}_trees.nwk"
    params_file = destination / f"parameters_{dataset}.txt"
    temp_trees = trees_file.with_name(trees_file.name + ".tmp")
    temp_params = params_file.with_name(params_file.name + ".tmp")

    log(f"\n{'=' * 60}")
    log(f"DATASET: {dataset}")
    log(f"Verified chunks: {len(chunks)} (unfinished, ignored: {len(all_chunk_dirs) - len(chunks)})")

    total, header, writer = 0, None, None
    seen = set()
    used, skipped, excluded_rows = [], [], []

    try:
        with open(temp_trees, "w", encoding="utf-8") as tree_out, \
                open(temp_params, "w", encoding="utf-8", newline="") as param_out:
            for chunk in chunks:
                try:
                    chunk_header, rows, trees, excluded = read_chunk(chunk)
                    if header is not None and chunk_header != header:
                        raise ValueError("parameter columns differ from the other chunks")
                    duplicates = seen & {tree_id(r["index"]) for r in rows}
                    if duplicates:
                        raise ValueError(f"tree indices already used: {sorted(duplicates)[:5]}")
                except (ValueError, OSError, KeyError) as exc:
                    skipped.append({"chunk": chunk.name, "reason": str(exc)})
                    log(f"  {chunk.name}: SKIPPED ({exc})")
                    continue

                if header is None:
                    header = chunk_header
                    writer = csv.DictWriter(param_out, fieldnames=header, delimiter="\t",
                                            lineterminator="\n")
                    writer.writeheader()

                for row, newick in zip(rows, trees):
                    writer.writerow(row)
                    tree_out.write(newick + "\n")
                    seen.add(tree_id(row["index"]))
                total += len(rows)
                used.append(chunk.name)
                for e in excluded:
                    excluded_rows.append({"chunk": chunk.name, "index": e.get("index"), "reason": e.get("reason")})

                note = f", {len(excluded)} excluded by Step 3" if excluded else ""
                log(f"  {chunk.name}: {len(rows)} trees{note} (total={total})")

        if total == 0:
            raise RuntimeError(f"No usable training trees for {dataset}")
        os.replace(temp_trees, trees_file)
        os.replace(temp_params, params_file)
    finally:
        for temp in (temp_trees, temp_params):
            if temp.exists():
                temp.unlink()

    def write_tsv(path, fields, rows):
        with open(path, "w", encoding="utf-8", newline="") as handle:
            out = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            out.writeheader()
            out.writerows(rows)

    write_tsv(destination / "excluded_rows.tsv", ["chunk", "index", "reason"], excluded_rows)
    write_tsv(destination / "skipped_chunks.tsv", ["chunk", "reason"], skipped)
    summary = {
        "dataset": dataset, "trees": total,
        "chunks_used": len(used), "chunks_skipped": len(skipped),
        "chunks_unfinished": len(all_chunk_dirs) - len(chunks),
        "rows_excluded_by_step3": len(excluded_rows),
        "targets_from": "raw/parameters_final.tsv.gz",
        "used": used, "skipped": skipped,
    }
    (destination / "PREPARED.json").write_text(json.dumps(summary, indent=2) + "\n")

    log(f"SUCCESS: {dataset} - {total:,} trees from {len(used)} chunks"
        f" | skipped chunks: {len(skipped)} | rows excluded by Step 3: {len(excluded_rows)}")
    log(f"Trees:      {trees_file}")
    log(f"Parameters: {params_file}")
    return summary


def main():
    log("=" * 60)
    log("STEP 4 DATA PREPARATION (verified Step 3 chunks only)")
    log("=" * 60)

    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Step 3 Data/ directory missing: {SOURCE}")
    if not REPO.is_dir():
        raise FileNotFoundError(f"PhyloCNN repository missing: {REPO}")

    results = {}
    for dataset in DATASETS:
        try:
            results[dataset] = assemble(dataset)
        except (FileNotFoundError, RuntimeError) as exc:
            log(f"\n{dataset}: NOT PREPARED - {exc}")

    log("\n" + "=" * 60)
    log("PREPARATION SUMMARY")
    log("=" * 60)
    log(f"{'DATASET':8s} {'TREES':>9s} {'CHUNKS':>7s} {'SKIPPED':>8s} {'UNFINISHED':>11s} {'EXCLUDED':>9s}")
    for dataset in DATASETS:
        s = results.get(dataset)
        if s is None:
            log(f"{dataset:8s} {'-':>9s}  not prepared")
            continue
        log(f"{dataset:8s} {s['trees']:>9,d} {s['chunks_used']:>7d} {s['chunks_skipped']:>8d} "
            f"{s['chunks_unfinished']:>11d} {s['rows_excluded_by_step3']:>9d}")
    log(f"TOTAL    {sum(s['trees'] for s in results.values()):>9,d}")
    if any(s["chunks_skipped"] for s in results.values()):
        log("\nSkipped chunks are listed in phyloCNN/Training_Data/<NAME>/skipped_chunks.tsv")

    if not (REPO / "Test_Sets").is_dir():
        log("WARNING: phyloCNN/Test_Sets not found (needed for the independent author test).")
    return 0 if len(results) == len(DATASETS) else 1


if __name__ == "__main__":
    sys.exit(main())
