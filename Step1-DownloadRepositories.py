#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()

REPOS = {
    "phyloCNN": "https://github.com/manolofperez/phyloCNN.git",
    "phylodeep": "https://github.com/evolbioinfo/phylodeep.git",
    "deeptimelearning": "https://github.com/JakubVoz/deeptimelearning.git",
}

failed = []

for name, url in REPOS.items():
    dest = ROOT / name
    print(f"\n[DOWNLOADING] {name}", flush=True)

    if (dest / ".git").exists():
        print("[SKIP] Already cloned", flush=True)
        continue

    if dest.exists():
        print("[ERROR] Directory already exists:", dest, flush=True)
        failed.append(name)
        continue

    result = subprocess.run(
        ["git", "clone", "--progress", url, str(dest)]
    )

    if result.returncode == 0:
        print("[SUCCESS]", name, flush=True)
    else:
        print("[FAILED]", name, flush=True)
        failed.append(name)

print("\n========== STEP 1 SUMMARY ==========")

for name in REPOS:
    status = "READY" if (ROOT / name / ".git").exists() else "MISSING"
    print(f"{name:20s} {status}")

if failed:
    sys.exit("Failed: " + ", ".join(failed))

print("\nSTEP 1 COMPLETED SUCCESSFULLY")
