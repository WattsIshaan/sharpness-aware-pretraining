#!/usr/bin/env python3
"""Remove all folders in 1b-experiments except cpt_checkpoints."""

import shutil
from pathlib import Path

BASE = Path("/tmp/iwatts/outputs/1b-experiments")
KEEP = "cpt_checkpoints"

def main():
    if not BASE.exists():
        print(f"Path does not exist: {BASE}")
        return 1

    for item in BASE.iterdir():
        if item.is_dir() and item.name != KEEP:
            print(f"Removing {item}")
            shutil.rmtree(item)

    print("Done. Remaining:")
    for item in sorted(BASE.iterdir()):
        print(f"  {item.name}")
    return 0

if __name__ == "__main__":
    exit(main())
