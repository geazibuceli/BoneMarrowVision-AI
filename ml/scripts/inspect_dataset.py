from __future__ import annotations

import argparse
import json
from pathlib import Path

from bone_marrow_vision.data.inspection import inspect_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a bone marrow image dataset.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="reports/dataset_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(inspect_dataset(args.data_dir, Path(args.output)), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Dataset inspection failed: {error}") from None
