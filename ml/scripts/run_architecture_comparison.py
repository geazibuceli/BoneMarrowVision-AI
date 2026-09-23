"""Run isolated smoke checks, then matched training and evaluation for selected architectures."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from audit_existing_splits import audit_splits
from bone_marrow_vision.data.dataset import SPLIT_NAMES, sha256_file, verify_split_manifest

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = ("resnet50", "efficientnet_b0", "convnext_tiny")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/experiments/resnet50_comparison")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/architecture_comparison_seed42")
    )
    parser.add_argument(
        "--full", action="store_true", help="Continue into 100-epoch runs after smoke checks"
    )
    args = parser.parse_args()
    if len(set(args.architectures)) != len(args.architectures):
        parser.error("Architectures must be unique")
    os.chdir(ROOT)
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    verify_split_manifest(data)
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "TORCH_HOME": str(ROOT / "models/torch-cache"),
        "PYTHONUNBUFFERED": "1",
    }
    state = {"status": "running", "started_at": datetime.now(UTC).isoformat(), "steps": []}

    def save_status():
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(output / "status.json")

    def run(name, script, *arguments):
        step = {"name": name, "status": "running", "started_at": datetime.now(UTC).isoformat()}
        state["steps"].append(step)
        save_status()
        command = [sys.executable, "-u", str(ROOT / "ml/scripts" / script), *map(str, arguments)]
        step["command"] = command
        with (output / f"{name}.stdout.log").open("w") as stdout:
            with (output / f"{name}.stderr.log").open("w") as stderr:
                result = subprocess.run(
                    command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
                )
        step["returncode"] = result.returncode
        step["finished_at"] = datetime.now(UTC).isoformat()
        step["status"] = "completed" if result.returncode == 0 else "failed"
        save_status()
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect its logs under {output}")

    save_status()
    try:
        # Freeze configs and record source hashes before starting any training.
        configs = {}
        for architecture in args.architectures:
            source = ROOT / "ml/configs" / f"{architecture}.yaml"
            target = output / source.name
            shutil.copyfile(source, target)
            configs[architecture] = target
        parsed = [yaml.safe_load(p.read_text()) for p in configs.values()]
        shared = [{k: v for k, v in c.items() if k != "architecture"} for c in parsed]
        if any(config != shared[0] for config in shared[1:]):
            raise ValueError("Comparison configs differ in more than architecture")
        state["inputs"] = {
            "split_manifest_sha256": sha256_file(data / "manifest.json"),
            "split_hashes": {name: sha256_file(data / f"{name}.json") for name in SPLIT_NAMES},
            "config_hashes": {name: sha256_file(path) for name, path in configs.items()},
        }
        smoke = output / "smoke_splits"
        smoke.mkdir()
        for name in SPLIT_NAMES:
            rows = json.loads((data / f"{name}.json").read_text())
            counts = Counter()
            selected = []
            for row in rows:
                if counts[row["label"]] < 2:
                    selected.append(row)
                    counts[row["label"]] += 1
            (smoke / f"{name}.json").write_text(json.dumps(selected, indent=2))
        smoke_manifest = audit_splits(smoke)
        if smoke_manifest["cross_split_duplicates"] or smoke_manifest["conflicting_labels"]:
            raise ValueError("Smoke split audit failed")
        (smoke / "manifest.json").write_text(json.dumps(smoke_manifest, indent=2))
        for architecture, config in configs.items():
            run(
                f"smoke_{architecture}",
                "train.py",
                "--config",
                config,
                "--data-dir",
                smoke,
                "--checkpoint",
                output / f"smoke_{architecture}/best.pt",
                "--epochs",
                "1",
            )
        if args.full:
            for architecture, config in configs.items():
                verify_split_manifest(data)
                if any(
                    sha256_file(data / f"{name}.json") != digest
                    for name, digest in state["inputs"]["split_hashes"].items()
                ):
                    raise ValueError("Comparison splits changed during execution")
                run(
                    f"train_{architecture}",
                    "train.py",
                    "--config",
                    config,
                    "--data-dir",
                    data,
                    "--checkpoint",
                    output / architecture / "best.pt",
                )
            for architecture in configs:
                run(
                    f"evaluate_{architecture}",
                    "evaluate.py",
                    "--data-dir",
                    data,
                    "--checkpoint",
                    output / architecture / "best.pt",
                    "--output",
                    output / architecture / "test_report.json",
                )
                run(
                    f"analyze_{architecture}",
                    "analyze_results.py",
                    "--data-dir",
                    data,
                    "--checkpoint",
                    output / architecture / "best.pt",
                    "--metrics",
                    output / architecture / "test_report.json",
                    "--output-dir",
                    output / architecture / "analysis",
                )
            reports = {
                name: json.loads((output / name / "test_report.json").read_text())
                for name in configs
            }
            names = list(reports)
            if any(r["class_names"] != reports[names[0]]["class_names"] for r in reports.values()):
                raise ValueError("Evaluation class vocabularies differ")
            if len({r["provenance"]["test_split"]["sha256"] for r in reports.values()}) != 1:
                raise ValueError("Evaluations used different test splits")
            metrics = (
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
            )
            table = [
                "# Architecture comparison",
                "",
                "Matched single-seed runs; patient/slide isolation is unverified.",
                "",
                "| Metric | " + " | ".join(names) + " |",
                "|---|" + "---:|" * len(names),
            ]
            table += [
                f"| {key} | " + " | ".join(f"{reports[name][key]:.6f}" for name in names) + " |"
                for key in metrics
            ]
            (output / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
        state["status"] = "completed" if args.full else "smoke_completed"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = str(error)
        raise
    finally:
        state["finished_at"] = datetime.now(UTC).isoformat()
        save_status()


if __name__ == "__main__":
    main()
