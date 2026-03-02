#!/usr/bin/env python3
"""
Read test result JSON files from a directory and generate combined model-list-data.json.
Marks models as verified:true when functional_pass is true.
"""

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    """Load all *-results.json files from the results directory."""
    combined = []
    if not results_dir.exists():
        return combined

    for path in sorted(results_dir.glob("*-results.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Handle both single-result and list formats
            if isinstance(data, list):
                combined.extend(data)
            else:
                combined.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Skipping {path}: {e}")
    return combined


def build_model_list(records: list[dict]) -> list[dict]:
    """
    Build model list with verification status.
    Each record should have functional_pass; verified = functional_pass.
    """
    model_list = []
    for r in records:
        functional = r.get("functional", {})
        functional_pass = functional.get("passed", r.get("functional_pass", False))
        entry = {
            "model_id": r.get("model_id", r.get("model", "unknown")),
            "domain": r.get("domain", r.get("task", "unknown")),
            "device": r.get("device", "npu"),
            "test_date": r.get("test_date", ""),
            "verified": bool(functional_pass),
            "functional_pass": functional_pass,
        }
        # Preserve other fields (e.g., duration, timestamp) if present
        for k, v in r.items():
            if k not in entry:
                entry[k] = v
        model_list.append(entry)
    return model_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate test results into model-list-data.json."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("."),
        help="Directory containing *-results.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model-list-data.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    records = load_results(args.results)
    model_list = build_model_list(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model_list, f, indent=2)

    verified = sum(1 for m in model_list if m.get("verified"))
    print(f"Processed {len(records)} records, {verified}/{len(model_list)} verified -> {args.output}")


if __name__ == "__main__":
    main()
