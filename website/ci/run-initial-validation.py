#!/usr/bin/env python3
"""
Initial validation runner for Ryzen AI documentation.

Run this ONCE before the docs site goes live, and then again whenever you
want a full re-validation of all code snippets and models.

This script orchestrates:
  1. Code block extraction and syntax checking (works anywhere)
  2. Code-from-source sample testing (requires Ryzen AI env)
  3. Model validation (requires AMD hardware + Ryzen AI env)

Usage:
    # Phase 1 only (syntax checks, no hardware needed):
    python run-initial-validation.py --phase syntax

    # Phase 2 only (run code samples on hardware):
    python run-initial-validation.py --phase code

    # Phase 3 only (model validation on hardware):
    python run-initial-validation.py --phase models

    # All phases:
    python run-initial-validation.py --phase all

    # Dry run (show what would be tested):
    python run-initial-validation.py --phase all --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CI_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CI_DIR / "validation-results"
DOCS_DIR = REPO_ROOT / "docs"
CODE_SAMPLES_DIR = CI_DIR.parent / "code-samples"

DIVIDER = "=" * 70


def print_header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(f"{DIVIDER}\n")


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase_syntax(dry_run: bool = False) -> dict:
    """Phase 1: Extract all code blocks and check syntax."""
    print_header("Phase 1: Code Block Extraction & Syntax Check")

    output_file = RESULTS_DIR / "syntax-check-results.json"
    cmd = [
        sys.executable,
        str(CI_DIR / "extract_code_blocks.py"),
        "--run",
        "--syntax-only",
        "--output-json", str(output_file),
    ]

    if dry_run:
        print(f"  Would run: {' '.join(cmd)}")
        return {"phase": "syntax", "status": "dry-run"}

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if output_file.exists():
        data = json.loads(output_file.read_text())
        print(f"\n  Total blocks: {data['total']}")
        print(f"  Passed:       {data['passed']}")
        print(f"  Failed:       {data['failed']}")
        print(f"  Skipped:      {data['skipped']}")
        if data["failures"]:
            print("\n  Failures:")
            for f in data["failures"]:
                print(f"    - {f['file']}:{f['line']} ({f['language']})")
                for line in f["error"].split("\n")[:2]:
                    print(f"      {line}")
        return data

    return {"phase": "syntax", "status": "error", "returncode": result.returncode}


def run_phase_code(dry_run: bool = False) -> dict:
    """Phase 2: Run code-from-source samples on hardware."""
    print_header("Phase 2: Code Sample Execution (Hardware Required)")

    results = {"phase": "code", "samples": []}
    sample_files = sorted(CODE_SAMPLES_DIR.rglob("*.py"))

    if not sample_files:
        print("  No code samples found.")
        return results

    print(f"  Found {len(sample_files)} code sample(s):\n")
    for f in sample_files:
        rel = f.relative_to(REPO_ROOT)
        print(f"    - {rel}")

    if dry_run:
        return {"phase": "code", "status": "dry-run", "count": len(sample_files)}

    for sample in sample_files:
        rel = str(sample.relative_to(REPO_ROOT))
        print(f"\n  Running: {rel}")

        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(sample)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
        elapsed = time.perf_counter() - start

        entry = {
            "file": rel,
            "passed": proc.returncode == 0,
            "elapsed_s": round(elapsed, 2),
        }
        if proc.returncode != 0:
            entry["stderr"] = proc.stderr[:500]
            print(f"    FAIL ({elapsed:.1f}s)")
            for line in proc.stderr.split("\n")[:3]:
                print(f"      {line}")
        else:
            print(f"    PASS ({elapsed:.1f}s)")
            if proc.stdout.strip():
                for line in proc.stdout.strip().split("\n")[:3]:
                    print(f"      {line}")

        results["samples"].append(entry)

    output_file = RESULTS_DIR / "code-sample-results.json"
    output_file.write_text(json.dumps(results, indent=2))
    passed = sum(1 for s in results["samples"] if s["passed"])
    print(f"\n  Code samples: {passed}/{len(results['samples'])} passed")
    return results


def run_phase_models(dry_run: bool = False) -> dict:
    """Phase 3: Model validation on AMD hardware."""
    print_header("Phase 3: Model Validation (Hardware Required)")

    config_path = CI_DIR / "hardware-registry.json"
    model_validation_dir = CI_DIR / "model-validation"
    discover_script = model_validation_dir / "discover.py"

    candidates_file = RESULTS_DIR / "model-candidates.json"

    # Step 1: Discover
    print("  Step 1: Discovering model candidates...")
    cmd = [
        sys.executable, str(discover_script),
        "--config", str(config_path),
        "--output", str(candidates_file),
    ]

    if dry_run:
        print(f"  Would run: {' '.join(cmd)}")
        # Still run discover to show what would be tested
        subprocess.run(cmd, cwd=str(REPO_ROOT))
        if candidates_file.exists():
            candidates = json.loads(candidates_file.read_text())
            for domain, models in candidates.items():
                print(f"    {domain}: {len(models)} candidates")
                for m in models:
                    print(f"      - {m}")
        return {"phase": "models", "status": "dry-run"}

    subprocess.run(cmd, cwd=str(REPO_ROOT))

    if not candidates_file.exists():
        print("  ERROR: discover.py did not produce candidates file")
        return {"phase": "models", "status": "error"}

    candidates = json.loads(candidates_file.read_text())

    # Step 2: Run harnesses per domain
    all_results = []

    for domain, harness_name in [("llm", "harness_llm.py"), ("vision", "harness_vision.py"), ("audio", "harness_audio.py")]:
        models = candidates.get(domain, [])
        if not models:
            continue

        harness = model_validation_dir / harness_name
        print(f"\n  Step 2: Testing {len(models)} {domain} model(s)...")

        for model_id in models:
            output_file = RESULTS_DIR / f"{domain}-{model_id.replace('/', '_')}.json"
            cmd = [
                sys.executable, str(harness),
                "--model-id", model_id,
                "--device", "npu",
                "--output", str(output_file),
            ]
            print(f"    Testing: {model_id}")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if output_file.exists():
                result = json.loads(output_file.read_text())
                passed = result.get("functional", {}).get("passed", False)
                print(f"      {'PASS' if passed else 'FAIL'}")
                all_results.append(result)
            else:
                print(f"      ERROR (no output)")
                all_results.append({
                    "model_id": model_id,
                    "domain": domain,
                    "functional": {"passed": False, "error": proc.stderr[:300]},
                })

    # Step 3: Gate
    print(f"\n  Step 3: Running gate to generate model-list-data.json...")
    gate_script = model_validation_dir / "gate.py"
    model_list_output = REPO_ROOT / "docs" / "reference" / "model-list-data.json"

    # Write combined results for gate
    combined_file = RESULTS_DIR / "combined-results.json"
    combined_file.write_text(json.dumps(all_results, indent=2))

    cmd = [
        sys.executable, str(gate_script),
        "--results", str(RESULTS_DIR),
        "--output", str(model_list_output),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT))

    verified = sum(1 for r in all_results if r.get("functional", {}).get("passed"))
    print(f"\n  Models: {verified}/{len(all_results)} verified")

    return {
        "phase": "models",
        "total": len(all_results),
        "verified": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run initial validation of all docs code and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["syntax", "code", "models", "all"],
        default="all",
        help="Which validation phase to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without executing",
    )
    args = parser.parse_args()

    print_header("Ryzen AI Documentation - Initial Validation")
    print(f"  Repository:  {REPO_ROOT}")
    print(f"  Docs dir:    {DOCS_DIR}")
    print(f"  Results dir:  {RESULTS_DIR}")
    print(f"  Phase:       {args.phase}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Python:      {sys.executable}")
    print(f"  Timestamp:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    ensure_results_dir()
    summary = {}

    if args.phase in ("syntax", "all"):
        summary["syntax"] = run_phase_syntax(dry_run=args.dry_run)

    if args.phase in ("code", "all"):
        summary["code"] = run_phase_code(dry_run=args.dry_run)

    if args.phase in ("models", "all"):
        summary["models"] = run_phase_models(dry_run=args.dry_run)

    # Final summary
    print_header("Validation Summary")
    summary_file = RESULTS_DIR / "validation-summary.json"
    summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"  Full results written to: {RESULTS_DIR}")
    print(f"  Summary: {summary_file}")

    # Check for any failures
    has_failures = False
    if "syntax" in summary:
        syntax = summary["syntax"]
        if isinstance(syntax, dict) and syntax.get("failed", 0) > 0:
            has_failures = True

    if "code" in summary:
        code = summary["code"]
        if isinstance(code, dict):
            samples = code.get("samples", [])
            if any(not s.get("passed") for s in samples):
                has_failures = True

    if has_failures:
        print("\n  STATUS: Some tests FAILED. Review the results above.")
        sys.exit(1)
    else:
        print("\n  STATUS: All tests PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
