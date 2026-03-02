#!/usr/bin/env python3
"""
Automatically update ci_validated front matter and CIStatus component
in MDX files based on CI test results.

Usage:
  python update_ci_status.py --results results.json
  python update_ci_status.py --passed-files passed.txt
  python update_ci_status.py --all-passed    # Mark all MDX files as validated
  python update_ci_status.py --reset         # Reset all to false
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent.parent


def find_mdx_files() -> list[Path]:
    """Find all content MDX files (not READMEs, templates, or website infra)."""
    skip_names = {"readme.mdx", "readme_c++.mdx", "advanced_quant_readme.mdx"}
    files = []
    for f in DOCS_ROOT.rglob("*.mdx"):
        if "website" in f.parts or "templates" in f.parts:
            continue
        if f.name.lower() in skip_names:
            continue
        files.append(f)
    return sorted(files)


def update_file(filepath: Path, validated: bool, run_date: str | None = None) -> bool:
    """Update a single MDX file's CI status. Returns True if changed."""
    content = filepath.read_text(encoding="utf-8")

    original = content
    today = run_date or date.today().isoformat()

    # Update front matter: ci_validated: false -> ci_validated: true (or vice versa)
    val_str = "true" if validated else "false"
    content = re.sub(
        r"^ci_validated:\s*(true|false)\s*$",
        f"ci_validated: {val_str}",
        content,
        flags=re.MULTILINE,
    )

    # Update or add ci_last_run date in front matter
    if validated and run_date:
        if "ci_last_run:" in content:
            content = re.sub(
                r"^ci_last_run:\s*.*$",
                f"ci_last_run: {today}",
                content,
                flags=re.MULTILINE,
            )
        else:
            content = re.sub(
                r"^(ci_validated:\s*(?:true|false))\s*$",
                f"\\1\nci_last_run: {today}",
                content,
                flags=re.MULTILINE,
            )

    # Update component: <CIStatus validated={false} /> -> <CIStatus validated={true} />
    if validated:
        content = re.sub(
            r'<CIStatus\s+validated=\{false\}\s*/>',
            f'<CIStatus validated={{true}} lastRun="{today}" />',
            content,
        )
    else:
        content = re.sub(
            r'<CIStatus\s+validated=\{true\}[^/]*/>' ,
            '<CIStatus validated={false} />',
            content,
        )

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False


def get_source_files_from_results(results_path: Path) -> dict[str, bool]:
    """
    Parse extract_code_blocks.py JSON output to determine which source
    MDX files had all their code blocks pass.
    """
    data = json.loads(results_path.read_text(encoding="utf-8"))

    file_status: dict[str, set] = {}
    failures_set: set[str] = set()

    for failure in data.get("failures", []):
        failures_set.add(failure["file"])

    # If total == passed + skipped (no failures), all passed
    if data.get("failed", 0) == 0:
        return {"__all__": True}

    return {"__failures__": failures_set}


def main():
    parser = argparse.ArgumentParser(
        description="Update ci_validated status in MDX files based on CI results"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", type=Path, help="JSON results from extract_code_blocks.py")
    group.add_argument("--passed-files", type=Path, help="Text file with one passed MDX path per line")
    group.add_argument("--all-passed", action="store_true", help="Mark all MDX files as CI validated")
    group.add_argument("--reset", action="store_true", help="Reset all MDX files to not validated")
    parser.add_argument("--date", type=str, default=date.today().isoformat(), help="CI run date (YYYY-MM-DD)")
    args = parser.parse_args()

    mdx_files = find_mdx_files()
    updated = 0

    if args.all_passed:
        for f in mdx_files:
            if update_file(f, validated=True, run_date=args.date):
                updated += 1
                print(f"  ✓ {f.relative_to(DOCS_ROOT)}")

    elif args.reset:
        for f in mdx_files:
            if update_file(f, validated=False):
                updated += 1
                print(f"  ○ {f.relative_to(DOCS_ROOT)}")

    elif args.passed_files:
        passed = set(args.passed_files.read_text().strip().splitlines())
        for f in mdx_files:
            rel = str(f.relative_to(DOCS_ROOT))
            if rel in passed:
                if update_file(f, validated=True, run_date=args.date):
                    updated += 1
                    print(f"  ✓ {rel}")

    elif args.results:
        result = get_source_files_from_results(args.results)
        if "__all__" in result:
            for f in mdx_files:
                if update_file(f, validated=True, run_date=args.date):
                    updated += 1
                    print(f"  ✓ {f.relative_to(DOCS_ROOT)}")
        else:
            failures = result.get("__failures__", set())
            for f in mdx_files:
                rel = str(f.relative_to(DOCS_ROOT))
                is_failed = any(rel in fail_path for fail_path in failures)
                if not is_failed:
                    if update_file(f, validated=True, run_date=args.date):
                        updated += 1
                        print(f"  ✓ {rel}")
                else:
                    if update_file(f, validated=False):
                        updated += 1
                        print(f"  ✗ {rel} (has failures)")

    print(f"\nUpdated {updated}/{len(mdx_files)} files")


if __name__ == "__main__":
    main()
