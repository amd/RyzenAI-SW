#!/usr/bin/env python3
"""
Append a CI run record to .github/scripts/ci-history.json.

Called at the end of each CI workflow. Captures what ran, when, pass/fail, and
per-page results (with the resolved owner) so the dashboard can show history.

Reads:
  - code-results.json (optional) from extract_code_blocks.py for per-page status
  - run metadata from flags / GITHUB_* env vars

Usage:
  python .github/scripts/record_run.py --workflow "Test Code Samples" --status success \
      --code-results code-results.json --history .github/scripts/ci-history.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resolve_owner import resolve  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def pages_from_code_results(path: Path, docs_root: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    seen: dict[str, dict] = {}
    for r in data:
        page = r.get("page", "")
        status = r.get("status")
        if status not in ("pass", "fail"):
            continue
        # A page fails if any block fails.
        prev = seen.get(page)
        page_status = "fail" if (status == "fail" or (prev and prev["status"] == "fail")) else "pass"
        rel = page.split("docs/", 1)[-1]
        owner_id = resolve(docs_root / rel) if (docs_root / rel).exists() else resolve(Path(page))
        seen[page] = {
            "page": rel,
            "owner_id": owner_id,
            "check": "code-execution",
            "status": page_status,
        }
        if status == "fail":
            seen[page]["detail"] = (r.get("detail") or "")[:200]
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--status", required=True, choices=["success", "failure"])
    ap.add_argument("--job", default=None, help="job name")
    ap.add_argument("--script", default=None, help="primary script run")
    ap.add_argument("--code-results", type=Path)
    ap.add_argument("--docs", type=Path, default=Path("docs"))
    ap.add_argument("--history", type=Path, default=Path(".github/scripts/ci-history.json"))
    args = ap.parse_args()

    record = {
        "run_id": os.environ.get("GITHUB_RUN_ID", f"local-{datetime.now():%Y%m%d-%H%M%S}"),
        "workflow": args.workflow,
        "event": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "actor": os.environ.get("GITHUB_ACTOR", "local"),
        "branch": os.environ.get("GITHUB_REF_NAME", "local"),
        "sha": (os.environ.get("GITHUB_SHA", "local"))[:12],
        "started_at": os.environ.get("RUN_STARTED_AT", now_iso()),
        "finished_at": now_iso(),
        "status": args.status,
        "jobs": [],
        "pages": pages_from_code_results(args.code_results, args.docs),
    }
    if args.job:
        record["jobs"].append({"name": args.job, "script": args.script or "", "status": args.status})

    hist_path = args.history
    if hist_path.exists():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
    else:
        hist = {"schema": 1, "runs": []}
    hist["runs"].insert(0, record)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(hist, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded run {record['run_id']} ({args.workflow}: {args.status}) "
          f"with {len(record['pages'])} page result(s).")


if __name__ == "__main__":
    main()
