#!/usr/bin/env python3
"""
Compose a GitHub-native owner notification for failing docs pages.

Resolves each failing page's owner (GitHub ID from the in-page header) and
writes a Markdown issue body that @mentions the owners. The notify-owner
workflow opens a GitHub issue with this body; the @mention triggers GitHub's
own notification/email to each owner.

No email addresses and no SMTP are used or stored - notifications ride entirely
on GitHub's notification system, keyed off the owner's GitHub ID.

Usage:
  python notify_owner.py --failed-pages failed-pages.txt --body-out body.md
  python notify_owner.py --file docs/installation.mdx   # single-file demo
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resolve_owner import resolve  # noqa: E402


def gha_output(**kv):
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed-pages", type=Path)
    ap.add_argument("--file", type=Path)
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "amd/RyzenAI-SW"))
    ap.add_argument("--run-url", default=os.environ.get("RUN_URL", ""))
    ap.add_argument("--notify-email", default=os.environ.get("NOTIFY_EMAIL", ""),
                    help="shared DL also emailed the report (e.g. dl.ryzenai.support@amd.com)")
    ap.add_argument("--body-out", type=Path, default=Path("owner-issue-body.md"))
    args = ap.parse_args()

    pages: list[str] = []
    if args.file:
        pages = [args.file.as_posix()]
    elif args.failed_pages and args.failed_pages.exists():
        pages = [ln.strip() for ln in args.failed_pages.read_text(encoding="utf-8").splitlines() if ln.strip()]

    if not pages:
        print("No failing pages; nothing to notify.")
        gha_output(has_targets="false", assignees="", title="")
        args.body_out.write_text("", encoding="utf-8")
        return

    owners: dict[str, list[str]] = {}
    for p in pages:
        owners.setdefault(resolve(Path(p)), []).append(p)

    title = f"[Docs CI] {len(pages)} page(s) failed checks - action needed"
    lines = [f"A docs CI check failed on {len(pages)} page(s) in `{args.repo}`.", ""]
    if args.run_url:
        lines += [f"Failed run: {args.run_url}", ""]
    lines.append("Owners are @mentioned below so GitHub notifies them directly.")
    lines.append("")
    for oid in sorted(owners):
        lines.append(f"### @{oid}")
        for p in sorted(owners[oid]):
            lines.append(f"- `{p}`")
        lines.append("")
    if args.notify_email:
        lines.append(f"_A copy of this report is emailed to the Ryzen AI support "
                     f"distribution list ({args.notify_email})._")
        lines.append("")
    args.body_out.write_text("\n".join(lines), encoding="utf-8")

    assignees = ",".join(sorted(owners))
    gha_output(has_targets="true", assignees=assignees, title=title)

    print("=" * 70)
    print(f"OWNER NOTIFICATION (GitHub-native) - {len(pages)} page(s)")
    print("=" * 70)
    for oid in sorted(owners):
        print(f"  @{oid}: {len(owners[oid])} page(s)")
    print(f"Assignees: {assignees}")
    print(f"Title: {title}")
    print(f"Body written to: {args.body_out}")


if __name__ == "__main__":
    main()
