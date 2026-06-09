#!/usr/bin/env python3
"""
Resolve the owner of a documentation page from its in-file owner header.

Ownership is the page owner's GitHub ID only (no email). Notifications are sent
GitHub-natively by @mentioning the ID, which triggers GitHub's own email.

Given a .mdx/.md file, print "<github_id>" and emit a GitHub Actions output.
Falls back to the default owner when no header is present.
"""

import os
import re
import sys
from pathlib import Path

DEFAULT_OWNER_ID = "dwithchenna"

# {/* owner: <id> */}  (an optional legacy "| email" tail is ignored)
OWNER_RE = re.compile(r"\{/\*\s*owner:\s*(?P<id>[^|*]+?)\s*(?:\|[^*]*)?\*/\}")


def resolve(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_OWNER_ID
    m = OWNER_RE.search(text)
    return m.group("id").strip() if m else DEFAULT_OWNER_ID


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: resolve_owner.py <path-to-page>", file=sys.stderr)
        sys.exit(2)
    owner_id = resolve(Path(sys.argv[1]))
    print(owner_id)
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"owner_id={owner_id}\n")


if __name__ == "__main__":
    main()
