#!/usr/bin/env python3
"""
Fail CI if any docs page is missing an owner header.

Every page under docs/ must declare an owner inline:
    {/* owner: <github-id> */}

This is our guarantee that "every docs page has ownership" - it complements
CODEOWNERS (which routes GitHub review) by recording the owner's GitHub ID on
the page itself. Notifications are sent GitHub-natively by @mentioning the ID.
"""

import re
import sys
from pathlib import Path

# {/* owner: <id> */}  (an optional legacy "| email" tail is tolerated)
OWNER_RE = re.compile(r"\{/\*\s*owner:\s*[^*|]+?\s*(?:\|[^*]*)?\*/\}")
DOCS = Path("docs")


def main() -> None:
    missing = []
    for mdx in sorted(DOCS.rglob("*.mdx")):
        text = mdx.read_text(encoding="utf-8", errors="replace")
        if not OWNER_RE.search(text):
            missing.append(mdx.as_posix())
    total = len(list(DOCS.rglob("*.mdx")))
    if missing:
        print(f"ERROR: {len(missing)}/{total} docs pages have no owner header:")
        for m in missing:
            print(f"  - {m}")
        print("\nAdd a header: {/* owner: <github-id> */}")
        sys.exit(1)
    print(f"OK: all {total} docs pages have an owner header.")


if __name__ == "__main__":
    main()
