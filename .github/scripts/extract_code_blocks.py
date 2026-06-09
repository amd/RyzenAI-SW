#!/usr/bin/env python3
"""
Extract and validate fenced code blocks from Mintlify .mdx docs.

Design (test-by-default / opt-out):
  - EVERY fenced block in a runnable language is EXECUTED by default.
  - Add `notest` to the fence info string to skip a block entirely.
  - Optional device tags `cpu` / `gpu` / `npu` declare which accelerator a block
    needs. With `--device X`, a block runs only if it is untagged (runs
    everywhere) or tagged with X.

This ports the useful ideas from the AMD Playbooks test runner
(amd/playbooks .github/scripts/run_playbook_tests.py) but (1) inverts the
default to "test everything", and (2) uses MDX-valid comment syntax
`{/* ... */}` instead of HTML comments `<!-- ... -->`.

================================ Capabilities ================================
All optional. None are required, and no docs use them yet - they are available
for authors when a page needs them.

Per-block, in the fence info string (after the language):
    ```python                  -> executed (default)
    ```python notest           -> skipped entirely
    ```python npu              -> device-scoped (cpu | gpu | npu)
    ```python timeout=600      -> per-block timeout (seconds)
    ```bash workdir=examples   -> run in <page-dir>/examples
    ```bash continue_on_error=true   -> a failure doesn't fail the page
    ```python setup=activate-venv    -> run a named @setup first
    ```text / ```json          -> ignored (non-runnable language)

Inline marker:
    any line ending with `#hide` is executed but meant to be hidden from the
    rendered site (shown as [hidden] in CI logs).

Page-level comment directives (MDX comments, invisible when rendered):
    Scope blocks (wrap one or more code blocks):
        {/* @os:windows */} ... {/* @os:end */}        (windows | linux)
        {/* @device:npu */} ... {/* @device:end */}    (cpu,gpu,npu lists ok)
    Reusable named setup (OS-scoped):
        {/* @setup:id=activate-venv command="..." */}
    Reusable device-aware variables, referenced in code as ${name}:
        {/* @var:id=model device=npu value="..." */}
    Inline a shared include (from docs/_includes via _includes/registry.json):
        {/* @require:common-install */}

Outputs:
  - --output-json: per-block results
  - --failed-pages: pages with >=1 failing block (consumed by notify-owner)

Usage:
  python extract_code_blocks.py --syntax-only                  # cloud
  python extract_code_blocks.py --run                          # hardware: all
  python extract_code_blocks.py --run --device npu             # npu + untagged
  python extract_code_blocks.py --run --platform windows       # os filter
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

FENCE_RE = re.compile(r"^```([^\n]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)

RUNNABLE = {"python", "bash", "sh", "shell", "powershell", "pwsh", "ps1",
            "cmd", "bat", "batch"}
DEVICE_TAGS = {"cpu", "gpu", "npu"}
SKIP_TAG = "notest"

# MDX comment directives: {/* @tag ... */}
SETUP_RE = re.compile(r"\{/\*\s*@setup:(.+?)\*/\}")
VAR_RE = re.compile(r"\{/\*\s*@var:(.+?)\*/\}")
REQUIRE_RE = re.compile(r"\{/\*\s*@require:([a-z0-9\-,]+)\s*\*/\}")
OS_OPEN_RE = re.compile(r"\{/\*\s*@os:(windows|linux)\s*\*/\}")
OS_CLOSE_RE = re.compile(r"\{/\*\s*@os:end\s*\*/\}")
DEVICE_OPEN_RE = re.compile(r"\{/\*\s*@device:([\w,]+)\s*\*/\}")
DEVICE_CLOSE_RE = re.compile(r"\{/\*\s*@device:end\s*\*/\}")

ATTR_RE = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_attr_string(s: str) -> dict:
    """Parse `key=value` / `key="value with spaces"` pairs (comment attrs)."""
    attrs = {}
    for m in ATTR_RE.finditer(s):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else m.group(3)
        if key == "timeout":
            val = int(val)
        elif key in ("continue_on_error", "hidden"):
            val = str(val).lower() == "true"
        attrs[key] = val
    return attrs


def parse_fence(info: str):
    """Split a fence info string into (lang, tags:set, attrs:dict).

    Bare words become tags (e.g. notest, npu); `key=value` become attrs.
    Fence values cannot contain spaces - use a named @setup for that.
    """
    tokens = info.strip().split()
    if not tokens:
        return None
    lang = tokens[0].lower()
    tags, attrs = set(), {}
    for t in tokens[1:]:
        if "=" in t:
            k, v = t.split("=", 1)
            if k == "timeout":
                try:
                    v = int(v)
                except ValueError:
                    pass
            elif k in ("continue_on_error", "hidden"):
                v = v.lower() == "true"
            attrs[k] = v
        else:
            tags.add(t.lower())
    return lang, tags, attrs


def find_nested_blocks(content, open_re, close_re):
    """Return (value, start, end) for nested directive blocks, innermost first."""
    close_spans = [(m.start(), m.end()) for m in close_re.finditer(content)]
    close_starts = {s for s, _ in close_spans}
    events = []  # (pos, kind, value, end)
    for m in open_re.finditer(content):
        if m.start() in close_starts:  # guard permissive open patterns
            continue
        events.append((m.start(), "open", m.group(1), m.end()))
    for s, e in close_spans:
        events.append((s, "close", "", e))
    events.sort(key=lambda x: x[0])

    stack, blocks = [], []
    for pos, kind, value, end in events:
        if kind == "open":
            stack.append((value, pos))
        elif kind == "close" and stack:
            ovalue, opos = stack.pop()
            blocks.append((ovalue, opos, end))
    blocks.sort(key=lambda b: b[2] - b[1])
    return blocks


def infer_scope(blocks, pos) -> str:
    for value, start, end in blocks:
        if start <= pos < end:
            return value
    return "all"


# --------------------------------------------------------------------------- #
# Reusable @setup / @var / @require
# --------------------------------------------------------------------------- #
def extract_setup_definitions(content: str) -> dict:
    defs: dict[str, dict[str, str]] = {}
    os_blocks = find_nested_blocks(content, OS_OPEN_RE, OS_CLOSE_RE)
    for m in SETUP_RE.finditer(content):
        attrs = parse_attr_string(m.group(1))
        sid, cmd = attrs.get("id"), attrs.get("command")
        if not sid or not cmd:
            continue
        platform = infer_scope(os_blocks, m.start())
        defs.setdefault(sid, {})
        if platform == "all":
            defs[sid]["windows"] = defs[sid]["linux"] = cmd
        else:
            defs[sid][platform] = cmd
    return defs


def resolve_setup(value, defs, platform) -> Optional[str]:
    if not value:
        return None
    if value in defs:
        return defs[value].get(platform)
    return value  # raw command (backward compatible)


def extract_var_definitions(content: str) -> dict:
    defs: dict[str, dict[str, str]] = {}
    device_blocks = find_nested_blocks(content, DEVICE_OPEN_RE, DEVICE_CLOSE_RE)
    for m in VAR_RE.finditer(content):
        attrs = parse_attr_string(m.group(1))
        vid, val = attrs.get("id"), attrs.get("value")
        if not vid or val is None:
            continue
        device_value = attrs.get("device") or infer_scope(device_blocks, m.start())
        defs.setdefault(vid, {})
        if device_value == "all":
            defs[vid]["all"] = val
        else:
            for d in (x.strip() for x in device_value.split(",")):
                if d:
                    defs[vid][d] = val
    return defs


def substitute_vars(code, var_defs, device, where) -> str:
    if not var_defs:
        return code
    pat = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def repl(m):
        name = m.group(1)
        if name not in var_defs:
            return m.group(0)
        mapping = var_defs[name]
        val = (mapping.get(device) if device else None) or mapping.get("all")
        if val is None:
            raise ValueError(
                f"{where}: @var '${{{name}}}' has no value for device "
                f"'{device or '<unset>'}' (have: {', '.join(sorted(mapping))})")
        return val

    return pat.sub(repl, code)


def resolve_requires(content: str, docs_root: Path) -> str:
    registry = docs_root / "_includes" / "registry.json"
    if not registry.exists():
        return content
    try:
        deps = json.loads(registry.read_text(encoding="utf-8")).get("dependencies", {})
    except Exception:
        return content

    def repl(m):
        parts = []
        for dep_id in (d.strip() for d in m.group(1).split(",") if d.strip()):
            info = deps.get(dep_id)
            if not info:
                continue
            f = docs_root / "_includes" / info["file"]
            if f.exists():
                parts.append(f.read_text(encoding="utf-8"))
        return "\n".join(parts) if parts else m.group(0)

    return REQUIRE_RE.sub(repl, content)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def strip_hide(code: str) -> str:
    out = []
    for line in code.splitlines():
        if line.rstrip().endswith("#hide"):
            out.append(re.sub(r"\s*#hide\s*$", "", line))
        else:
            out.append(line)
    return "\n".join(out)


def check_python_syntax(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<doc-block>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def run_block(lang, code, timeout, setup, cwd) -> tuple[bool, str]:
    code = strip_hide(code)
    is_win = sys.platform == "win32"
    try:
        if lang == "python":
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(code)
                path = f.name
            if setup:
                shell = "cmd" if is_win else "bash"
                flag = "/c" if is_win else "-c"
                cmd = [shell, flag, f'{setup} && python "{path}"']
            else:
                cmd = [sys.executable, path]
        elif lang in ("bash", "sh", "shell"):
            body = f"{setup}\n{code}" if setup else code
            cmd = ["bash", "-c", body]
        elif lang in ("powershell", "pwsh", "ps1"):
            body = f"{setup}\n{code}" if setup else code
            cmd = ["powershell", "-NoProfile", "-Command", body]
        elif lang in ("bat", "cmd", "batch"):
            body = code if not setup else "\n".join([setup, code])
            with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False, encoding="utf-8") as f:
                f.write(body)
                path = f.name
            cmd = ["cmd", "/c", path]
        else:
            return True, "skipped (unsupported lang)"
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(cwd) if cwd else None)
        return p.returncode == 0, (p.stderr or p.stdout)[-500:]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return False, f"runner error: {e}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=Path("docs"))
    ap.add_argument("--run", action="store_true", help="execute runnable blocks (default-on)")
    ap.add_argument("--syntax-only", action="store_true", help="python syntax checks only")
    ap.add_argument("--device", choices=sorted(DEVICE_TAGS), default=None,
                    help="only run blocks tagged with this device (or untagged)")
    ap.add_argument("--platform", choices=["windows", "linux"], default=None,
                    help="filter @os: blocks (defaults to the host OS)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--output-json", type=Path, default=Path("code-results.json"))
    ap.add_argument("--failed-pages", type=Path, default=Path("failed-pages.txt"))
    args = ap.parse_args()

    platform = args.platform or ("windows" if sys.platform == "win32" else "linux")
    results, failed_pages = [], set()

    # Scan every Markdown file (.mdx pages AND .md example READMEs) - no doc
    # with code blocks is left untested.
    docs_files = sorted(set(args.docs.rglob("*.mdx")) | set(args.docs.rglob("*.md")))
    for mdx in docs_files:
        raw = mdx.read_text(encoding="utf-8")
        content = resolve_requires(raw, args.docs)
        rel = mdx.as_posix()

        os_blocks = find_nested_blocks(content, OS_OPEN_RE, OS_CLOSE_RE)
        device_blocks = find_nested_blocks(content, DEVICE_OPEN_RE, DEVICE_CLOSE_RE)
        setup_defs = extract_setup_definitions(content)
        var_defs = extract_var_definitions(content)

        for i, m in enumerate(FENCE_RE.finditer(content)):
            parsed = parse_fence(m.group(1))
            if not parsed:
                continue
            lang, tags, attrs = parsed
            code = m.group(2)
            pos = m.start()

            if SKIP_TAG in tags:
                results.append(_rec(rel, i, lang, tags, False, "skipped", "notest"))
                continue
            if lang not in RUNNABLE:
                results.append(_rec(rel, i, lang, tags, False, "skipped", "non-runnable lang"))
                continue

            # OS scope filter
            block_os = infer_scope(os_blocks, pos)
            if block_os != "all" and block_os != platform:
                results.append(_rec(rel, i, lang, tags, False, "skipped", f"os!={platform}"))
                continue

            # Device filter: fence tags take precedence, else surrounding @device block
            block_devices = (tags & DEVICE_TAGS)
            if not block_devices:
                scoped = infer_scope(device_blocks, pos)
                if scoped != "all":
                    block_devices = {d.strip() for d in scoped.split(",") if d.strip()}
            if args.device and block_devices and args.device not in block_devices:
                results.append(_rec(rel, i, lang, tags, False, "skipped", f"device!={args.device}"))
                continue

            # Resolve var substitutions and setup
            try:
                code = substitute_vars(code, var_defs, args.device, f"{rel}#block{i}")
            except ValueError as e:
                failed_pages.add(rel)
                results.append(_rec(rel, i, lang, tags, False, "fail", str(e)))
                continue
            setup = resolve_setup(attrs.get("setup"), setup_defs, platform)
            timeout = attrs.get("timeout", args.timeout)
            workdir = (mdx.parent / attrs["workdir"]) if attrs.get("workdir") else mdx.parent
            cont = attrs.get("continue_on_error", False)

            status, detail, ran = "skipped", "", False
            if lang == "python":
                ok, detail = check_python_syntax(code)
                status = "pass" if ok else "fail"
                if not ok and not cont:
                    failed_pages.add(rel)

            if args.run and not args.syntax_only:
                ok, detail = run_block(lang, code, timeout, setup, workdir)
                ran = True
                status = "pass" if ok else "fail"
                if not ok and not cont:
                    failed_pages.add(rel)

            results.append(_rec(rel, i, lang, tags, ran, status, detail))

    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.failed_pages.write_text("\n".join(sorted(failed_pages)), encoding="utf-8")

    n_fail = len(failed_pages)
    n_exec = sum(1 for r in results if r["executed"])
    print(f"Checked {len(results)} blocks across docs "
          f"({n_exec} executed); {n_fail} page(s) with failures.")
    for r in results:
        if r["status"] == "fail":
            print(f"  FAIL {r['page']} block#{r['block']} ({r['lang']}): {r['detail'][:160]}")
    sys.exit(1 if n_fail else 0)


def _rec(page, block, lang, tags, executed, status, detail):
    return {"page": page, "block": block, "lang": lang,
            "tags": sorted(tags), "executed": executed,
            "status": status, "detail": detail}


if __name__ == "__main__":
    main()
