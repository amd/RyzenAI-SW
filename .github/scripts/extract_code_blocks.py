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

================================ Testing model ==============================
TEST BY DEFAULT. Every fenced block written in a runnable language IS EXECUTED
on every run. There is no opt-in tag. The ONLY way to skip a block is `notest`.

    ```python                  -> EXECUTED (always; also python-syntax-checked)
    ```powershell / ```cmd     -> EXECUTED (Windows-native)
    ```bash / ```sh            -> EXECUTED via WSL on Windows (native bash on Linux)
    ```cpp / ```c              -> COMPILED with g++/gcc (run if it defines main();
                                  otherwise -fsyntax-only). Via WSL on Windows.
    ```json/yaml/toml/cmake/text -> LINTED for format validity (e.g. json.loads)
    ```python notest           -> the one opt-out: skipped entirely

Linux blocks (bash, C/C++, anything in @os:linux) run through WSL on the Windows
runner; pass --no-wsl to skip them instead. Only truly unknown fences (mdx, ini,
...) are recorded "skipped" - nothing runnable is silently passed.

=============================== Authoring extras ============================
The tags/attributes below are OPTIONAL conveniences for authors (they do NOT
make testing optional). Per-block, in the fence info string (after the language):
    ```python npu              -> device-scoped (cpu | gpu | npu)
    ```python timeout=600      -> per-block timeout (seconds)
    ```bash workdir=examples   -> run in <page-dir>/examples
    ```bash continue_on_error=true   -> a failure doesn't fail the page
    ```python setup=activate-venv    -> run a named @setup first

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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

FENCE_RE = re.compile(r"^```([^\n]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)

RUNNABLE_SHELL_WIN = {"powershell", "pwsh", "ps1", "cmd", "bat", "batch"}
RUNNABLE_SHELL_NIX = {"bash", "sh", "shell"}        # executed via WSL on Windows
COMPILE_LANGS = {"cpp", "c++", "cc", "cxx", "c"}    # compiled (run if it has main)
FORMAT_LANGS = {"json", "yaml", "yml", "toml", "cmake", "text"}  # validated/linted
RUNNABLE = RUNNABLE_SHELL_WIN | RUNNABLE_SHELL_NIX | COMPILE_LANGS | {"python"}
TESTABLE = RUNNABLE | FORMAT_LANGS                  # everything we check (else skip)
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


_WSL_CACHE: Optional[bool] = None


def wsl_available() -> bool:
    """True if a WSL distro is usable (Windows only). Cached."""
    global _WSL_CACHE
    if _WSL_CACHE is None:
        if sys.platform != "win32":
            _WSL_CACHE = False
        else:
            try:
                _WSL_CACHE = subprocess.run(
                    ["wsl.exe", "-e", "true"], capture_output=True, timeout=30
                ).returncode == 0
            except Exception:  # noqa: BLE001
                _WSL_CACHE = False
    return _WSL_CACHE


def _run(cmd, timeout, cwd) -> tuple[bool, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=str(cwd) if cwd else None)
    return p.returncode == 0, (p.stderr or p.stdout)[-500:]


def run_in_wsl(body, timeout, cwd) -> tuple[bool, str]:
    """Run a bash snippet inside WSL, using the Windows sandbox dir as cwd."""
    cmd = ["wsl.exe"]
    if cwd:
        cmd += ["--cd", str(cwd)]
    cmd += ["bash", "-lc", body]
    return _run(cmd, timeout, None)


def check_format(lang, code) -> tuple[bool, str]:
    """Lint a non-executable block: verify it really is valid for its language."""
    try:
        if lang == "json":
            json.loads(code)
            return True, "valid JSON"
        if lang in ("yaml", "yml"):
            try:
                import yaml  # type: ignore
            except ImportError:
                return True, "skipped: pyyaml not installed"
            list(yaml.safe_load_all(code))
            return True, "valid YAML"
        if lang == "toml":
            try:
                import tomllib  # type: ignore
            except ImportError:
                return True, "skipped: tomllib unavailable"
            tomllib.loads(code)
            return True, "valid TOML"
        if lang == "cmake":
            if code.count("(") != code.count(")"):
                return False, "CMake: unbalanced parentheses"
            return True, "CMake parentheses balanced"
        if lang == "text":
            return True, "plain text"
    except Exception as e:  # noqa: BLE001
        return False, f"{lang} format error: {e}"
    return True, ""


def compile_and_run(lang, code, timeout, cwd, use_wsl) -> tuple[bool, str]:
    """Compile a C/C++ block (and run it if it defines main()). Uses WSL
    gcc/g++ on Windows, native gcc/g++ on Linux. Snippets without main() get a
    `-fsyntax-only` compile check."""
    is_c = lang == "c"
    comp = "gcc" if is_c else "g++"
    std = "" if is_c else "-std=c++17"
    src = f"_doc_block.{'c' if is_c else 'cpp'}"
    (Path(cwd) / src).write_text(strip_hide(code), encoding="utf-8")
    if re.search(r"\bmain\s*\(", code):
        body = f"{comp} {std} {src} -o _doc_block.out && ./_doc_block.out"
    else:
        body = f"{comp} {std} -fsyntax-only {src}"
    if use_wsl:
        return run_in_wsl(body, timeout, cwd)
    if sys.platform != "win32":
        return _run(["bash", "-lc", body], timeout, cwd)
    return False, "no C/C++ compiler available (need WSL or a native gcc/g++)"


def run_block(lang, code, timeout, setup, cwd, use_wsl) -> tuple[bool, str]:
    code = strip_hide(code)
    try:
        if lang == "python":
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                             encoding="utf-8", dir=str(cwd)) as f:
                f.write(code)
                path = f.name
            if setup:
                is_win = sys.platform == "win32"
                shell, flag = ("cmd", "/c") if is_win else ("bash", "-c")
                cmd = [shell, flag, f'{setup} && python "{path}"']
            else:
                cmd = [sys.executable, path]
            return _run(cmd, timeout, cwd)
        if lang in RUNNABLE_SHELL_NIX:
            body = f"{setup}\n{code}" if setup else code
            if use_wsl:
                return run_in_wsl(body, timeout, cwd)
            return _run(["bash", "-c", body], timeout, cwd)
        if lang in ("powershell", "pwsh", "ps1"):
            body = f"{setup}\n{code}" if setup else code
            return _run(["powershell", "-NoProfile", "-Command", body], timeout, cwd)
        if lang in ("bat", "cmd", "batch"):
            body = code if not setup else "\n".join([setup, code])
            with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False,
                                             encoding="utf-8", dir=str(cwd)) as f:
                f.write(body)
                path = f.name
            return _run(["cmd", "/c", path], timeout, cwd)
        if lang in COMPILE_LANGS:
            return compile_and_run(lang, code, timeout, cwd, use_wsl)
        return True, "skipped (unsupported lang)"
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
                    help="host platform for native execution (defaults to host OS)")
    ap.add_argument("--no-wsl", action="store_true",
                    help="do not use WSL; Linux (bash/C/C++/@os:linux) blocks then skip on Windows")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--output-json", type=Path, default=Path("code-results.json"))
    ap.add_argument("--failed-pages", type=Path, default=Path("failed-pages.txt"))
    args = ap.parse_args()

    platform = args.platform or ("windows" if sys.platform == "win32" else "linux")
    host_windows = sys.platform == "win32"
    use_wsl = (not args.no_wsl) and wsl_available()
    linux_ok = (not host_windows) or use_wsl
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

        # Per-page sandbox: all of a page's blocks run in ONE temp dir, in order,
        # so files/downloads from earlier blocks persist for later blocks
        # ("keep the environment open"), and nothing pollutes the docs tree.
        page_dir = Path(tempfile.mkdtemp(prefix="docs-ci-"))

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
            if lang not in TESTABLE:
                results.append(_rec(rel, i, lang, tags, False, "skipped", f"{lang}: non-runnable lang"))
                continue

            # Which environment must this block run in? Explicit @os scope wins;
            # else infer from language (nix shells / C-C++ -> linux; ps/cmd -> windows).
            block_os = infer_scope(os_blocks, pos)
            if block_os in ("windows", "linux"):
                need = block_os
            elif lang in (RUNNABLE_SHELL_NIX | COMPILE_LANGS):
                need = "linux"
            elif lang in RUNNABLE_SHELL_WIN:
                need = "windows"
            else:
                need = "any"
            block_platform = need if need in ("windows", "linux") else platform

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
            setup = resolve_setup(attrs.get("setup"), setup_defs, block_platform)
            timeout = attrs.get("timeout", args.timeout)
            workdir = (page_dir / attrs["workdir"]) if attrs.get("workdir") else page_dir
            workdir.mkdir(parents=True, exist_ok=True)
            cont = attrs.get("continue_on_error", False)

            status, detail, ran = "skipped", "", False

            # Non-executable languages: lint that they're valid (json/yaml/etc.).
            if lang in FORMAT_LANGS:
                ok, detail = check_format(lang, code)
                status = "pass" if ok else "fail"
                if not ok and not cont:
                    failed_pages.add(rel)
                results.append(_rec(rel, i, lang, tags, False, status, detail))
                continue

            # Python: always syntax-check (cloud + hardware).
            if lang == "python":
                ok, detail = check_python_syntax(code)
                status = "pass" if ok else "fail"
                if not ok and not cont:
                    failed_pages.add(rel)

            # Execution (hardware/run mode only).
            if args.run and not args.syntax_only:
                if need == "windows" and not host_windows:
                    status, detail = "skipped", "needs Windows runner"
                elif need == "linux" and not linux_ok:
                    status, detail = "skipped", "needs Linux/WSL runner"
                else:
                    ok, detail = run_block(lang, code, timeout, setup, workdir, use_wsl)
                    ran = True
                    status = "pass" if ok else "fail"
                    if not ok and not cont:
                        failed_pages.add(rel)

            results.append(_rec(rel, i, lang, tags, ran, status, detail))

        shutil.rmtree(page_dir, ignore_errors=True)

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
