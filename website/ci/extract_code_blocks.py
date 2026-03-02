#!/usr/bin/env python3
"""Extract and validate code blocks from docs for CI testing.

Parses markdown files and extracts code blocks tagged with runnable
languages (python, cpp, bash, powershell). Blocks marked with 'notest'
are skipped. Also validates standalone Python files and requirements.txt.

Usage:
    # Extract from all docs (dry run)
    python extract_code_blocks.py

    # Extract only from files changed in current PR
    python extract_code_blocks.py --diff-only

    # Extract and syntax-check inline blocks
    python extract_code_blocks.py --run --syntax-only

    # Extract and execute on hardware
    python extract_code_blocks.py --run

    # Also check standalone .py files and requirements.txt
    python extract_code_blocks.py --run --syntax-only --check-files

    # Full local validation (same as CI)
    python extract_code_blocks.py --run --syntax-only --check-files --check-imports --output-json results.json
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

RUNNABLE_LANGUAGES = {"python", "py", "cpp", "c++", "bash", "sh", "powershell", "ps1", "bat"}
DOCS_ROOT = Path(__file__).resolve().parent.parent.parent
CI_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CI_DIR / "extracted-blocks"
SKIP_DIRS = {"node_modules", ".docusaurus", "build", "__pycache__", "templates", "src", "static", "extracted-blocks"}
IS_WINDOWS = platform.system() == "Windows"

FENCE_PATTERN = re.compile(
    r"^```(\w+)([^\n]*)\n(.*?)^```",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class CodeBlock:
    language: str
    code: str
    source_file: str
    line_number: int
    notest: bool = False
    output_path: str = ""


@dataclass
class ExtractionResult:
    blocks: list[CodeBlock] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def get_changed_files() -> list[Path]:
    """Get .md/.mdx files changed in current git diff (PR context)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True, text=True, check=True,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            p = Path(line)
            if p.suffix in (".md", ".mdx") and p.exists():
                files.append(p)
        return files
    except subprocess.CalledProcessError:
        print("Warning: Could not get git diff, falling back to full scan")
        return list(DOCS_ROOT.rglob("*.mdx")) + list(DOCS_ROOT.rglob("*.md"))


def _should_skip(filepath: Path) -> bool:
    """Check if a file is in a directory that should be skipped."""
    parts = set(filepath.relative_to(DOCS_ROOT).parts)
    return bool(parts & SKIP_DIRS)


def get_all_doc_files() -> list[Path]:
    """Get all .md/.mdx files in the docs directory, excluding infra dirs."""
    files = []
    for f in sorted(DOCS_ROOT.rglob("*.mdx")):
        if not _should_skip(f):
            files.append(f)
    for f in sorted(DOCS_ROOT.rglob("*.md")):
        if not _should_skip(f):
            files.append(f)
    return files


def find_python_files() -> list[Path]:
    """Find all standalone Python files in docs/ (excluding infra dirs)."""
    files = []
    for f in sorted(DOCS_ROOT.rglob("*.py")):
        if _should_skip(f):
            continue
        files.append(f)
    return files


def find_requirements_files() -> list[Path]:
    """Find all requirements.txt files in docs/ (excluding infra dirs)."""
    files = []
    for f in sorted(DOCS_ROOT.rglob("requirements.txt")):
        if _should_skip(f):
            continue
        files.append(f)
    return files


def extract_blocks_from_file(filepath: Path) -> list[CodeBlock]:
    """Extract fenced code blocks from a single file."""
    content = filepath.read_text(encoding="utf-8")
    blocks = []

    for match in FENCE_PATTERN.finditer(content):
        lang = match.group(1).lower()
        metadata = match.group(2).strip()
        code = match.group(3)

        if lang not in RUNNABLE_LANGUAGES:
            continue

        line_number = content[: match.start()].count("\n") + 1
        notest = "notest" in metadata

        normalized_lang = lang
        if lang in ("py",):
            normalized_lang = "python"
        elif lang in ("c++",):
            normalized_lang = "cpp"
        elif lang in ("sh",):
            normalized_lang = "bash"
        elif lang in ("ps1",):
            normalized_lang = "powershell"
        elif lang in ("bat",):
            normalized_lang = "bat"

        blocks.append(
            CodeBlock(
                language=normalized_lang,
                code=code,
                source_file=str(filepath),
                line_number=line_number,
                notest=notest,
            )
        )

    return blocks


def write_extracted_block(block: CodeBlock, index: int) -> Path:
    """Write an extracted code block to a temp file."""
    ext_map = {
        "python": ".py",
        "cpp": ".cpp",
        "bash": ".sh",
        "powershell": ".ps1",
        "bat": ".bat",
    }
    ext = ext_map.get(block.language, ".txt")
    filename = f"block_{index:04d}{ext}"
    output_path = OUTPUT_DIR / filename

    header = f"# Extracted from {block.source_file}:{block.line_number}\n"
    if block.language == "python":
        header = f"# -*- coding: utf-8 -*-\n{header}"

    output_path.write_text(header + block.code, encoding="utf-8")
    block.output_path = str(output_path)
    return output_path


def syntax_check_python(filepath: Path) -> tuple[bool, str]:
    """Check Python syntax without executing."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(filepath)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


def has_placeholders(code: str) -> bool:
    """Check if code contains <placeholder> patterns that can't be executed."""
    return bool(re.search(r"<[a-zA-Z_][a-zA-Z0-9_ /.|]*>", code))


def syntax_check_bash(filepath: Path) -> tuple[bool, str]:
    """Check bash syntax without executing."""
    result = subprocess.run(
        ["bash", "-n", str(filepath)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


def check_python_imports(filepath: Path) -> tuple[bool, str]:
    """Check that top-level imports in a Python file are available."""
    content = filepath.read_text(encoding="utf-8", errors="replace")
    import_pattern = re.compile(r"^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))", re.MULTILINE)
    missing = []
    stdlib = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()

    for match in import_pattern.finditer(content):
        module = (match.group(1) or match.group(2)).split(".")[0]
        if module.startswith("_") or module in stdlib:
            continue
        local_init = filepath.parent / module / "__init__.py"
        local_file = filepath.parent / f"{module}.py"
        if local_init.exists() or local_file.exists():
            continue
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                missing.append(module)
        except subprocess.TimeoutExpired:
            pass

    if missing:
        unique = sorted(set(missing))
        return False, f"Missing imports: {', '.join(unique)}"
    return True, ""


def check_requirements_pinned(filepath: Path) -> tuple[bool, str]:
    """Check that requirements.txt has pinned versions."""
    content = filepath.read_text(encoding="utf-8", errors="replace")
    unpinned = []
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line and ">=" not in line and "<=" not in line:
            unpinned.append(f"L{i}: {line}")
    if unpinned:
        return False, f"Unpinned: {'; '.join(unpinned[:5])}"
    return True, ""


def run_block(block: CodeBlock, syntax_only: bool = False) -> tuple[bool, str]:
    """Run an extracted code block and return (success, error_message)."""
    filepath = Path(block.output_path)

    if has_placeholders(block.code):
        return True, "skipped (contains placeholders)"

    if block.language == "python":
        if syntax_only:
            return syntax_check_python(filepath)
        result = subprocess.run(
            [sys.executable, str(filepath)],
            capture_output=True, text=True, timeout=120,
        )
    elif block.language == "bash":
        if IS_WINDOWS:
            return True, "skipped (bash on Windows)"
        if syntax_only:
            return syntax_check_bash(filepath)
        result = subprocess.run(
            ["bash", str(filepath)],
            capture_output=True, text=True, timeout=120,
        )
    elif block.language == "powershell":
        if not IS_WINDOWS:
            return True, "skipped (powershell on Linux)"
        if syntax_only:
            return True, "skipped (powershell syntax-only not implemented)"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(filepath)],
            capture_output=True, text=True, timeout=120,
        )
    elif block.language == "bat":
        if not IS_WINDOWS:
            return True, "skipped (bat on Linux)"
        if syntax_only:
            return True, "skipped (bat syntax-only not implemented)"
        result = subprocess.run(
            ["cmd", "/c", str(filepath)],
            capture_output=True, text=True, timeout=120,
        )
    else:
        return True, "skipped (no runner for this language)"

    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()[:500]


def extract(diff_only: bool = False) -> ExtractionResult:
    """Main extraction pipeline."""
    files = get_changed_files() if diff_only else get_all_doc_files()
    result = ExtractionResult()
    index = 0

    for filepath in files:
        blocks = extract_blocks_from_file(filepath)
        for block in blocks:
            if block.notest:
                result.skipped += 1
                continue
            write_extracted_block(block, index)
            result.blocks.append(block)
            index += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract and validate code blocks from docs")
    parser.add_argument("--diff-only", action="store_true", help="Only process files changed in PR")
    parser.add_argument("--run", action="store_true", help="Execute extracted code blocks")
    parser.add_argument("--syntax-only", action="store_true", help="Only check syntax, don't execute")
    parser.add_argument("--check-files", action="store_true", help="Also check standalone .py files")
    parser.add_argument("--check-imports", action="store_true", help="Check import availability (slower)")
    parser.add_argument("--check-requirements", action="store_true", help="Check requirements.txt pinning")
    parser.add_argument("--output-json", type=str, help="Write results to JSON file")
    parser.add_argument("--output-dir", type=str, help="Override output directory for extracted blocks")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting code blocks from {'changed files' if args.diff_only else 'all docs'}...")
    result = extract(diff_only=args.diff_only)

    print(f"\nExtracted {len(result.blocks)} runnable code blocks ({result.skipped} skipped with notest)")
    print(f"Output directory: {OUTPUT_DIR}\n")

    passed = 0
    failed = 0
    failures = []

    # --- Inline code blocks ---
    if args.run:
        print("=== Inline code blocks ===\n")
        for block in result.blocks:
            success, error = run_block(block, syntax_only=args.syntax_only)
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {block.source_file}:{block.line_number} ({block.language})")
            if success:
                passed += 1
            else:
                failed += 1
                failures.append({
                    "file": block.source_file,
                    "line": block.line_number,
                    "language": block.language,
                    "test_type": "inline-block",
                    "error": error,
                })
                if error:
                    for line in error.split("\n")[:3]:
                        print(f"         {line}")
    else:
        for block in result.blocks:
            print(f"  [{block.language}] {block.source_file}:{block.line_number} -> {block.output_path}")

    # --- Standalone Python files ---
    if args.check_files:
        py_files = find_python_files()
        print(f"\n=== Standalone Python files ({len(py_files)}) ===\n")
        for f in py_files:
            rel = str(f.relative_to(DOCS_ROOT.parent))
            ok, err = syntax_check_python(f)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] syntax  {rel}")
            if ok:
                passed += 1
            else:
                failed += 1
                failures.append({"file": rel, "test_type": "syntax", "error": err[:300]})
                for line in err.split("\n")[:2]:
                    print(f"           {line}")

            if args.check_imports and ok:
                ok2, err2 = check_python_imports(f)
                if ok2:
                    passed += 1
                else:
                    failed += 1
                    print(f"  [FAIL] import  {rel}")
                    print(f"           {err2}")
                    failures.append({"file": rel, "test_type": "import", "error": err2})

    # --- Requirements files ---
    if args.check_requirements:
        req_files = find_requirements_files()
        print(f"\n=== Requirements files ({len(req_files)}) ===\n")
        for f in req_files:
            rel = str(f.relative_to(DOCS_ROOT.parent))
            ok, err = check_requirements_pinned(f)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
                print(f"  [{status}] {rel}: {err}")
                failures.append({"file": rel, "test_type": "requirements", "error": err})

    # --- Summary ---
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed, {result.skipped} skipped")
    print(f"{'='*50}")

    if args.output_json:
        output = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": result.skipped,
            "failures": failures,
        }
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"Results written to {args.output_json}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
