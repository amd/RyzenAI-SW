# Docs-as-Code CI

Everything that powers the documentation CI for the Mintlify site in `../../docs`.
This is the **single** doc for the CI - scripts, conventions, runners, reporting,
and deployment. (The only other file you should ever read here is the generated
`CODE_TEST_REPORT.md`.)

## Table of contents

1. [Pipeline at a glance](#pipeline-at-a-glance)
2. [Scripts](#scripts)
3. [Code-block testing model (test-by-default)](#code-block-testing-model-test-by-default)
4. [Languages: what runs, what doesn't (incl. C++)](#languages-what-runs-what-doesnt-incl-c)
5. [Authoring conventions](#authoring-conventions)
6. [Runners (hardware execution)](#runners-hardware-execution)
7. [Per-page report & dashboard](#per-page-report--dashboard)
8. [Failure routing (CODEOWNERS + notify)](#failure-routing-codeowners--notify)
9. [End-to-end walkthrough](#end-to-end-walkthrough)
10. [Deploying the site](#deploying-the-site)
11. [Run it locally](#run-it-locally)

---

## Pipeline at a glance

| Workflow (`.github/workflows/`) | What it does | Where |
|---|---|---|
| `mintlify-checks.yml` | `mint validate`, internal `broken-links` (blocking); external links + anchors + redirects (non-blocking); Vale prose; cspell | cloud (ubuntu) |
| `link-check.yml` | full external link check, opens a tracking issue on failure | cloud, weekly + manual |
| `test-code-samples.yml` | python syntax check (cloud) + **execute every runnable block** on AMD hardware | cloud + self-hosted |
| `codeowners.yml` | regenerate `CODEOWNERS`, fail if the committed file is stale | cloud |
| `notify-owner.yml` | on failure, resolve the page owner and open an issue that @mentions them | cloud |
| `update-model-list.yml` | refresh the model tables from Hugging Face | cloud, scheduled |

Every code/link run ends by appending a record to `ci-history.json` (the data
store behind the dashboard).

## Scripts

| Script | Purpose |
|---|---|
| `extract_code_blocks.py` | Parse every fenced block in `docs/**` (`.mdx` **and** `.md`) and **execute** each runnable one. `--syntax-only` for cloud; `--run` for hardware. |
| `report.py` | Turn a run's JSON into `CODE_TEST_REPORT.md` (per-page + per-block dashboard) and inject the summary table into the CI dashboard page. |
| `generate_codeowners.py` | Regenerate `docs/CODEOWNERS` from each page's `{/* owner: id */}` header (no external map). |
| `resolve_owner.py` | Read the owner id from a single page. |
| `check_owners.py` | Verify every page has an owner header. |
| `notify_owner.py` | Compose the GitHub-native failure notification (issue @mentioning owners). |
| `record_run.py` | Append a run (status + per-page results + owner) to `ci-history.json`. |
| `fetch_models.py` | Regenerate the model tables on the Vision/LLMs/Audio pages. |
| `gen_cards.py` | Regenerate the "bubble" card lists on each category index page. |

## Code-block testing model (test-by-default)

**Every fenced block in a runnable language is executed on every run.** There is
no opt-in. The single opt-out is `notest`.

````
```python          -> EXECUTED (always; also python-syntax-checked)
```powershell       -> EXECUTED (Windows-native)
```bash             -> EXECUTED via WSL on Windows
```cpp / ```c       -> COMPILED with g++/gcc (run if it has main())
```json / ```yaml   -> LINTED for format validity
```python notest    -> the only opt-out: skipped entirely
````

Optional **authoring** tags (they do not make testing optional):

- `npu` / `gpu` / `cpu` - device scope (`--device npu` runs npu-tagged + untagged).
- `timeout=600`, `workdir=examples`, `continue_on_error=true`, `setup=<id>`.
- Page directives in MDX comments: `{/* @os:windows */}…{/* @os:end */}`,
  `{/* @device:npu */}…`, `{/* @setup:id=… command="…" */}`,
  `{/* @var:id=… device=npu value="…" */}`, `{/* @require:<id> */}`.

A page's blocks run **in order in one sandbox dir**, so a `git clone` / `cd` in
an early block persists for later blocks, and nothing pollutes the docs tree.

## Languages: what runs, what doesn't (incl. C++)

| Fence | Tested how | Where it runs |
|---|---|---|
| `python` | syntax-compiled, then **run** | Ryzen AI conda env (NPU/GPU/CPU providers visible) |
| `powershell` / `pwsh` / `ps1` | **run** | Windows-native (`powershell -NoProfile -Command`) |
| `cmd` / `bat` / `batch` | **run** | Windows-native (`cmd /c`) |
| `bash` / `sh` / `shell` | **run** | **WSL** on Windows (`wsl bash -lc`); native `bash` on Linux |
| `cpp` / `c` | **compiled** with `g++`/`gcc` (and run if it defines `main()`; otherwise `-fsyntax-only`) | WSL on Windows; native on Linux |
| `json` / `yaml` / `toml` / `cmake` / `text` | **linted** for format validity (`json.loads`, `yaml.safe_load`, `tomllib`, paren-balance, plain-text) | in-process (no shell) |
| `mdx` / `ini` / other | skipped | n/a (not runnable or lintable) |

**Linux vs Windows routing:** `bash`, C/C++, and anything inside an `@os:linux`
scope are Linux work, so on the Windows runner they execute through **WSL**
(Ubuntu). Windows blocks (`powershell`, `cmd`) run natively. Pass `--no-wsl` to
skip Linux blocks instead of running them through WSL. A dedicated Ubuntu runner
can replace WSL later with no doc changes (the routing is automatic).

## Authoring conventions

1. **Owner header (required)** - first line after frontmatter:
   `{/* owner: <github-id> */}`. Drives CODEOWNERS + failure routing. Default
   owner: `@dwithchenna`.
2. **Language tabs** - when the same step exists in multiple languages, show
   **Python first**, then **C++**, using Mintlify `<Tabs><Tab title="Python">…`.
3. **2-level page paths** - `folder/page.mdx` (e.g. `llms/hybrid_oga.mdx`). The
   link checker and Mintlify only resolve 2-level page paths; do deeper grouping
   in `docs.json` nav, not on disk. (Top-level standalone pages like
   `index.mdx` and `installation.mdx` are fine.)
4. **Icons** - only on top-level categories (group `icon` in `docs.json`, or a
   frontmatter `icon:` on a top-level page). Never on second-level pages.

## Runners (hardware execution)

The `test-hardware` job runs on a **self-hosted runner**. Devices are chosen by
the `DOCS_CI_DEVICES` repo variable (a JSON array) so you never edit the
workflow to add hardware:

| Variable | Default | Meaning |
|---|---|---|
| `DOCS_CI_DEVICES` | `["halo"]` | runner device labels to target (e.g. `["halo","stxp","krk"]`) |
| `DOCS_CI_OS` | `Windows` | OS label in the runner triple |
| `RYZEN_AI_ENV` | `ryzen-ai-1.7.1` | conda env on the runner with the NPU/GPU/CPU providers |

A job targets `runs-on: [self-hosted, <DOCS_CI_OS>, <device>]`. To add a machine:
label it `self-hosted`, the OS, and a device tag, then add that tag to
`DOCS_CI_DEVICES`.

**Today:** a single local **Strix Halo** box (`AMD Ryzen AI Max+`, NPU present)
is registered to this repo with the label `halo`, so `DOCS_CI_DEVICES=["halo"]`
runs the whole suite end-to-end on real hardware.

**Shared AMD pool (future):** the AMD Playbooks lab machines (`xsj-aimlab-halo-*`,
`xsj-aimlab-stxp-*`, `…-krk-*`) use the same `[self-hosted, Windows, <device>]`
label scheme. They are registered to an AMD **org runner group**, so to use them
the docs repo must live under the `amd` org and be added to that group (a fork
under a personal account gets `403` and the job queues forever). Then set
`DOCS_CI_DEVICES=["halo","stxp","krk"]`.

Register the local box (stopgap): repo -> Settings -> Actions -> Runners -> New
self-hosted runner (Windows); add the label `halo`; run as a service
(`./svc.cmd install && ./svc.cmd start`).

## Per-page report & dashboard

- **`report.py`** reads a run's JSON (`--output-json` from
  `extract_code_blocks.py`) and writes **`CODE_TEST_REPORT.md`** - a summary
  table (one row per page: blocks / pass / fail / skip / owner) plus a per-page
  detail table (one row per block: `#`, lang, result, short detail). It covers
  **every** `.mdx` and `.md`, including pages with no code (`no code`).
- It also injects the summary table into `docs/reference/ci-dashboard.mdx`
  between `{/* RESULTS_START */}` / `{/* RESULTS_END */}`, so the published CI
  dashboard reflects all pages.
- **`ci-history.json`** is the append-only run log (status + per-page result +
  owner) that `record_run.py` writes; the in-repo dashboard
  (`.github/scripts/dashboard/index.html`) and the Cursor canvas read it.

## Failure routing (CODEOWNERS + notify)

- `generate_codeowners.py` rebuilds `docs/CODEOWNERS` from page headers: a
  catch-all default, infra rules, a per-folder default (the folder's dominant
  owner), then a per-page rule for every page. `codeowners.yml` fails a PR if
  the committed file is stale.
- On a failing run, `notify_owner.py` resolves the owner from the page header
  and opens an issue that **@mentions** them - GitHub emails them through its own
  system, so no individual email addresses are stored anywhere.
- **Plus a full report by email to the shared support DL.** `notify-owner.yml`
  also emails the report to `NOTIFY_EMAIL` (repo variable, default
  `dl.ryzenai.support@amd.com`). A distribution list is safe to keep public - it
  is not a person's address. This email step runs only when the SMTP relay
  secrets are configured: `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`,
  `SMTP_PASSWORD`. Without them, the GitHub issue is still opened; the email is
  simply skipped.

## End-to-end walkthrough

Take `docs/vision/super_resolution.mdx` (owner `@bconsolvo`), which has a
runnable `python` PSNR block.

1. A PR edits `docs/**` -> `Mintlify Docs Checks` and `Test Code Samples` run.
2. `mint validate` + `broken-links` confirm the build and links.
3. `extract_code_blocks.py --syntax-only` compiles every python block (cloud).
4. `extract_code_blocks.py --run` executes the PSNR block on the Strix Halo
   runner; it prints `PSNR @ MSE=100 -> 28.13 dB` and passes.
5. If it regressed, the page is written to `failed-pages.txt`, `notify-owner`
   resolves `@bconsolvo` from the header and opens an issue mentioning them.
6. `generate_codeowners.py` keeps `CODEOWNERS` in sync from the same header, so
   review assignment and the notifier always agree.

## Deploying the site

The site is hosted by **Mintlify** from the `/docs` subfolder; the GitHub repo
can stay private while the site is public.

1. mintlify.com -> connect the repo via the Mintlify GitHub App (works on
   private repos; scope it to this repo).
2. Dashboard -> Git Settings -> enable **monorepo**, set the docs path to
   `/docs` (otherwise it looks for `docs.json` at the root and fails).
3. Push/merge to `main` -> auto rebuild + deploy. Every PR gets a preview build.
4. The public URL (e.g. `https://ryzen-ai-xxxx.mintlify.app`) shows on the
   dashboard Overview.

Notes: GitHub/Discord sidebar links, icons, theme, and the AI "Ask/Copy" menu
live in `docs/docs.json`. The "View as Markdown" / Ask-AI routes are produced by
Mintlify's hosted build and 404 under local `mint dev` - expected.

## Run it locally

```bash
# Cloud-equivalent syntax check (fast, no hardware)
python .github/scripts/extract_code_blocks.py --syntax-only --docs docs

# Full hardware run (inside the Ryzen AI conda env), then build the report
conda run -n ryzen-ai-1.7.1 python .github/scripts/extract_code_blocks.py \
    --run --docs docs --output-json report_run.json --failed-pages report_failed.txt
python .github/scripts/report.py --results report_run.json --docs docs \
    --out .github/scripts/CODE_TEST_REPORT.md --dashboard docs/reference/ci-dashboard.mdx

# Regenerate CODEOWNERS / model tables / index cards
python .github/scripts/generate_codeowners.py
python .github/scripts/fetch_models.py
python .github/scripts/gen_cards.py

# Preview the site
cd docs && npx mint dev      # http://localhost:3000
```
