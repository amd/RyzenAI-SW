# Docs-as-Code CI

Self-contained CI for the Mintlify docs in `../docs`. Three workflows in
`.github/workflows/`:

| Workflow | What it does | Runner |
|----------|--------------|--------|
| `mintlify-checks.yml` | `mint validate`, `mint broken-links`, `mint a11y`, Vale prose, cspell | cloud (ubuntu) |
| `test-code-samples.yml` | python syntax checks of all blocks; executes blocks tagged `test` on hardware | cloud + self-hosted Strix / Strix Halo (Windows) |
| `notify-owner.yml` | on failure, resolves the page owner and emails them the failure | cloud (ubuntu) |

## Scripts

- `resolve_owner.py` - read the hidden `{/* owner: id | email */}` header from a page.
- `generate_codeowners.py` - regenerate `CODEOWNERS` from page headers + the example-owner map (CI fails if the committed file is stale).
- `extract_code_blocks.py` - parse/validate fenced code blocks (opt-in execution via the `test` tag; skip with `notest`).
- `notify_owner.py` - compose the owner notification; dry-run preview when SMTP is not configured; `--send`/`--to` for local test emails.
- `record_run.py` - append a run record (what ran, when, pass/fail, per-page results + owner) to `ci-history.json`.

## CI tracking dashboard

Every workflow ends with a `record` job that calls `record_run.py` to append the
run to `ci-history.json` (committed on `main`, uploaded as an artifact on PRs).
Two ways to view it:

- **In-repo dashboard:** open `.github/scripts/dashboard/index.html` (or host the
  `.github/scripts/dashboard/` folder on GitHub Pages). It reads `ci-history.json` and
  shows summary cards, the run log (time, workflow, trigger, scripts, status),
  and a per-page table with owners.
- **Cursor canvas:** `rai-docs-ci.canvas.tsx` (in the workspace `canvases/`
  folder) renders the same data beside the chat.

`ci-history.json` is the single data store for both.

## Code-block conventions (test-by-default / opt-out)

````
```python          -> executed by default (also syntax-checked)
```bash             -> executed by default
```python npu       -> executed; device-scoped (cpu | gpu | npu)
```bash notest      -> skipped entirely (use for non-runnable snippets)
```text / ```json   -> ignored (non-runnable language)
````

Every runnable block runs unless you add `notest`. Device tags scope a block to
an accelerator: with `--device npu`, only `npu`-tagged and untagged blocks run.

### Advanced (optional) - ported from AMD Playbooks, none used yet

These are available for authors but no page uses them today. Tags use MDX-valid
`{/* ... */}` comments (not Playbooks' `<!-- ... -->`). See the
`extract_code_blocks.py` docstring for full syntax.

- Per-block attrs in the fence: `timeout=600`, `workdir=examples`,
  `continue_on_error=true`, `setup=<named-setup>`.
- `#hide` on a line - executed, but meant to be hidden on the rendered site.
- Scope blocks: `{/* @os:windows */}…{/* @os:end */}`,
  `{/* @device:npu */}…{/* @device:end */}`.
- Reusable `{/* @setup:id=… command="…" */}` (OS-scoped) and device-aware
  `{/* @var:id=… device=npu value="…" */}` referenced in code as `${name}`.
- `{/* @require:<id> */}` to inline a shared include from `docs/_includes/`.

## Secrets / variables (optional, enable real sends)

- `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` (secrets) - enable real emails. Without them, the notifier prints a full dry-run preview (used for the demo).
- `NOTIFY_TEST_RECIPIENT` (variable) - redirect all notifications to one address while testing.

## Demo

Trigger `Notify Owner On Failure` via "Run workflow" with a page (default `docs/getting-started/inst.mdx`) to watch owner resolution + notification end to end. Locally:

```bash
python .github/scripts/notify_owner.py --file docs/getting-started/inst.mdx
```

## Hardware

`test-hardware` targets `[self-hosted, Windows, stxp]` (Strix Point) and
`[self-hosted, Windows, halo]` (Strix Halo) - the real AMD Playbooks device
labels. See `RUNNERS.md` for where those machines live and how to get the docs
repo added to the runner group that hosts them.
