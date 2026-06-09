# Hooking the docs CI up to the AMD Playbooks hardware runners

How the `test-hardware` job in `test-code-samples.yml` runs on real Ryzen AI
machines instead of this local box. This is the result of inspecting the
public `amd/playbooks` repo's Actions setup (the source of truth for the runner
pool).

## Where the machines actually live

The Playbooks runners are physical machines in AMD's **San Jose AI/ML lab**
(the `xsj-aimlab-*` naming = Xilinx San Jose AIML lab), plus a few VMs. From
`amd/playbooks/.github/workflows/runner-heartbeat.yml`, the current pool is:

| Runner name | Silicon | OS |
|-------------|---------|-----|
| `xsj-aimlab-halo-0`, `-1`, `-02`, `-03` | Strix Halo (Ryzen AI Max) | Windows/Linux |
| `xsj-aimlab-stxp-01`, `-02`, `-03`, `-05` | Strix Point | Windows/Linux |
| `xsj-aimlab-krk-01`..`-04` | Krackan Point | Windows/Linux |
| `APEXX-T4P-03` | workstation | - |
| `tp401-linux-r9700-vm`, `tp401-linux-w7900-vm`, `xsj-aimlab-radeon-7900-vm01` | Radeon GPU VMs | Linux |

They are kept alive by two scheduled workflows in that repo:
- `runner-heartbeat.yml` - each runner writes a heartbeat artifact weekly.
- `monitor-runners.yml` - checks heartbeats and fires a **Teams alert**
  (`TEAMS_WEBHOOK_URL` secret) if a runner goes stale (>8 days).

## How jobs target them (the label scheme)

There are **two** labelling conventions on these runners:

1. **Capability labels** - used by `test-playbooks.yml` to pick *any* matching
   machine. The label triple is built as `["self-hosted", <OS>, <device>]`
   where `<OS>` is `Windows` or `Linux` (capitalised) and `<device>` is the key
   from a playbook's `tested_platforms` (e.g. `halo`, `stxp`, `krk`, `halo_box`).
   Example: `runs-on: [self-hosted, Windows, halo]`.
2. **Machine-name labels** - used by `runner-heartbeat.yml` to hit one specific
   box: `runs-on: [self-hosted, "xsj-aimlab-halo-0"]`.

Our `test-code-samples.yml` uses convention #1, now corrected to the real
device labels:

```yaml
matrix:
  hw: [stxp, halo]          # Strix Point, Strix Halo
runs-on: [self-hosted, Windows, "${{ matrix.hw }}"]
```

> Previously this said `[self-hosted, windows, strix-halo]` - wrong on two
> counts: lowercase `windows` and the device label `strix-halo`/`strix`. The
> pool uses capitalised `Windows` and devices `halo` / `stxp`. A job with the
> wrong labels never gets picked up - it just queues forever.

## What access you need (and why you can't self-serve it)

The runners are **registered to the `amd/playbooks` repo / an AMD org runner
group**, not to your fork. Confirmed by API: listing the repo's runners returns
`403 (need repo admin)` and listing org runner-groups returns
`403 (need admin:org)`. So a fork under a personal account **cannot** see or use
them. To run on this hardware you need one of:

1. **(Recommended) Org runner group access.** Have the docs repo live under the
   `amd` org (this is the planned `amd/RyzenAI-SW` consolidation), then ask the
   Playbooks/lab admins to **add `amd/RyzenAI-SW` to the org runner group** that
   owns the `xsj-aimlab-*` machines. No new hardware, just a group membership.
2. **Run the docs CI from within `amd/playbooks`** (or a repo that already has
   the group) - least desirable; couples docs to their repo.
3. **Register your own runner** (e.g. this Strix Halo box) to your repo for
   testing only - fine as a stopgap, not the shared pool. See below.

### Who to ask

Recent `amd/playbooks` maintainers (commit history): `maxdokukin-amd`,
`anna-amd-com`, `ldokovic-personal`, Daniel Holanda (`danielholanda`), Sreeram.
Open an issue on `amd/playbooks` or ping them. Request template:

> Requesting that `amd/RyzenAI-SW` be added to the org runner group that hosts
> the `xsj-aimlab-halo-*` and `xsj-aimlab-stxp-*` runners, so our docs-as-code
> CI can execute `test`-tagged code samples on Strix Halo and Strix Point.
> Windows only, scheduled + on-PR, low volume. Labels used:
> `[self-hosted, Windows, halo]` and `[self-hosted, Windows, stxp]`.

## Stopgap: register THIS machine as a runner for your repo

This box is a Strix Halo (`AMD Ryzen AI Max+ Pro 395`, NPU present), so it can
serve as a real `halo` runner for your own repo while you wait for pool access:

1. Repo (or org) -> Settings -> Actions -> Runners -> New self-hosted runner -> Windows.
2. Follow the shown `config.cmd` download/registration steps (uses a one-time token).
3. When prompted for labels, add: `halo` (keep the default `self-hosted`,
   `Windows`). Now `runs-on: [self-hosted, Windows, halo]` resolves here.
4. Run it as a service so jobs are picked up without an interactive session:
   `./svc.cmd install && ./svc.cmd start`.

This proves the end-to-end hardware path on one machine; the org runner group
is what makes it the *shared* Strix Point + Strix Halo pool.

## Verify once you have access

```bash
# A job with these labels should leave "queued" and start within seconds
# if a matching idle runner exists in your runner group.
gh run list --workflow "Test Code Samples"
gh api repos/<owner>/<repo>/actions/runners --jq '.runners[] | {name,status,labels:[.labels[].name]}'
```
