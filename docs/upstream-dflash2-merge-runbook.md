# Upstream DFlash2 Merge Runbook

Last updated: 2026-08-20

## Background

We want this fork to track the official `jundot/omlx` branch as closely as
possible, and keep only the local changes that are specific to our single-user
long-context DFlash workflow.

Current local state at the time of evaluation:

- Local branch: `main` at `4c4fa238` (`Reduce DFlash KV prealloc probe logging`).
- Existing remote `origin`: `https://github.com/z-lab/omlx-fork.git`.
- Upstream evaluated: `https://github.com/jundot/omlx.git`, fetched as
  `jundot/main` at `146d2724` (`formula: bump to 0.6.3rc1`).
- Divergence from local `main` to `jundot/main`: local-only `15` commits,
  upstream-only `42` commits.

The upstream branch has now absorbed the DFlash2 support that this fork carried
earlier, plus additional work around DFlash2 runtime plumbing, ANE prefill,
Laguna decode optimizations, cluster reliability, CLI/API fixes, and benchmark
UI improvements. The long-term maintenance target is therefore:

1. Treat `jundot/main` as the base implementation.
2. Reapply only our local single-user DFlash KV capacity optimization.
3. Keep that local delta small, documented, and guarded by focused tests.

## Upstream Changes To Accept

Accept the upstream versions unless a local KV-capacity behavior explicitly
depends on the old code:

- `pyproject.toml`: accept the upstream `dflash-mlx` pin bump to
  `b7f625504a733216d9e06caa3f7847a048241d94` (`0.1.10+omlx.6`). This runtime
  includes DFlash2 config handling, sampling support, candidate selector logic,
  `min_p`, and the sink=0 L2 write fix.
- `omlx/patches/dflash_draft_config.py`: accept upstream removal. The new
  `dflash-mlx` runtime has the DFlash2 draft config compatibility in
  `DFlashDraftModelArgs.from_dict`, so the oMLX-side normalizer is no longer
  needed.
- `omlx/engine/dflash.py`: accept upstream DFlash request plumbing for
  `min_p`, per-request `seed`, updated preflight comments for stock DFlash,
  and the removal of non-prealloc live-residency memory credit.
- Admin/macOS UI changes for DFlash2 defaults and draft filtering should be
  accepted. Our KV capacity knobs do not need to be first-class UI controls;
  they can remain advanced JSON/model-settings fields unless we explicitly
  decide otherwise.
- Accept the Laguna and Qwen3.5 ANE performance work, cluster reliability work,
  API fixes, and benchmark/admin UI improvements unless tests show a direct
  regression in our DFlash path.

## Local Changes To Preserve

The local feature that must survive the merge is the DFlash single-user
long-context KV capacity mode:

- `omlx/patches/dflash_kv_prealloc.py`
  - Keep the file.
  - It patches `mlx_lm.models.cache.KVCache.update_and_fetch`,
    `dflash_mlx.cache.codecs` snapshot adoption/hydration,
    `PrefixSnapshotBuilder`, `slice_snapshot_at_sidecar_boundary`,
    `DFlashPrefixCache.lookup`, and `FullContextDraftKVCache._grow`.
  - Verified against `dflash-mlx@b7f6255` (fetched and inspected 2026-08-20):
    every hook point above still exists. All wrapped entry points use
    wrap-and-delegate (backing up and calling `original_slice` /
    `original_lookup` / `_grow`), so upstream's sink=0 empty-chunk L2 write fix
    flows through untouched. The only re-implemented paths are the
    `clone=False` branch of `serialize_target_cache` and
    `hydrate_target_cache`, which only touch KV arrays and do not interact
    with the sink packing path. The `mlx-lm` pin is identical on both branches
    (`ab1806e`), so the `KVCache.update_and_fetch` hook is stable.

- `ModelSettings`
  - Preserve these fields:
    - `dflash_kv_capacity_tokens: int = 0`
    - `dflash_prefill_step_size: Optional[int] = None`
    - `dflash_max_snapshot_tokens: Optional[int] = None`
  - Preserve JSON round-trip behavior and existing tests.
  - These can stay hidden from the normal UI; they are operational knobs for
    a specialized deployment profile.

- `DFlashEngine`
  - Preserve parsing of the three local settings above.
  - Preserve passing `prefill_step_size` and `max_snapshot_tokens` into
    `runtime_config_from_defaults`.
  - Preserve the L1 byte-budget floor when `dflash_kv_capacity_tokens > 0`:
    `max(configured_l1_bytes, capacity_tokens * 80 * 1024)`.
  - Preserve installing `install_dflash_kv_prealloc(capacity)` after target and
    draft models are loaded, before serving requests.
  - Preserve `restore_dflash_kv_prealloc()` on DFlash eviction and `stop()`.
  - Preserve the explicit KV capacity preflight check:
    `prompt_tokens + max_tokens <= dflash_kv_capacity_tokens`.
  - Revisit live-residency credit carefully: upstream removed it for stock
    DFlash because snapshot hydration allocates KV during the request. If we
    keep a residency credit, gate it behind `dflash_kv_capacity_tokens > 0` and
    document why capacity-backed reuse makes the accounting different.

- `omlx/scheduler.py`
  - Keep the two keep-flat tuning lines from `bb5f333b`:
    `mlx_cache_cleanup_interval` 512 -> 16 steps, and the periodic clear
    threshold `max(limit // 3, 2GB)` -> `max(limit // 16, 1GB)`. With
    capacity-backed KV the allocator pool stays small, and this cadence is
    part of what keeps the steady-state footprint flat. Auto-merges cleanly
    (4 lines); do not hand-resolve it away.

- `omlx/server.py`
  - Keep the local request-path wiring from `bb5f333b`, notably the
    `preflight_completion` call that surfaces the engine-level capacity
    preflight as request rejection (~102 lines, auto-merges cleanly).
    Upstream additions such as `distributed_unavailable_handler` come in with
    the merge and must survive resolution.

- Tests and docs
  - Preserve `tests/test_dflash_kv_prealloc.py`.
  - Preserve `tests/test_dflash_prefill_memory_guard.py`, but treat it as
    coupled to the `_resident_cache_tokens` decision above: its 69 local
    lines encode the fork's admission expectations, so if the residency
    credit is dropped or re-gated, update these tests in the same change or
    the suite fails post-merge.
  - Preserve/refresh `docs/dflash-single-session-memory.md` and
    `docs/kv-prealloc-plan.md`. They are local-only additions, so the merge
    keeps them automatically; the existence check in §5 is insurance only.

## Expected Merge Conflicts

A dry-run merge with `git merge-tree --write-tree main jundot/main` reported
three content conflicts:

1. `omlx/engine/dflash.py`
2. `omlx/model_settings.py`
3. `tests/test_dflash_engine.py`

Most other upstream files auto-merge cleanly. Two silent deletions to expect:
`omlx/patches/dflash_draft_config.py` and
`tests/test_dflash_draft_config.py` exist in the merge base, are unchanged
locally, and are absent from `jundot/main`, so the merge removes them without
stopping. A merged-tree grep confirmed no stale `dflash_draft_config`
references remain afterwards.

The effective post-merge delta versus `jundot/main` is exactly the file set
of local commit `bb5f333b` plus the three conflict files listed above: the
KV prealloc module and its test, the two docs, `model_settings.py`,
`engine/dflash.py`, `scheduler.py`, `server.py`, `test_dflash_engine.py`,
and `test_dflash_prefill_memory_guard.py`. Nothing outside `omlx/`,
`tests/`, and `docs/` survives from the local side.

## Execution Plan

### 1. Prepare A Merge Branch

Start from a clean worktree.

```bash
git status --short
```

Fetch the upstream branch into a stable local reference. If an `upstream` remote
does not exist, either add it or fetch directly into `jundot/main`.

```bash
git remote add jundot https://github.com/jundot/omlx.git 2>/dev/null || true
git fetch jundot main:refs/remotes/jundot/main
git switch -c merge/jundot-main-dflash2
```

Run the merge.

```bash
git merge --no-ff jundot/main
```

### 2. Resolve `omlx/model_settings.py`

Base the file on upstream, then re-add only the local KV settings.

Required resolution:

- Keep upstream DFlash defaults and comments for DFlash2.
- Re-add the three local settings to the dataclass.
- Re-add their docstring entries.
- Ensure `from_dict()` still accepts persisted settings files containing these
  fields.
- Ensure `to_dict()` still includes `dflash_kv_capacity_tokens == 0` as today,
  while omitting `None` optional fields.

Do not re-add removed `dflash_speculative_tokens` behavior.

### 3. Resolve `omlx/engine/dflash.py`

Base the file on upstream, then reapply the local KV capacity hooks.

Keep from upstream:

- DFlash2-compatible `dflash-mlx` assumptions.
- Removal of `dflash_draft_config` normalizer.
- `min_p` and per-request `seed` plumbing.
- Stock-DFlash preflight behavior that charges full prompt KV when capacity
  mode is disabled.

Reapply locally:

- `_nonnegative_int`, `_positive_int_or_none`, and `_nonnegative_int_or_none`
  helpers if still needed for local settings parsing.
- `self._kv_capacity_tokens`, `self._dflash_prefill_step_size`, and
  `self._dflash_max_snapshot_tokens` initialization.
- L1 max-byte floor in `_build_runtime_context()` when capacity mode is on.
- `prefill_step_size=` and `max_snapshot_tokens=` arguments to
  `runtime_config_from_defaults()`.
- `install_dflash_kv_prealloc()` in `start()` when capacity mode is on.
- `restore_dflash_kv_prealloc()` in `_evict_dflash_and_start_fallback()` and
  `stop()`.
- `_preflight_kv_capacity_or_raise()` and calls from chat/completion/generate
  paths before generation starts.
- Capacity-mode log line showing capacity, estimated target KV, prefill step,
  and max snapshot tokens.

Decision point:

- If keeping `_resident_cache_tokens`, enable its credit only when
  `self._kv_capacity_tokens > 0`. Keep upstream full-prompt accounting when
  capacity mode is disabled.

### 4. Resolve `tests/test_dflash_engine.py`

Base the tests on upstream, then re-add local expectations for:

- `ModelSettings` defaults for the three KV fields.
- `to_dict()` / `from_dict()` round-trip of local fields.
- Runtime config propagation of `dflash_prefill_step_size` and
  `dflash_max_snapshot_tokens`.
- Capacity preflight rejection behavior.
- Upstream `min_p` / `seed` tests must remain intact.

### 5. Restore Local Files If The Merge Deletes Them

After resolving conflicts, verify these files still exist:

```bash
test -f omlx/patches/dflash_kv_prealloc.py
test -f tests/test_dflash_kv_prealloc.py
test -f docs/dflash-single-session-memory.md
test -f docs/kv-prealloc-plan.md
```

If any are removed by the merge, restore them from local `main`:

```bash
git restore --source=main -- \
  omlx/patches/dflash_kv_prealloc.py \
  tests/test_dflash_kv_prealloc.py \
  docs/dflash-single-session-memory.md \
  docs/kv-prealloc-plan.md
```

### 6. Run Focused Tests

Run the narrow DFlash and settings tests first.

```bash
pytest tests/test_dflash_kv_prealloc.py \
  tests/test_dflash_engine.py \
  tests/test_dflash_prefill_memory_guard.py \
  tests/test_dflash_laguna.py \
  tests/test_dflash_muse_glimmer.py
```

Then run tests touched by upstream changes that are likely to interact with the
merge.

```bash
pytest tests/test_admin_model_settings_template.py \
  tests/test_admin_profiles_api.py \
  tests/test_benchmark.py \
  tests/test_cli.py \
  tests/test_responses.py
```

If time permits before landing, run a wider pass around DFlash, ANE, and
settings:

```bash
pytest tests/test_dflash_*.py tests/test_qwen35_ane_prefill.py tests/test_ane_tuning.py
```

### 7. Manual Smoke Tests

Both smokes must run in a dedicated environment with the upstream runtime
actually installed — never through the app bundle interpreter (`PYTHONHOME`
against `/Applications/oMLX.app`), which still carries `0.1.10+omlx.5`. A
smoke that silently passes against the old wheel says nothing about
`omlx.6`; this is the same environment-drift class that broke the dist venv
on 2026-08-20.

```bash
uv venv .venv-merge --python 3.12
uv pip install --python .venv-merge/bin/python \
  "dflash-mlx @ git+https://github.com/jundot/dflash-mlx@b7f625504a733216d9e06caa3f7847a048241d94"
# then install the merged omlx tree into the same venv (editable install is fine)
```

Do at least one runtime smoke with DFlash capacity mode disabled and one with it
enabled.

Capacity disabled (`dflash_kv_capacity_tokens = 0`):

- Confirm ordinary DFlash2 generation works.
- Confirm `min_p` and `seed` are accepted without error. Note that
  `stream_dflash_generate` in `b7f6255` has no `seed` parameter (only
  temperature/top_p/top_k/min_p): verify seed handling at the oMLX layer
  rather than assuming it is passed through to the dflash runtime.
- Confirm memory preflight charges the full prompt, matching upstream behavior.

Capacity enabled:

- Use the single-user long-context profile, e.g.
  `dflash_kv_capacity_tokens = 163840`, `dflash_prefill_step_size = 512`.
- Confirm startup logs include `DFlash KV prealloc enabled`.
- Run a multi-turn same-session continuation and confirm the capacity buffer is
  reused rather than reallocated every turn.
- Switch to a different conversation and confirm the L1 release log appears:
  `KV prealloc conversation switch: released ... pinned capacity buffer(s)`.
- Confirm a request where `prompt_tokens + max_tokens` exceeds capacity returns
  a clean 400-style preflight error rather than OOMing.

## Definition Of Done

The merge is ready to land when all of the following are true:

- The branch is based on `jundot/main` and contains only the intentional local
  KV-capacity delta on top.
- `dflash-mlx` is pinned to upstream `b7f6255` or newer, not the old local
  DFlash2 pin.
- `dflash_draft_config.py` remains removed.
- `dflash_kv_prealloc.py` remains present and installed/restored only when
  `dflash_kv_capacity_tokens > 0`.
- Local KV settings round-trip through `ModelSettings`.
- Stock DFlash mode still follows upstream memory accounting.
- Capacity mode still passes the single-user long-context smoke test.
- Focused tests above pass, or any skipped tests are explained in the merge PR
  notes with environment limitations.
- Post-merge `git diff jundot/main HEAD --stat` lists exactly the intended
  preserve set (KV prealloc module and test, the two docs, model settings,
  engine, scheduler, server, and the two DFlash test files); any additional
  file in that diff must be explained in the merge notes.
- Both manual smokes ran in a venv with `dflash-mlx@b7f6255` installed, not
  through the app bundle interpreter.

## Downstream: Refresh omlx-dist

The merge invalidates the current dist state: the vendored wheel is
`0.1.10+omlx.5`, dist `pyproject.toml` pins `dflash-mlx==0.1.10+omlx.5`, and
dist carries the now-obsolete `omlx/patches/dflash_draft_config.py`
weight-filter shim. After the merge branch lands on fork `main`:

1. Vendor the new runtime: install `dflash-mlx@b7f6255` into a scratch venv,
   rebuild the wheel from the installed package (same RECORD-regeneration
   procedure used for the omlx.5 wheel), and replace
   `vendor/dflash_mlx-0.1.10+omlx.5-py3-none-any.whl` with the
   `0.1.10+omlx.6` build.
2. Update dist `pyproject.toml` to `dflash-mlx==0.1.10+omlx.6` (keep the
   `--find-links vendor` install flow).
3. Regenerate dist `omlx/` from the merged fork. The weight-filter file and
   its engine wiring disappear with the merge: the new runtime consumes the
   DFlash2 draft tensors natively.
4. Re-run the end-to-end verification: fresh venv install from vendor, parity
   audit of the key packages, service health, and the prose/code bench
   (`bench_spec.py`) against the redeployed service.
5. Commit dist and re-sync the distribution copy at `~/synol/juz/omlx-dist`
   (same excludes as before: `.git`, `.venv`, `.omlx`; `models/` on the
   target stays untouched).

## Follow-Up Cleanup

After the merge lands, consider moving the KV-capacity feature behind a clearly
named local profile or documented JSON snippet instead of broad UI exposure.
This keeps future upstream merges smaller: most official DFlash changes can be
accepted wholesale, and the only recurring local conflict should remain the
small `DFlashEngine`/`ModelSettings` capacity-mode overlay.
