# SPDX-License-Identifier: Apache-2.0
"""Cross-session reuse of the shared system prefix (system prompt + tools).

dflash-mlx's L1 lookup only resumes at boundaries that carry full resume
state: snapshot ends, sidecar boundaries, and stride-aligned frontier
snapshots. Two sessions that share a system prompt + tool definitions
diverge *inside* the old session's snapshots, so the lookup misses and the
conversation-switch relief in ``dflash_kv_prealloc`` clears L1 and the new
session re-prefills everything (~121 tok/s measured for DFlash prefill).

This patch adds one additional, exact boundary: the first user-turn marker
of the rendered prompt. The engine (``omlx.engine.dflash``) warms a base
snapshot by running a real ``max_tokens=1`` generation over exactly the
shared prefix — a real forward pass is the only way to materialize GDN
recurrent states at that position — and registers it here. Later, when a
new session's lookup misses, the wrapper serves the registered base as a
synthetic prefix hit so prefill resumes at the boundary instead of at 0.

Memory safety relies on capacity mode: the base's FA arrays alias the same
capacity-backed buffer the live session writes beyond, so serving it never
allocates a second buffer. The small state (GDN states, last_logits) is
cloned at capture so later generation cannot mutate the base in place.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_BACKUPS: dict[tuple[int, str], tuple[Any, str, Any]] = {}

# Single slot: in capacity mode two bases would alias the same physical KV
# rows and overwrite each other, so only the newest base is kept.
_ACTIVE_BASE: Optional["_BaseEntry"] = None
_CAPTURE_TARGET: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class _BaseEntry:
    key: Any
    token_prefix: tuple[int, ...]
    snapshot: Any
    created_at: float


def _backup(obj: Any, attr: str) -> None:
    key = (id(obj), attr)
    if key not in _BACKUPS:
        _BACKUPS[key] = (obj, attr, getattr(obj, attr))


def _clone_array(arr: Any) -> Any:
    try:
        from dflash_mlx.cache.codecs import _clone_array as clone

        return clone(arr)
    except Exception:
        cloned = mx.array(arr)
        mx.eval(cloned)
        return cloned


def _clone_gdn_states(
    gdn_states: Any,
) -> Any:
    if gdn_states is None:
        return None
    cloned_layers = []
    for layer in gdn_states:
        if layer is None:
            cloned_layers.append(None)
            continue
        cloned_layers.append(
            tuple(
                None if arr is None else _clone_array(arr) for arr in layer
            )
        )
    return tuple(cloned_layers)


def arm_capture(target_tokens: tuple[int, ...]) -> None:
    """Arm one-shot capture of the next prefill snapshot equal to target."""
    global _CAPTURE_TARGET
    _CAPTURE_TARGET = tuple(int(t) for t in target_tokens)


def disarm_capture() -> None:
    global _CAPTURE_TARGET
    _CAPTURE_TARGET = None


def active_base() -> Optional[_BaseEntry]:
    return _ACTIVE_BASE


def active_base_prefix() -> Optional[tuple[int, ...]]:
    return _ACTIVE_BASE.token_prefix if _ACTIVE_BASE is not None else None


def reset() -> None:
    """Drop the registered base (engine evict/stop unload the arrays)."""
    global _ACTIVE_BASE, _CAPTURE_TARGET
    _ACTIVE_BASE = None
    _CAPTURE_TARGET = None


def _capture_matches(snapshot: Any, target: tuple[int, ...], kind: Any) -> bool:
    """True when a built snapshot can yield a base equal to target.

    A warmup whose whole prompt is the stable prefix publishes no prefill
    snapshot (the runtime skips it when boundary == prompt_len) — only a
    generation snapshot whose sidecar_boundary sits exactly at the prompt
    end. That sidecar state (GDN + logits at the boundary) is precisely
    what the base needs, so generation snapshots are captured via the
    sidecar slice; prefill snapshots are captured directly.
    """
    if kind == "prefill":
        return tuple(snapshot.token_ids) == target
    if kind == "generation":
        boundary = int(getattr(snapshot, "sidecar_boundary", 0) or 0)
        return (
            boundary == len(target)
            and boundary > 0
            and snapshot.sidecar_gdn_states is not None
            and snapshot.sidecar_last_logits is not None
            and tuple(snapshot.token_ids[:boundary]) == target
        )
    return False


def _slice_at_sidecar(snapshot: Any) -> Any:
    try:
        from dflash_mlx.cache import prefix_l1

        sliced = prefix_l1.slice_snapshot_at_sidecar_boundary(snapshot)
        return sliced
    except Exception as exc:
        logger.warning("DFlash session base sidecar slice failed: %s", exc)
        return None


def _capture_base(snapshot: Any, target: tuple[int, ...]) -> None:
    global _ACTIVE_BASE
    if snapshot.kind == "generation":
        snapshot = _slice_at_sidecar(snapshot)
        if snapshot is None:
            return
    if tuple(snapshot.token_ids) != target:
        logger.debug(
            "DFlash session base capture skipped: token mismatch after slice"
        )
        return
    SnapshotT = type(snapshot)
    base_snapshot = SnapshotT(
        token_ids=snapshot.token_ids,
        fa_states=snapshot.fa_states,
        gdn_states=_clone_gdn_states(snapshot.gdn_states),
        target_hidden_chunks=snapshot.target_hidden_chunks,
        target_hidden_chunk_spans=snapshot.target_hidden_chunk_spans,
        target_hidden_total_len=snapshot.target_hidden_total_len,
        last_logits=(
            _clone_array(snapshot.last_logits)
            if snapshot.last_logits is not None
            else None
        ),
        key=snapshot.key,
        kind="prefill",
        created_at=snapshot.created_at,
        sidecar_boundary=0,
        sidecar_gdn_states=None,
        sidecar_last_logits=None,
    )
    _ACTIVE_BASE = _BaseEntry(
        key=snapshot.key,
        token_prefix=tuple(snapshot.token_ids),
        snapshot=base_snapshot,
        created_at=time.time(),
    )
    logger.info(
        "DFlash session base captured: %d tokens (kind=prefill, GDN state "
        "and last_logits cloned)",
        len(base_snapshot.token_ids),
    )


def _snapshot_covers(snapshot: Any, prefix_len: int) -> bool:
    try:
        from dflash_mlx.engine.prefill import snapshot_covers_prefix

        return bool(snapshot_covers_prefix(snapshot, prefix_len))
    except Exception:
        return False


def _serve_base(req_tuple: tuple[int, ...], key: Any) -> Optional[tuple[int, Any]]:
    base = _ACTIVE_BASE
    if base is None or key != base.key:
        return None
    prefix_len = len(base.token_prefix)
    if prefix_len <= 0 or len(req_tuple) < prefix_len:
        return None
    if req_tuple[:prefix_len] != base.token_prefix:
        return None
    return prefix_len, base.snapshot


def install() -> bool:
    """Install the session-base capture and serve patches.

    Composes with ``dflash_kv_prealloc``: whichever installs first ends up
    innermost; both wrappers pass misses through so ordering only changes
    logging. Returns True when at least one hook was installed or already
    present.
    """
    installed = False

    try:
        from dflash_mlx.cache import codecs as _codecs
    except ImportError:
        logger.debug("dflash_mlx.cache.codecs not importable; session base skipped")
        return False

    builder_cls = getattr(_codecs, "PrefixSnapshotBuilder", None)
    if builder_cls is not None and not getattr(
        builder_cls.build, "_omlx_dflash_session_base", False
    ):
        _backup(builder_cls, "build")
        original_build = builder_cls.build

        def build(self: Any, **kwargs: Any) -> Any:
            snapshot = original_build(self, **kwargs)
            target = _CAPTURE_TARGET
            if target is not None and _capture_matches(
                snapshot, target, kwargs.get("kind")
            ):
                try:
                    _capture_base(snapshot, target)
                except Exception as exc:
                    logger.warning("DFlash session base capture failed: %s", exc)
            return snapshot

        build._omlx_dflash_session_base = True  # type: ignore[attr-defined]
        builder_cls.build = build
        installed = True

    try:
        from dflash_mlx.cache import prefix_l1 as _prefix_l1
    except ImportError:
        logger.debug("dflash_mlx.cache.prefix_l1 not importable; session base skipped")
        return installed

    l1_cls = getattr(_prefix_l1, "DFlashPrefixCache", None)
    if l1_cls is not None and not getattr(
        l1_cls.lookup, "_omlx_dflash_session_base", False
    ):
        _backup(l1_cls, "lookup")
        original_lookup = l1_cls.lookup

        def lookup(
            self: Any,
            req_tokens: Any,
            key: Any,
            *,
            record: bool = True,
            request_id: int | None = None,
            require_full_coverage: bool = False,
        ) -> Any:
            result = original_lookup(
                self,
                req_tokens,
                key,
                record=record,
                request_id=request_id,
                require_full_coverage=require_full_coverage,
            )
            if result[0] == 0:
                req_tuple = tuple(int(t) for t in req_tokens)
                served = _serve_base(req_tuple, key)
                if served is not None and (
                    not require_full_coverage
                    or _snapshot_covers(served[1], served[0])
                ):
                    # Miss means every L1 entry is dead lineage for this
                    # request. In capacity mode those snapshots alias the
                    # shared buffer and would serve overwritten rows if the
                    # old session ever returns, so drop them before serving.
                    try:
                        self.clear()
                        mx.clear_cache()
                    except Exception as exc:
                        logger.debug("session base lineage purge failed: %s", exc)
                    logger.info(
                        "DFlash session base reuse: serving %d-token system "
                        "prefix as synthetic hit",
                        served[0],
                    )
                    return served
            return result

        lookup._omlx_dflash_session_base = True  # type: ignore[attr-defined]
        l1_cls.lookup = lookup
        installed = True

    return installed


def restore() -> None:
    """Restore patched methods and drop all module state."""
    global _ACTIVE_BASE, _CAPTURE_TARGET
    for _key, (obj, attr, original) in list(_BACKUPS.items()):
        try:
            setattr(obj, attr, original)
        except Exception as exc:
            logger.debug("session base restore failed for %r.%s: %s", obj, attr, exc)
    _BACKUPS.clear()
    _ACTIVE_BASE = None
    _CAPTURE_TARGET = None


__all__ = [
    "active_base",
    "active_base_prefix",
    "arm_capture",
    "disarm_capture",
    "install",
    "reset",
    "restore",
]
