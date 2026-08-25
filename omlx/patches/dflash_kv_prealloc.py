# SPDX-License-Identifier: Apache-2.0
"""Capacity-backed KV caches for DFlash long-context single-session runs.

The patch is intentionally narrow and process-local. While installed, plain
``mlx_lm.models.cache.KVCache`` and dflash-mlx's full-context draft cache grow
directly to a fixed token capacity on first use. dflash-mlx prefix snapshots can
then adopt those physical arrays without trimming them back to logical length,
so hydrating the next request keeps the same capacity-backed buffers instead of
forcing another concat growth cycle.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_CAPACITY_TOKENS = 0
_BACKUPS: dict[tuple[int, str], tuple[Any, str, Any]] = {}


def _backup(obj: Any, attr: str) -> None:
    key = (id(obj), attr)
    if key not in _BACKUPS:
        _BACKUPS[key] = (obj, attr, getattr(obj, attr))


def _capacity() -> int:
    return max(0, int(_CAPACITY_TOKENS))


def _shape_mismatch(
    cache: Any, keys: mx.array, values: mx.array, capacity: int
) -> bool:
    if getattr(cache, "keys", None) is None or getattr(cache, "values", None) is None:
        return True
    return (
        int(cache.keys.shape[0]) != int(keys.shape[0])
        or int(cache.keys.shape[1]) != int(keys.shape[1])
        or int(cache.keys.shape[2]) < int(capacity)
        or int(cache.keys.shape[3]) != int(keys.shape[3])
        or int(cache.values.shape[0]) != int(values.shape[0])
        or int(cache.values.shape[1]) != int(values.shape[1])
        or int(cache.values.shape[2]) < int(capacity)
        or int(cache.values.shape[3]) != int(values.shape[3])
    )


def _ensure_capacity(
    cache: Any, keys: mx.array, values: mx.array, capacity: int
) -> None:
    if not _shape_mismatch(cache, keys, values, capacity):
        return

    prev = max(0, int(getattr(cache, "offset", 0) or 0))
    new_k = mx.zeros(
        (
            int(keys.shape[0]),
            int(keys.shape[1]),
            int(capacity),
            int(keys.shape[3]),
        ),
        dtype=keys.dtype,
    )
    new_v = mx.zeros(
        (
            int(values.shape[0]),
            int(values.shape[1]),
            int(capacity),
            int(values.shape[3]),
        ),
        dtype=values.dtype,
    )
    old_k = getattr(cache, "keys", None)
    old_v = getattr(cache, "values", None)
    if old_k is not None and old_v is not None and prev > 0:
        copy_len = min(prev, int(old_k.shape[2]), int(old_v.shape[2]), int(capacity))
        if copy_len > 0:
            new_k[..., :copy_len, :] = old_k[..., :copy_len, :]
            new_v[..., :copy_len, :] = old_v[..., :copy_len, :]
    cache.keys = new_k
    cache.values = new_v
    mx.eval(cache.keys, cache.values)
    logger.debug(
        "KV prealloc realloc: %s old_shape=%s offset=%d -> capacity=%d "
        "(%.2fGB per array pair)",
        type(cache).__name__,
        tuple(int(d) for d in old_k.shape) if old_k is not None else None,
        prev,
        capacity,
        (new_k.nbytes + new_v.nbytes) / 1024**3,
    )


def _install_mlx_lm_kv_cache(capacity: int) -> bool:
    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        logger.debug("mlx_lm KVCache not importable; DFlash KV prealloc skipped")
        return False

    if getattr(KVCache.update_and_fetch, "_omlx_dflash_kv_prealloc", False):
        return True
    _backup(KVCache, "update_and_fetch")
    original = KVCache.update_and_fetch

    def update_and_fetch(
        self: Any, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        cap = _capacity()
        if cap <= 0:
            return original(self, keys, values)
        prev = max(0, int(getattr(self, "offset", 0) or 0))
        new_tokens = int(keys.shape[2])
        needed = prev + new_tokens
        if needed > cap:
            raise RuntimeError(
                f"KV cache capacity exceeded: requested {needed} tokens, "
                f"reserved {cap}"
            )
        _ensure_capacity(self, keys, values, cap)
        self.offset = needed
        self.keys[..., prev:needed, :] = keys
        self.values[..., prev:needed, :] = values
        return self.keys[..., :needed, :], self.values[..., :needed, :]

    update_and_fetch._omlx_dflash_kv_prealloc = True  # type: ignore[attr-defined]
    KVCache.update_and_fetch = update_and_fetch
    return True


def _install_dflash_codecs() -> bool:
    try:
        from dflash_mlx.cache import codecs
    except ImportError:
        logger.debug("dflash_mlx.cache.codecs not importable; codec patch skipped")
        return False

    installed = False
    if not getattr(codecs.serialize_target_cache, "_omlx_dflash_kv_prealloc", False):
        _backup(codecs, "serialize_target_cache")
        original_serialize = codecs.serialize_target_cache

        def serialize_target_cache(
            target_cache: list[Any], *, clone: bool = True
        ) -> Any:
            if clone:
                return original_serialize(target_cache, clone=clone)
            fa_nbytes = 0
            entry_types: dict[str, int] = {}
            for probe in target_cache:
                entry_types[type(probe).__name__] = (
                    entry_types.get(type(probe).__name__, 0) + 1
                )
                probe_k = getattr(probe, "keys", None)
                if probe_k is not None:
                    probe_v = getattr(probe, "values", None)
                    fa_nbytes += int(probe_k.nbytes) + int(
                        probe_v.nbytes if probe_v is not None else 0
                    )
            logger.debug(
                "KV prealloc serialize(adopt): entries=%s fa_physical=%.2fGB "
                "capacity=%d",
                entry_types,
                fa_nbytes / 1024**3,
                _capacity(),
            )

            def grab(a: Any) -> Any:
                return a

            fa: list[Any] = []
            gdn: list[Any] = []
            for layer_idx, entry in enumerate(target_cache):
                if isinstance(entry, codecs.RecurrentRollbackCache):
                    fa.append(None)
                    gdn.append(tuple(grab(a) for a in entry.cache))
                elif isinstance(entry, codecs.RotatingKVCache):
                    keys = getattr(entry, "keys", None)
                    values = getattr(entry, "values", None)
                    if keys is None or values is None:
                        fa.append(None)
                        gdn.append(None)
                    else:
                        fa.append(
                            (
                                grab(keys),
                                grab(values),
                                int(entry.offset),
                                int(entry._idx),
                            )
                        )
                        gdn.append(None)
                elif isinstance(entry, codecs.KVCache):
                    keys = getattr(entry, "keys", None)
                    values = getattr(entry, "values", None)
                    if keys is None or values is None:
                        fa.append(None)
                        gdn.append(None)
                    else:
                        fa.append((grab(keys), grab(values), int(entry.offset)))
                        gdn.append(None)
                else:
                    raise TypeError(
                        f"Cache entry type {type(entry).__name__} at layer {layer_idx} "
                        "is not supported for prefix-cache serialization."
                    )
            return tuple(fa), tuple(gdn)

        serialize_target_cache._omlx_dflash_kv_prealloc = True  # type: ignore[attr-defined]
        codecs.serialize_target_cache = serialize_target_cache
        installed = True

    if not getattr(codecs.hydrate_target_cache, "_omlx_dflash_kv_prealloc", False):
        _backup(codecs, "hydrate_target_cache")

        def hydrate_target_cache(snapshot: Any, template_cache: list[Any]) -> list[Any]:
            if len(template_cache) != len(snapshot.fa_states):
                raise ValueError(
                    f"Template cache length {len(template_cache)} != "
                    f"snapshot layer count {len(snapshot.fa_states)}"
                )
            codecs._validate_snapshot_cache_prefix_len(
                snapshot.fa_states,
                prefix_len=snapshot.prefix_len,
            )

            result: list[Any] = []
            for i, tmpl in enumerate(template_cache):
                fa_state = snapshot.fa_states[i]
                gdn_state = snapshot.gdn_states[i]
                if isinstance(tmpl, codecs.RecurrentRollbackCache):
                    if gdn_state is None:
                        raise ValueError(f"Snapshot missing GDN state at layer {i}")
                    new_cache = codecs.RecurrentRollbackCache(
                        size=len(tmpl.cache),
                        conv_kernel_size=tmpl.conv_kernel_size,
                    )
                    new_cache.cache = list(gdn_state)
                    result.append(new_cache)
                elif isinstance(tmpl, codecs.KVCache):
                    if fa_state is None:
                        raise ValueError(f"Snapshot missing FA state at layer {i}")
                    k, v, offset = fa_state[:3]
                    if int(k.shape[2]) < int(offset) or int(v.shape[2]) < int(offset):
                        raise ValueError(
                            f"Snapshot FA arrays at layer {i} are shorter than "
                            f"offset (keys={int(k.shape[2])}, values={int(v.shape[2])}, "
                            f"offset={int(offset)}); cannot adopt"
                        )
                    new_cache = codecs.KVCache()
                    new_cache.keys = k
                    new_cache.values = v
                    new_cache.offset = int(offset)
                    logger.debug(
                        "KV prealloc hydrate adopt: layer=%d array_len=%d "
                        "offset=%d %s",
                        i,
                        int(k.shape[2]),
                        int(offset),
                        (
                            "capacity-backed"
                            if int(k.shape[2]) >= _capacity()
                            else "UNDERSIZED (will realloc on next update)"
                        ),
                    )
                    result.append(new_cache)
                elif isinstance(tmpl, codecs.RotatingKVCache):
                    if fa_state is None:
                        raise ValueError(
                            f"Snapshot missing rotating FA state at layer {i}"
                        )
                    if len(fa_state) != 4:
                        raise ValueError(
                            f"Snapshot missing rotating FA ring index at layer {i}"
                        )
                    k, v, offset = fa_state[:3]
                    new_cache = codecs.RotatingKVCache(
                        max_size=int(tmpl.max_size),
                        keep=int(tmpl.keep),
                    )
                    new_cache.keys = codecs._clone_array(k)
                    new_cache.values = codecs._clone_array(v)
                    new_cache.offset = int(offset)
                    new_cache._idx = int(fa_state[3])
                    result.append(new_cache)
                else:
                    raise TypeError(
                        f"Cannot hydrate cache of type {type(tmpl).__name__} at layer {i}"
                    )
            return result

        hydrate_target_cache._omlx_dflash_kv_prealloc = True  # type: ignore[attr-defined]
        codecs.hydrate_target_cache = hydrate_target_cache
        installed = True

    # Prefill-phase snapshots (kind="prefill") are built with clone=True by
    # default, materializing logical-length copies. In agent continuation the
    # next request usually serves from those (the generation snapshot's token
    # ids diverge once the reply is re-templated), so a cloned prefill
    # snapshot forces a capacity realloc — and a transient second
    # capacity-sized buffer — on every turn. Force the zero-copy adopt path
    # for all snapshot kinds while capacity mode is armed; the arrays are
    # shared with the live session, which only ever appends beyond the
    # snapshotted offset.
    builder_cls = getattr(codecs, "PrefixSnapshotBuilder", None)
    if builder_cls is not None and not getattr(
        builder_cls.build, "_omlx_dflash_kv_prealloc", False
    ):
        _backup(builder_cls, "build")
        original_build = builder_cls.build

        def build(self: Any, **kwargs: Any) -> Any:
            if _capacity() > 0 and not kwargs.get("adopt_cache_arrays", False):
                kwargs = dict(kwargs)
                kwargs["adopt_cache_arrays"] = True
            return original_build(self, **kwargs)

        build._omlx_dflash_kv_prealloc = True  # type: ignore[attr-defined]
        builder_cls.build = build
        installed = True

    # The sidecar path slices the carrier snapshot's FA arrays at the
    # boundary, producing logical-length views that trigger the same
    # per-turn realloc. Keep the full (capacity-backed) arrays and encode the
    # boundary as the FA state offset instead. prefix_l1 imported the
    # original by name, so rebind it there too.
    try:
        from dflash_mlx.cache import prefix_l1 as _prefix_l1
    except ImportError:
        _prefix_l1 = None

    # Both 0.1.10+omlx.5 and .6 re-export this from dflash_mlx.cache.codecs;
    # guard anyway so a future relocation skips the sidecar wrap instead of
    # crashing install.
    SnapshotT = getattr(codecs, "DFlashPrefixSnapshot", None)

    if (
        _prefix_l1 is not None
        and SnapshotT is not None
        and not getattr(
            _prefix_l1.slice_snapshot_at_sidecar_boundary,
            "_omlx_dflash_kv_prealloc",
            False,
        )
    ):
        _backup(_prefix_l1, "slice_snapshot_at_sidecar_boundary")
        original_slice = _prefix_l1.slice_snapshot_at_sidecar_boundary
        covers = None
        try:
            from dflash_mlx.engine.prefill import snapshot_covers_prefix as covers
        except ImportError:
            covers = None

        def slice_snapshot_at_sidecar_boundary(
            snapshot: Any, *, require_full_coverage: bool = False
        ) -> Any:
            if _capacity() <= 0:
                return original_slice(
                    snapshot, require_full_coverage=require_full_coverage
                )
            boundary = int(snapshot.sidecar_boundary)
            if not 0 < boundary < snapshot.prefix_len:
                raise ValueError(
                    f"Sidecar boundary {boundary} outside (0, {snapshot.prefix_len})"
                )
            if (
                snapshot.sidecar_gdn_states is None
                or snapshot.sidecar_last_logits is None
            ):
                raise ValueError(
                    "Snapshot has a sidecar boundary but no sidecar states"
                )
            if (
                require_full_coverage
                and covers is not None
                and not covers(snapshot, boundary)
            ):
                raise ValueError(
                    f"Snapshot feature spans do not cover sidecar boundary {boundary}"
                )
            fa: list[Any] = []
            for layer_idx, state in enumerate(snapshot.fa_states):
                if state is None:
                    fa.append(None)
                    continue
                if len(state) != 3:
                    raise ValueError(
                        f"FA state at layer {layer_idx} is not boundary-sliceable"
                    )
                k, v, _offset = state
                fa.append((k, v, boundary))
            chunks: list[Any] = []
            spans: list[tuple[int, int]] = []
            for chunk, (start, end) in zip(
                snapshot.target_hidden_chunks,
                snapshot.target_hidden_chunk_spans,
            ):
                if start >= boundary:
                    continue
                keep = min(end, boundary) - start
                chunks.append(chunk[:, :keep, :])
                spans.append((start, start + keep))
            return SnapshotT(
                token_ids=snapshot.token_ids[:boundary],
                fa_states=tuple(fa),
                gdn_states=snapshot.sidecar_gdn_states,
                target_hidden_chunks=tuple(chunks),
                target_hidden_chunk_spans=tuple(spans),
                target_hidden_total_len=boundary,
                last_logits=snapshot.sidecar_last_logits,
                key=snapshot.key,
                kind="prefill",
                created_at=snapshot.created_at,
            )

        slice_snapshot_at_sidecar_boundary._omlx_dflash_kv_prealloc = True  # type: ignore[attr-defined]
        _prefix_l1.slice_snapshot_at_sidecar_boundary = (
            slice_snapshot_at_sidecar_boundary
        )
        installed = True

    # Conversation-switch relief: on a recorded lookup miss (a different
    # conversation — the resident one's tokens no longer prefix-match), the
    # old generation snapshot still pins a full capacity-sized buffer while
    # the fresh session allocates a second one, doubling the KV footprint
    # for the whole first prefill (~37GB vs ~28GB). Drop entries that pin
    # capacity buffers and return the memory to the OS before the new
    # session's first update; the fresh zeros() then reuses the released
    # wired region instead of growing past it.
    _L1Cls = getattr(_prefix_l1, "DFlashPrefixCache", None) if _prefix_l1 else None
    if _L1Cls is not None and not getattr(
        _L1Cls.lookup, "_omlx_dflash_kv_prealloc_switch", False
    ):
        _backup(_L1Cls, "lookup")
        original_lookup = _L1Cls.lookup

        def _pins_capacity(snapshot: Any, cap: int) -> bool:
            for state in snapshot.fa_states:
                if state is None:
                    continue
                k = state[0]
                if k is not None and int(k.shape[2]) >= cap:
                    return True
            return False

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
            cap = _capacity()
            if record and cap > 0 and result[0] == 0:
                try:
                    pinned = [
                        s for s in self.resident_snapshots() if _pins_capacity(s, cap)
                    ]
                    if pinned:
                        self.clear()
                        mx.clear_cache()
                        logger.info(
                            "KV prealloc conversation switch: released %d pinned "
                            "capacity buffer(s) before fresh allocation",
                            len(pinned),
                        )
                except Exception as exc:
                    logger.debug("KV prealloc switch release failed: %s", exc)
            return result

        lookup._omlx_dflash_kv_prealloc_switch = True  # type: ignore[attr-defined]
        _L1Cls.lookup = lookup
        installed = True

    return installed


def _install_dflash_full_context_draft_cache(capacity: int) -> bool:
    try:
        from dflash_mlx.model import FullContextDraftKVCache
    except ImportError:
        logger.debug("dflash_mlx.model not importable; draft KV prealloc skipped")
        return False
    except AttributeError:
        logger.debug("dflash_mlx.model.FullContextDraftKVCache missing")
        return False

    if getattr(FullContextDraftKVCache._grow, "_omlx_dflash_kv_prealloc", False):
        return True
    _backup(FullContextDraftKVCache, "_grow")
    original_grow = FullContextDraftKVCache._grow

    def _grow(
        self: Any, template_keys: mx.array, template_values: mx.array, needed: int
    ) -> None:
        cap = _capacity()
        if cap <= 0:
            return original_grow(self, template_keys, template_values, needed)
        needed = int(needed)
        if needed > cap:
            raise RuntimeError(
                f"Draft KV cache capacity exceeded: requested {needed} tokens, "
                f"reserved {cap}"
            )
        if (
            getattr(self, "keys", None) is not None
            and getattr(self, "values", None) is not None
            and int(self.keys.shape[2]) >= cap
            and int(self.values.shape[2]) >= cap
        ):
            return
        batch = int(template_keys.shape[0])
        heads = int(template_keys.shape[1])
        key_dim = int(template_keys.shape[3])
        value_dim = int(template_values.shape[3])
        new_keys = mx.zeros((batch, heads, cap, key_dim), dtype=template_keys.dtype)
        new_values = mx.zeros(
            (batch, heads, cap, value_dim), dtype=template_values.dtype
        )
        length = max(0, int(getattr(self, "_length", 0) or 0))
        if getattr(self, "keys", None) is not None and length > 0:
            copy_len = min(
                length, int(self.keys.shape[2]), int(self.values.shape[2]), cap
            )
            if copy_len > 0:
                new_keys[:, :, :copy_len, :] = self.keys[:, :, :copy_len, :]
                new_values[:, :, :copy_len, :] = self.values[:, :, :copy_len, :]
        self.keys = new_keys
        self.values = new_values
        mx.eval(self.keys, self.values)
        logger.debug(
            "Draft KV prealloc grow: %s -> capacity=%d (%.2fGB per array pair)",
            type(self).__name__,
            cap,
            (new_keys.nbytes + new_values.nbytes) / 1024**3,
        )

    _grow._omlx_dflash_kv_prealloc = True  # type: ignore[attr-defined]
    FullContextDraftKVCache._grow = _grow
    return True


def install_dflash_kv_prealloc(capacity_tokens: int) -> bool:
    """Install DFlash KV capacity patches.

    Returns True when at least one patch point was installed or was already
    installed. A non-positive capacity is a no-op.
    """
    global _CAPACITY_TOKENS
    capacity = int(capacity_tokens or 0)
    if capacity <= 0:
        return False
    if _BACKUPS and capacity != _CAPACITY_TOKENS:
        restore_dflash_kv_prealloc()
    _CAPACITY_TOKENS = capacity
    installed = False
    installed |= _install_mlx_lm_kv_cache(capacity)
    installed |= _install_dflash_codecs()
    installed |= _install_dflash_full_context_draft_cache(capacity)
    if installed:
        logger.info("DFlash KV prealloc patches armed for %d tokens", capacity)
    return installed


def restore_dflash_kv_prealloc() -> None:
    """Restore all patched methods/classes to their original objects."""
    global _CAPACITY_TOKENS
    if not _BACKUPS:
        _CAPACITY_TOKENS = 0
        return
    restored = 0
    for _key, (obj, attr, original) in list(_BACKUPS.items()):
        try:
            setattr(obj, attr, original)
            restored += 1
        except Exception as exc:
            logger.debug(
                "DFlash KV prealloc restore failed for %r.%s: %s", obj, attr, exc
            )
    _BACKUPS.clear()
    _CAPACITY_TOKENS = 0
    logger.info("DFlash KV prealloc patches restored on %d object(s)", restored)


def active_capacity_tokens() -> int:
    """Return the currently armed KV capacity, or 0 when disabled."""
    return _capacity()


__all__ = [
    "active_capacity_tokens",
    "install_dflash_kv_prealloc",
    "restore_dflash_kv_prealloc",
]
