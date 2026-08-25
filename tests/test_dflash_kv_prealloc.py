# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlash fixed-capacity KV cache preallocation patches."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches import dflash_kv_prealloc as prealloc  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_prealloc():
    prealloc.restore_dflash_kv_prealloc()
    yield
    prealloc.restore_dflash_kv_prealloc()


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__package__ = name.rpartition(".")[0]
    return mod


def _purge_dflash_modules(monkeypatch):
    """Drop leftover real dflash_mlx imports so the fakes are hermetic.

    Earlier test modules (e.g. test_dflash_engine) import the real runtime,
    leaving real submodules in sys.modules. The patch's
    ``from dflash_mlx.cache import prefix_l1`` would then resolve a real
    module alongside the fake codecs and crash on symbols the fake lacks.
    """
    for name in list(sys.modules):
        if name == "dflash_mlx" or name.startswith("dflash_mlx."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _install_fake_mlx_lm(monkeypatch):
    mlx_lm = _module("mlx_lm")
    models = _module("mlx_lm.models")
    cache_mod = _module("mlx_lm.models.cache")

    class KVCache:
        def __init__(self):
            self.keys = None
            self.values = None
            self.offset = 0

        def update_and_fetch(self, keys, values):
            self.keys = keys
            self.values = values
            self.offset = int(keys.shape[2])
            return keys, values

    cache_mod.KVCache = KVCache
    models.cache = cache_mod
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    return KVCache


def _install_fake_dflash_codecs(monkeypatch):
    _purge_dflash_modules(monkeypatch)
    dflash = _module("dflash_mlx")
    cache_pkg = _module("dflash_mlx.cache")
    codecs = _module("dflash_mlx.cache.codecs")

    class KVCache:
        def __init__(self):
            self.keys = None
            self.values = None
            self.offset = 0

        @property
        def state(self):
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    class RotatingKVCache:
        def __init__(self, max_size=0, keep=0):
            self.max_size = max_size
            self.keep = keep
            self.keys = None
            self.values = None
            self.offset = 0
            self._idx = 0

    class RecurrentRollbackCache:
        def __init__(self, size=0, conv_kernel_size=0):
            self.cache = [None] * size
            self.conv_kernel_size = conv_kernel_size

    def serialize_target_cache(_target_cache, *, clone=True):
        return "stock-serialize", clone

    def hydrate_target_cache(_snapshot, _template_cache):
        return ["stock-hydrate"]

    def _validate_snapshot_cache_prefix_len(_states, *, prefix_len):
        assert prefix_len >= 0

    codecs.KVCache = KVCache
    codecs.RotatingKVCache = RotatingKVCache
    codecs.RecurrentRollbackCache = RecurrentRollbackCache
    codecs.serialize_target_cache = serialize_target_cache
    codecs.hydrate_target_cache = hydrate_target_cache
    codecs._validate_snapshot_cache_prefix_len = _validate_snapshot_cache_prefix_len
    codecs._clone_array = lambda arr: arr
    cache_pkg.codecs = codecs
    dflash.cache = cache_pkg
    monkeypatch.setitem(sys.modules, "dflash_mlx", dflash)
    monkeypatch.setitem(sys.modules, "dflash_mlx.cache", cache_pkg)
    monkeypatch.setitem(sys.modules, "dflash_mlx.cache.codecs", codecs)
    return codecs


def _install_fake_dflash_model(monkeypatch):
    dflash = sys.modules.get("dflash_mlx") or _module("dflash_mlx")
    model_mod = _module("dflash_mlx.model")

    class FullContextDraftKVCache:
        def __init__(self):
            self.keys = None
            self.values = None
            self._length = 0

        def _grow(self, template_keys, template_values, needed):
            self.keys = template_keys
            self.values = template_values
            self._length = int(needed)

    model_mod.FullContextDraftKVCache = FullContextDraftKVCache
    dflash.model = model_mod
    monkeypatch.setitem(sys.modules, "dflash_mlx", dflash)
    monkeypatch.setitem(sys.modules, "dflash_mlx.model", model_mod)
    return FullContextDraftKVCache


def test_mlx_lm_kv_cache_allocates_once_to_reserved_capacity(monkeypatch):
    kv_cache_cls = _install_fake_mlx_lm(monkeypatch)

    assert prealloc.install_dflash_kv_prealloc(8) is True
    cache = kv_cache_cls()
    keys = mx.ones((1, 2, 2, 4), dtype=mx.float16)
    values = mx.ones((1, 2, 2, 4), dtype=mx.float16)

    k, v = cache.update_and_fetch(keys, values)
    assert cache.offset == 2
    assert cache.keys.shape == (1, 2, 8, 4)
    assert cache.values.shape == (1, 2, 8, 4)
    assert k.shape == (1, 2, 2, 4)
    assert v.shape == (1, 2, 2, 4)

    cache.update_and_fetch(keys, values)
    assert cache.offset == 4
    assert cache.keys.shape == (1, 2, 8, 4)


def test_mlx_lm_kv_cache_rejects_capacity_overflow(monkeypatch):
    kv_cache_cls = _install_fake_mlx_lm(monkeypatch)
    prealloc.install_dflash_kv_prealloc(3)
    cache = kv_cache_cls()
    keys = mx.ones((1, 1, 2, 2), dtype=mx.float16)
    values = mx.ones((1, 1, 2, 2), dtype=mx.float16)

    cache.update_and_fetch(keys, values)
    with pytest.raises(RuntimeError, match="KV cache capacity exceeded"):
        cache.update_and_fetch(keys, values)


def test_restore_reverts_fake_mlx_lm_patch(monkeypatch):
    kv_cache_cls = _install_fake_mlx_lm(monkeypatch)
    original = kv_cache_cls.update_and_fetch

    prealloc.install_dflash_kv_prealloc(8)
    assert kv_cache_cls.update_and_fetch is not original
    prealloc.restore_dflash_kv_prealloc()
    assert kv_cache_cls.update_and_fetch is original
    assert prealloc.active_capacity_tokens() == 0


def test_codecs_keep_physical_capacity_for_kv_snapshots(monkeypatch):
    _install_fake_mlx_lm(monkeypatch)
    codecs = _install_fake_dflash_codecs(monkeypatch)

    prealloc.install_dflash_kv_prealloc(8)
    entry = codecs.KVCache()
    entry.keys = mx.ones((1, 2, 8, 4), dtype=mx.float16)
    entry.values = mx.ones((1, 2, 8, 4), dtype=mx.float16)
    entry.offset = 2

    fa_states, gdn_states = codecs.serialize_target_cache([entry], clone=False)
    assert fa_states[0][0].shape == (1, 2, 8, 4)
    assert fa_states[0][2] == 2
    assert gdn_states == (None,)

    snapshot = SimpleNamespace(
        fa_states=fa_states,
        gdn_states=gdn_states,
        prefix_len=2,
    )
    hydrated = codecs.hydrate_target_cache(snapshot, [codecs.KVCache()])
    assert hydrated[0].keys is fa_states[0][0]
    assert hydrated[0].values is fa_states[0][1]
    assert hydrated[0].offset == 2
    assert hydrated[0].keys.shape[2] == 8


def test_full_context_draft_cache_grows_to_reserved_capacity(monkeypatch):
    draft_cache_cls = _install_fake_dflash_model(monkeypatch)
    prealloc.install_dflash_kv_prealloc(6)

    cache = draft_cache_cls()
    keys = mx.ones((1, 2, 2, 3), dtype=mx.float16)
    values = mx.ones((1, 2, 2, 5), dtype=mx.float16)

    cache._grow(keys, values, 2)
    assert cache.keys.shape == (1, 2, 6, 3)
    assert cache.values.shape == (1, 2, 6, 5)

    with pytest.raises(RuntimeError, match="Draft KV cache capacity exceeded"):
        cache._grow(keys, values, 7)
