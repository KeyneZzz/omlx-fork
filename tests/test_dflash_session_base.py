# SPDX-License-Identifier: Apache-2.0
"""Tests for cross-session system-prefix base reuse (dflash_session_base)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches import dflash_session_base as sb  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_session_base():
    sb.restore()
    yield
    sb.restore()


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__package__ = name.rpartition(".")[0]
    return mod


def _purge_dflash_modules(monkeypatch):
    """Drop leftover real dflash_mlx imports so the fakes are hermetic."""
    for name in list(sys.modules):
        if name == "dflash_mlx" or name.startswith("dflash_mlx."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _install_fake_dflash(monkeypatch, *, covers=True):
    """Install fake codecs/prefix_l1/engine.prefill modules.

    Returns a namespace with the fakes for direct manipulation.
    """
    _purge_dflash_modules(monkeypatch)
    dflash = _module("dflash_mlx")
    cache_pkg = _module("dflash_mlx.cache")
    codecs = _module("dflash_mlx.cache.codecs")
    prefix_l1 = _module("dflash_mlx.cache.prefix_l1")
    engine_pkg = _module("dflash_mlx.engine")
    prefill = _module("dflash_mlx.engine.prefill")

    def _clone_array(arr):
        cloned = mx.array(arr)
        mx.eval(cloned)
        return cloned

    class PrefixSnapshotBuilder:
        def build(self, *, token_ids, kind="prefill", **kwargs):
            return SimpleNamespace(
                token_ids=tuple(token_ids),
                fa_states=kwargs.get("fa_states", ()),
                gdn_states=kwargs.get("gdn_states"),
                target_hidden_chunks=kwargs.get("target_hidden_chunks", ()),
                target_hidden_chunk_spans=kwargs.get(
                    "target_hidden_chunk_spans", ()
                ),
                target_hidden_total_len=kwargs.get("target_hidden_total_len", 0),
                last_logits=kwargs.get("last_logits"),
                key=kwargs.get("key"),
                kind=kind,
                created_at=kwargs.get("created_at", 0.0),
                sidecar_boundary=kwargs.get("sidecar_boundary", 0),
                sidecar_gdn_states=kwargs.get("sidecar_gdn_states"),
                sidecar_last_logits=kwargs.get("sidecar_last_logits"),
            )

    class DFlashPrefixCache:
        def __init__(self):
            self.entries = {}
            self.clear_count = 0
            self.last_record = None

        def lookup(
            self,
            req_tokens,
            key,
            *,
            record=True,
            request_id=None,
            require_full_coverage=False,
        ):
            self.last_record = record
            return (0, None)

        def clear(self):
            self.clear_count += 1
            self.entries.clear()

        def resident_snapshots(self):
            return tuple(self.entries.values())

    state = SimpleNamespace(covers=covers)

    def snapshot_covers_prefix(snapshot, prefix_len):
        return bool(state.covers)

    def slice_snapshot_at_sidecar_boundary(snapshot, *, require_full_coverage=False):
        boundary = int(snapshot.sidecar_boundary)
        fields = dict(vars(snapshot))
        fields["token_ids"] = snapshot.token_ids[:boundary]
        fields["gdn_states"] = snapshot.sidecar_gdn_states
        fields["last_logits"] = snapshot.sidecar_last_logits
        fields["kind"] = "prefill"
        return SimpleNamespace(**fields)

    codecs.PrefixSnapshotBuilder = PrefixSnapshotBuilder
    codecs._clone_array = _clone_array
    prefix_l1.DFlashPrefixCache = DFlashPrefixCache
    prefix_l1.slice_snapshot_at_sidecar_boundary = slice_snapshot_at_sidecar_boundary
    prefill.snapshot_covers_prefix = snapshot_covers_prefix
    cache_pkg.codecs = codecs
    cache_pkg.prefix_l1 = prefix_l1
    engine_pkg.prefill = prefill
    dflash.cache = cache_pkg
    dflash.engine = engine_pkg
    monkeypatch.setitem(sys.modules, "dflash_mlx", dflash)
    monkeypatch.setitem(sys.modules, "dflash_mlx.cache", cache_pkg)
    monkeypatch.setitem(sys.modules, "dflash_mlx.cache.codecs", codecs)
    monkeypatch.setitem(sys.modules, "dflash_mlx.cache.prefix_l1", prefix_l1)
    monkeypatch.setitem(sys.modules, "dflash_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "dflash_mlx.engine.prefill", prefill)
    return SimpleNamespace(
        codecs=codecs,
        prefix_l1=prefix_l1,
        builder=PrefixSnapshotBuilder(),
        cache_cls=DFlashPrefixCache,
        state=state,
    )


def _fake_snapshot(token_ids, *, key="key", kind="prefill", gdn=None, logits=None):
    return SimpleNamespace(
        token_ids=tuple(token_ids),
        fa_states=(mx.ones((1, 1, 8, 4), dtype=mx.float16),) ,
        gdn_states=gdn if gdn is not None else ((mx.zeros((2, 2), dtype=mx.float32),),),
        target_hidden_chunks=(),
        target_hidden_chunk_spans=(),
        target_hidden_total_len=len(token_ids),
        last_logits=logits if logits is not None else mx.zeros((4,), dtype=mx.float32),
        key=key,
        kind=kind,
        created_at=0.0,
        sidecar_boundary=0,
        sidecar_gdn_states=None,
        sidecar_last_logits=None,
    )


# ---------------------------------------------------------------------------
# Capture (builder wrapper)
# ---------------------------------------------------------------------------


def test_capture_matching_prefill_build(monkeypatch):
    fake = _install_fake_dflash(monkeypatch)
    assert sb.install() is True

    gdn = ((mx.zeros((2, 2), dtype=mx.float32),),)
    fa = (mx.ones((1, 1, 8, 4), dtype=mx.float16),)
    logits = mx.zeros((4,), dtype=mx.float32)
    target = tuple(range(10, 20))
    sb.arm_capture(target)
    snapshot = fake.builder.build(
        token_ids=list(target), kind="prefill", gdn_states=gdn,
        fa_states=fa, last_logits=logits,
    )
    sb.disarm_capture()

    base = sb.active_base()
    assert base is not None
    assert base.token_prefix == target
    assert base.snapshot.token_ids == target
    assert base.snapshot.kind == "prefill"
    # GDN state and last_logits cloned, FA arrays aliased.
    assert base.snapshot.gdn_states[0][0] is not snapshot.gdn_states[0][0]
    assert base.snapshot.last_logits is not snapshot.last_logits
    assert base.snapshot.fa_states[0] is snapshot.fa_states[0]
    assert sb.active_base_prefix() == target


def test_capture_ignores_non_matching_or_unarmed(monkeypatch):
    fake = _install_fake_dflash(monkeypatch)
    sb.install()

    # Unarmed: never captures.
    fake.builder.build(token_ids=[1, 2, 3], kind="prefill")
    assert sb.active_base() is None

    # Armed but kind=generation: ignored.
    sb.arm_capture((1, 2, 3))
    fake.builder.build(token_ids=[1, 2, 3], kind="generation")
    assert sb.active_base() is None

    # Armed but different tokens: ignored.
    fake.builder.build(token_ids=[1, 2, 4], kind="prefill")
    assert sb.active_base() is None

    # Disarmed after a capture: later builds do not replace the base.
    sb.arm_capture((1, 2, 3))
    fake.builder.build(token_ids=[1, 2, 3], kind="prefill")
    sb.disarm_capture()
    fake.builder.build(token_ids=[1, 2, 3, 9], kind="prefill")
    assert sb.active_base().token_prefix == (1, 2, 3)


def test_capture_from_generation_snapshot_via_sidecar(monkeypatch):
    """Warmups whose prompt is all stable prefix publish no prefill snapshot;
    the generation snapshot's sidecar boundary is the capture source."""
    fake = _install_fake_dflash(monkeypatch)
    sb.install()

    target = tuple(range(10, 20))
    sidecar_gdn = ((mx.ones((2, 2), dtype=mx.float32) * 7,),)
    sidecar_logits = mx.ones((4,), dtype=mx.float32) * 5
    sb.arm_capture(target)
    fake.builder.build(
        token_ids=list(target) + [999],
        kind="generation",
        gdn_states=((mx.zeros((2, 2), dtype=mx.float32),),),
        last_logits=mx.zeros((4,), dtype=mx.float32),
        sidecar_boundary=len(target),
        sidecar_gdn_states=sidecar_gdn,
        sidecar_last_logits=sidecar_logits,
    )
    sb.disarm_capture()

    base = sb.active_base()
    assert base is not None
    assert base.token_prefix == target
    assert base.snapshot.kind == "prefill"
    assert base.snapshot.sidecar_boundary == 0
    # Base state comes from the sidecar (cloned), not the generation end.
    assert base.snapshot.gdn_states[0][0] is not sidecar_gdn[0][0]
    assert base.snapshot.last_logits is not sidecar_logits
    assert float(base.snapshot.gdn_states[0][0].max()) == 7.0
    assert float(base.snapshot.last_logits.max()) == 5.0


def test_capture_ignores_generation_without_matching_sidecar(monkeypatch):
    fake = _install_fake_dflash(monkeypatch)
    sb.install()

    target = tuple(range(10, 20))
    sb.arm_capture(target)
    # Boundary != len(target).
    fake.builder.build(
        token_ids=list(target) + [999],
        kind="generation",
        sidecar_boundary=3,
        sidecar_gdn_states=((mx.zeros((2, 2)),),),
        sidecar_last_logits=mx.zeros((4,)),
    )
    # No sidecar states.
    fake.builder.build(
        token_ids=list(target) + [999], kind="generation", sidecar_boundary=len(target)
    )
    sb.disarm_capture()
    assert sb.active_base() is None


def test_capture_replaces_single_slot(monkeypatch):
    fake = _install_fake_dflash(monkeypatch)
    sb.install()

    sb.arm_capture((1, 2, 3))
    fake.builder.build(token_ids=[1, 2, 3], kind="prefill")
    sb.disarm_capture()
    first = sb.active_base()

    sb.arm_capture((4, 5, 6))
    fake.builder.build(token_ids=[4, 5, 6], kind="prefill")
    sb.disarm_capture()
    second = sb.active_base()

    assert first is not second
    assert second.token_prefix == (4, 5, 6)


def test_restore_clears_registry_and_methods(monkeypatch):
    fake = _install_fake_dflash(monkeypatch)
    original_build = fake.codecs.PrefixSnapshotBuilder.build
    original_lookup = fake.prefix_l1.DFlashPrefixCache.lookup
    sb.install()

    sb.arm_capture((1, 2))
    fake.builder.build(token_ids=[1, 2], kind="prefill")
    assert sb.active_base() is not None

    sb.restore()
    assert sb.active_base() is None
    assert sb.active_base_prefix() is None
    assert fake.codecs.PrefixSnapshotBuilder.build is original_build
    assert fake.prefix_l1.DFlashPrefixCache.lookup is original_lookup


# ---------------------------------------------------------------------------
# Serve (lookup wrapper)
# ---------------------------------------------------------------------------


def _install_with_base(monkeypatch, prefix, *, key="key", covers=True):
    fake = _install_fake_dflash(monkeypatch, covers=covers)
    sb.install()
    sb.arm_capture(tuple(prefix))
    fake.builder.build(token_ids=list(prefix), kind="prefill", key=key)
    sb.disarm_capture()
    return fake


def test_lookup_serves_base_and_purges_lineage(monkeypatch):
    prefix = tuple(range(100, 115))
    fake = _install_with_base(monkeypatch, prefix)
    cache = fake.cache_cls()
    cache.entries["dead"] = _fake_snapshot(range(999))

    req = list(prefix) + [7, 8, 9]
    matched, snapshot = cache.lookup(req, "key")

    assert matched == len(prefix)
    assert snapshot is sb.active_base().snapshot
    # Dead lineage purged before serving.
    assert cache.clear_count == 1
    assert cache.entries == {}


def test_lookup_serves_on_probe_call(monkeypatch):
    """The L2 probe path (record=False) must serve identically."""
    prefix = tuple(range(5))
    fake = _install_with_base(monkeypatch, prefix)
    cache = fake.cache_cls()

    matched, snapshot = cache.lookup(list(prefix) + [1], "key", record=False)
    assert cache.last_record is False
    assert matched == len(prefix)
    assert snapshot is sb.active_base().snapshot


def test_lookup_exact_match_served(monkeypatch):
    prefix = tuple(range(5))
    fake = _install_with_base(monkeypatch, prefix)
    cache = fake.cache_cls()
    matched, _snapshot = cache.lookup(list(prefix), "key")
    assert matched == len(prefix)


def test_lookup_no_serve_when_prefix_diverges(monkeypatch):
    prefix = tuple(range(10, 20))
    fake = _install_with_base(monkeypatch, prefix)
    cache = fake.cache_cls()

    req = list(prefix[:5]) + [999] + list(prefix[6:])
    matched, snapshot = cache.lookup(req, "key")
    assert matched == 0
    assert snapshot is None


def test_lookup_no_serve_on_key_mismatch(monkeypatch):
    prefix = tuple(range(10, 20))
    fake = _install_with_base(monkeypatch, prefix, key="key-a")
    cache = fake.cache_cls()

    matched, snapshot = cache.lookup(list(prefix) + [1], "key-b")
    assert matched == 0
    assert snapshot is None
    assert cache.clear_count == 0


def test_lookup_no_serve_when_request_shorter_than_base(monkeypatch):
    prefix = tuple(range(10, 20))
    fake = _install_with_base(monkeypatch, prefix)
    cache = fake.cache_cls()

    matched, snapshot = cache.lookup(list(prefix[:5]), "key")
    assert matched == 0
    assert snapshot is None


def test_lookup_coverage_gate(monkeypatch):
    prefix = tuple(range(10, 20))
    fake = _install_with_base(monkeypatch, prefix, covers=False)
    cache = fake.cache_cls()
    req = list(prefix) + [1]

    matched, snapshot = cache.lookup(req, "key", require_full_coverage=True)
    assert matched == 0
    assert snapshot is None

    fake.state.covers = True
    matched, snapshot = cache.lookup(req, "key", require_full_coverage=True)
    assert matched == len(prefix)
    assert snapshot is not None


def test_lookup_hit_passes_through_untouched(monkeypatch):
    """A real lookup hit must never reach the serve logic."""
    prefix = tuple(range(5))
    fake = _install_fake_dflash(monkeypatch)

    def hit_lookup(self, req_tokens, key, **kwargs):
        return (3, "real-snapshot")

    # Replace the class method BEFORE install so the wrapper wraps the hit.
    fake.prefix_l1.DFlashPrefixCache.lookup = hit_lookup
    sb.install()

    cache = fake.cache_cls()
    matched, snapshot = cache.lookup(list(prefix) + [9], "key")
    assert matched == 3
    assert snapshot == "real-snapshot"


# ---------------------------------------------------------------------------
# Engine integration (marker detection, warmup, hysteresis)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    unk_token_id = 0

    def __init__(self, ids=(151644, 872)):
        self._ids = ids

    def convert_tokens_to_ids(self, tokens):
        assert tokens == ["<|im_start|>", "user"]
        return list(self._ids)


def _make_engine(monkeypatch, *, min_tokens=4, base_prefix=None):
    from omlx.engine.dflash import DFlashEngine

    engine = DFlashEngine.__new__(DFlashEngine)
    engine._kv_capacity_tokens = 163840
    engine._session_base_reuse = True
    engine._session_base_min_tokens = min_tokens
    engine._session_base_marker_ids = None
    engine._session_base_candidate = None
    engine._session_base_candidate_seen = 0
    engine._in_session_base_warmup = False
    engine._executor_tokenizer = _FakeTokenizer()

    monkeypatch.setattr(sb, "active_base_prefix", lambda: base_prefix)
    calls = {
        "armed": [],
        "disarmed": 0,
        "warmed": [],
        "flags_during": [],
    }

    def arm(target):
        calls["armed"].append(tuple(target))

    def disarm():
        calls["disarmed"] += 1

    def fake_stream(**kwargs):
        calls["warmed"].append(kwargs)
        calls["flags_during"].append(engine._in_session_base_warmup)
        return iter([]), None, None

    monkeypatch.setattr(sb, "arm_capture", arm)
    monkeypatch.setattr(sb, "disarm_capture", disarm)
    monkeypatch.setattr(engine, "_stream_dflash_events", fake_stream)
    return engine, calls


def test_first_user_boundary_detection(monkeypatch):
    engine, _ = _make_engine(monkeypatch)
    im_start, user = 151644, 872
    tokens = [1, 2, im_start, user, 5, 6]
    assert engine._first_user_boundary(tokens) == 2
    assert engine._first_user_boundary([1, 2, 3]) == 0
    # Marker resolution cached after first call.
    assert engine._session_base_marker_ids == (im_start, user)


def test_first_user_boundary_without_marker_support(monkeypatch):
    from omlx.engine.dflash import DFlashEngine

    engine = DFlashEngine.__new__(DFlashEngine)
    engine._session_base_marker_ids = None
    engine._executor_tokenizer = SimpleNamespace()  # no convert_tokens_to_ids
    assert engine._first_user_boundary([1, 2, 3]) == 0
    assert engine._session_base_marker_ids == (None, None)


def test_warmup_runs_when_no_base(monkeypatch):
    engine, calls = _make_engine(monkeypatch, min_tokens=2, base_prefix=None)
    im_start, user = 151644, 872
    prompt = [1, 2, im_start, user, 5]

    engine._maybe_warm_session_base(
        prompt, max_tokens=64, temperature=0.5, top_p=0.9, top_k=0, min_p=0.0
    )

    assert calls["armed"] == [(1, 2)]
    assert calls["disarmed"] == 1
    assert len(calls["warmed"]) == 1
    warm = calls["warmed"][0]
    assert warm["prompt_tokens"] == [1, 2]
    assert warm["max_tokens"] == 1
    assert warm["temperature"] == 0.5
    # The warmup suppresses the generation snapshot so the runtime's
    # prefill-snapshot publish gate stays open.
    assert warm["suppress_generation_snapshot"] is True
    assert calls["flags_during"] == [True]
    assert engine._in_session_base_warmup is False


def test_warmup_skipped_when_base_matches(monkeypatch):
    im_start, user = 151644, 872
    prompt = [1, 2, im_start, user, 5]
    engine, calls = _make_engine(
        monkeypatch, min_tokens=2, base_prefix=(1, 2)
    )
    engine._maybe_warm_session_base(
        prompt, max_tokens=8, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0
    )
    assert calls["warmed"] == []
    assert calls["armed"] == []


def test_warmup_below_threshold(monkeypatch):
    engine, calls = _make_engine(monkeypatch, min_tokens=1024, base_prefix=None)
    im_start, user = 151644, 872
    engine._maybe_warm_session_base(
        [1, 2, im_start, user, 5],
        max_tokens=8, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
    )
    assert calls["warmed"] == []


def test_warmup_replacement_hysteresis(monkeypatch):
    """A changed prefix warms only after two consecutive sightings."""
    im_start, user = 151644, 872
    prompt = [1, 2, im_start, user, 5]
    new_prefix = (1, 2)
    engine, calls = _make_engine(monkeypatch, min_tokens=2, base_prefix=(9, 9))
    kwargs = dict(max_tokens=8, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0)

    engine._maybe_warm_session_base(prompt, **kwargs)
    assert calls["warmed"] == []  # first sighting: candidate only
    assert engine._session_base_candidate == new_prefix
    assert engine._session_base_candidate_seen == 1

    engine._maybe_warm_session_base(prompt, **kwargs)
    assert len(calls["warmed"]) == 1  # second consecutive: warm
    assert engine._session_base_candidate is None

    # Alternating prefixes never confirm.
    other = [1, 3, im_start, user, 5]
    engine, calls = _make_engine(monkeypatch, min_tokens=2, base_prefix=(9, 9))
    for _ in range(6):
        engine._maybe_warm_session_base(prompt, **kwargs)
        engine._maybe_warm_session_base(other, **kwargs)
    assert calls["warmed"] == []


def test_warmup_failure_does_not_propagate(monkeypatch):
    engine, calls = _make_engine(monkeypatch, min_tokens=2, base_prefix=None)

    def boom(**kwargs):
        raise RuntimeError("prefill exploded")

    engine._stream_dflash_events = boom
    im_start, user = 151644, 872
    engine._maybe_warm_session_base(
        [1, 2, im_start, user, 5],
        max_tokens=8, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
    )
    # Flag and capture state restored despite the failure.
    assert engine._in_session_base_warmup is False
    assert calls["disarmed"] == 1
