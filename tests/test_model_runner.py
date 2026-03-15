"""Tests for the model runner."""

from __future__ import annotations

import mlx.core as mx
import pytest

from sglang_mlx.model_runner import (
    ForwardMode,
    ModelConfig,
    ModelRunner,
    RequestState,
    _extract_model_config,
)

# ---- Constants for mock model ----

NUM_LAYERS = 2
N_KV_HEADS = 4
HEAD_DIM = 8
VOCAB_SIZE = 32
HIDDEN_SIZE = 32
NUM_ATTENTION_HEADS = 4
POOL_SIZE = 64


# ---- Mock model ----


class MockModel:
    """Minimal model that simulates mlx-lm cache interaction.

    On each ``__call__``, iterates over the cache list and calls
    ``update_and_fetch`` with zero-valued KV tensors, just like a real
    model's attention layers would.  Returns zero logits so that argmax
    always selects token 0.
    """

    def __init__(
        self,
        num_layers: int = NUM_LAYERS,
        n_kv_heads: int = N_KV_HEADS,
        head_dim: int = HEAD_DIM,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_attention_heads: int = NUM_ATTENTION_HEADS,
    ):
        self.args = _make_args(
            num_hidden_layers=num_layers,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.layers = [None] * num_layers
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._vocab_size = vocab_size
        self._call_count = 0

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        B, S = inputs.shape
        self._call_count += 1

        if cache is not None:
            for c in cache:
                keys = mx.zeros((B, self._n_kv_heads, S, self._head_dim))
                values = mx.zeros((B, self._n_kv_heads, S, self._head_dim))
                c.update_and_fetch(keys, values)

        return mx.zeros((B, S, self._vocab_size))


def _make_args(**kwargs):
    """Create a simple namespace-like object from kwargs."""
    import types

    return types.SimpleNamespace(**kwargs)


def _make_runner(
    pool_size: int = POOL_SIZE,
    num_layers: int = NUM_LAYERS,
    n_kv_heads: int = N_KV_HEADS,
    head_dim: int = HEAD_DIM,
    vocab_size: int = VOCAB_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    num_attention_heads: int = NUM_ATTENTION_HEADS,
) -> tuple[ModelRunner, MockModel]:
    """Build a ModelRunner wired to a MockModel."""
    model = MockModel(
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    config = ModelConfig(
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    runner = ModelRunner(
        model, tokenizer=None, config=config, pool_size=pool_size, dtype=mx.float32
    )
    return runner, model


# ---- TestModelConfig ----


class TestModelConfig:
    """Tests for _extract_model_config."""

    def test_from_model_args(self):
        """Extract config from model.args (standard mlx-lm path)."""
        model = MockModel()
        config = _extract_model_config(model)
        assert config.num_layers == NUM_LAYERS
        assert config.n_kv_heads == N_KV_HEADS
        assert config.head_dim == HEAD_DIM
        assert config.vocab_size == VOCAB_SIZE
        assert config.hidden_size == HIDDEN_SIZE
        assert config.num_attention_heads == NUM_ATTENTION_HEADS
        assert config.sliding_window is None

    def test_head_dim_fallback(self):
        """head_dim computed from hidden_size // num_attention_heads."""
        model = MockModel()
        # Remove explicit head_dim from args
        del model.args.head_dim
        config = _extract_model_config(model)
        assert config.head_dim == HIDDEN_SIZE // NUM_ATTENTION_HEADS

    def test_config_dict_fallback(self):
        """Falls back to config_dict when model.args is missing fields."""
        model = MockModel()
        del model.args.vocab_size
        config = _extract_model_config(model, config_dict={"vocab_size": 128})
        assert config.vocab_size == 128

    def test_num_layers_from_model_layers(self):
        """num_layers from len(model.layers) takes priority."""
        model = MockModel(num_layers=4)
        model.args.num_hidden_layers = 99  # should be ignored
        config = _extract_model_config(model)
        assert config.num_layers == 4

    def test_kv_heads_defaults_to_attention_heads(self):
        """When num_key_value_heads is absent, use num_attention_heads (MHA)."""
        model = MockModel()
        del model.args.num_key_value_heads
        config = _extract_model_config(model)
        assert config.n_kv_heads == NUM_ATTENTION_HEADS

    def test_sliding_window(self):
        """sliding_window extracted when present."""
        model = MockModel()
        model.args.sliding_window = 512
        config = _extract_model_config(model)
        assert config.sliding_window == 512


# ---- TestRequestState ----


class TestRequestState:
    """Tests for RequestState properties."""

    def test_seq_len(self):
        state = RequestState(
            request_id="r1",
            prompt_tokens=[1, 2, 3],
            output_tokens=[4, 5],
            all_slot_indices=[10, 11, 12, 13, 14],
            radix_node=None,
            prefix_len=0,
            kv_caches=[],
        )
        assert state.seq_len == 5

    def test_total_cached_len(self):
        state = RequestState(
            request_id="r1",
            prompt_tokens=[1, 2, 3],
            output_tokens=[],
            all_slot_indices=[10, 11, 12],
            radix_node=None,
            prefix_len=0,
            kv_caches=[],
        )
        assert state.total_cached_len == 3


# ---- TestForwardPrefill ----


class TestForwardPrefill:
    """Tests for forward_prefill."""

    def test_cold_start(self):
        """No prefix match — all tokens are new."""
        runner, model = _make_runner()
        prompt = [10, 20, 30, 40]

        state, result = runner.forward_prefill("req-1", prompt)

        assert result.forward_mode == ForwardMode.PREFILL
        assert result.logits.shape == (VOCAB_SIZE,)
        assert state.prompt_tokens == prompt
        assert state.output_tokens == []
        assert state.prefix_len == 0
        assert len(state.all_slot_indices) == 4
        assert len(state.kv_caches) == NUM_LAYERS
        assert model._call_count == 1

    def test_slot_allocation(self):
        """Slots are allocated from the pool and tracked in state."""
        runner, _ = _make_runner(pool_size=32)
        prompt = [1, 2, 3, 4, 5]
        avail_before = runner._allocator.available_size

        state, _ = runner.forward_prefill("req-1", prompt)

        # 5 slots allocated
        assert runner._allocator.available_size == avail_before - 5
        # All slot indices are valid (>= 1)
        assert all(s >= 1 for s in state.all_slot_indices)

    def test_radix_insert(self):
        """Prompt tokens are inserted into the radix cache after prefill."""
        runner, _ = _make_runner()
        prompt = [10, 20, 30]
        runner.forward_prefill("req-1", prompt)

        # Matching the same prefix should return the slots
        match = runner._radix_cache.match_prefix(prompt)
        assert len(match.device_indices) == 3

    def test_radix_lock(self):
        """The radix node is locked after prefill."""
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])

        assert state.radix_node is not None
        assert state.radix_node.lock_ref > 0

    def test_active_request_tracking(self):
        """Request is tracked in active_requests."""
        runner, _ = _make_runner()
        runner.forward_prefill("req-1", [1, 2, 3])
        assert "req-1" in runner._active_requests

    def test_full_cache_hit(self):
        """Full cache hit recomputes last token for logits."""
        runner, model = _make_runner()
        prompt = [10, 20, 30]

        # First request inserts prompt into cache
        state1, _ = runner.forward_prefill("req-1", prompt)
        runner.finish_request(state1)

        # Second request with same prompt — full cache hit
        state2, result = runner.forward_prefill("req-2", prompt)

        assert result.forward_mode == ForwardMode.PREFILL
        assert result.logits.shape == (VOCAB_SIZE,)
        assert len(state2.all_slot_indices) == 3
        # Model was called twice total (once for each prefill)
        assert model._call_count == 2


# ---- TestForwardDecode ----


class TestForwardDecode:
    """Tests for forward_decode."""

    def test_single_decode_step(self):
        """One decode step allocates 1 slot and produces logits."""
        runner, model = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])
        slots_before = len(state.all_slot_indices)
        avail_before = runner._allocator.available_size

        result = runner.forward_decode(state, token_id=42)

        assert result.forward_mode == ForwardMode.DECODE
        assert result.logits.shape == (VOCAB_SIZE,)
        assert len(state.all_slot_indices) == slots_before + 1
        assert runner._allocator.available_size == avail_before - 1
        assert state.output_tokens == [42]
        assert model._call_count == 2  # 1 prefill + 1 decode

    def test_multiple_decode_steps(self):
        """Multiple decode steps accumulate tokens and slots."""
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20])

        for token_id in [30, 40, 50]:
            runner.forward_decode(state, token_id)

        assert state.output_tokens == [30, 40, 50]
        assert len(state.all_slot_indices) == 5  # 2 prompt + 3 decode
        assert state.seq_len == 5

    def test_radix_insert_after_decode(self):
        """Each decode step inserts the extended sequence into radix cache."""
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20])
        runner.forward_decode(state, token_id=30)

        # The cache should now have [10, 20, 30]
        match = runner._radix_cache.match_prefix([10, 20, 30])
        assert len(match.device_indices) == 3

    def test_kv_cache_offset_grows(self):
        """Each decode step should set offset = total cached tokens."""
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])

        # After prefill, decode caches should have offset = 3
        # (set_indices is called in forward_decode before model call)
        runner.forward_decode(state, token_id=40)
        # After decode, the next set_indices would set offset = 4
        # Verify state tracks this correctly
        assert state.total_cached_len == 4


# ---- TestFinishRequest ----


class TestFinishRequest:
    """Tests for finish_request."""

    def test_unlock(self):
        """Finishing a request unlocks the radix node."""
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])
        node = state.radix_node
        assert node.lock_ref > 0

        runner.finish_request(state)
        assert node.lock_ref == 0

    def test_removes_from_active(self):
        """Finishing removes the request from active tracking."""
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])
        assert "req-1" in runner._active_requests

        runner.finish_request(state)
        assert "req-1" not in runner._active_requests

    def test_slots_remain_in_radix(self):
        """After finish, slots remain in the radix cache for future reuse."""
        runner, _ = _make_runner()
        prompt = [10, 20, 30]
        state, _ = runner.forward_prefill("req-1", prompt)
        runner.finish_request(state)

        # Prefix should still be matchable
        match = runner._radix_cache.match_prefix(prompt)
        assert len(match.device_indices) == 3


# ---- TestGenerate ----


class TestGenerate:
    """Tests for the generate convenience method."""

    def test_basic_generate(self):
        """Generate with argmax sampler (all zeros → always token 0)."""
        runner, model = _make_runner()
        tokens = runner.generate([10, 20, 30], max_tokens=3)

        assert tokens == [0, 0, 0]
        # 1 prefill + 3 decode calls (last decode happens before checking max)
        assert model._call_count == 4

    def test_stop_token(self):
        """Generation stops when a stop token is produced."""
        runner, _ = _make_runner()
        # argmax always returns 0, so stop_tokens={0} stops immediately
        tokens = runner.generate([10, 20, 30], max_tokens=100, stop_tokens={0})

        assert tokens == [0]

    def test_custom_sampler(self):
        """Custom sampler overrides argmax."""
        runner, _ = _make_runner()
        # Always return token 5
        tokens = runner.generate(
            [10, 20, 30],
            max_tokens=2,
            sampler=lambda logits: 5,
        )
        assert tokens == [5, 5]

    def test_cleanup_after_generate(self):
        """Request is cleaned up after generate completes."""
        runner, _ = _make_runner()
        runner.generate([10, 20, 30], max_tokens=2)

        assert len(runner._active_requests) == 0

    def test_slots_allocated_correctly(self):
        """Total slots = prompt + generated tokens."""
        runner, _ = _make_runner(pool_size=100)
        avail_before = runner._allocator.available_size

        runner.generate([10, 20, 30], max_tokens=5)

        # 3 prompt + 5 generated = 8 slots (in radix cache, not freed)
        used = avail_before - runner._allocator.available_size
        assert used == 8


# ---- TestPrefixReuse ----


class TestPrefixReuse:
    """Tests for prefix reuse across requests."""

    def test_shared_prefix(self):
        """Two requests sharing a prefix reuse radix cache slots."""
        runner, model = _make_runner(pool_size=100)

        # Request 1: prompt [10, 20, 30, 40, 50]
        state1, _ = runner.forward_prefill("req-1", [10, 20, 30, 40, 50])
        runner.finish_request(state1)

        avail_before = runner._allocator.available_size

        # Request 2: shares prefix [10, 20, 30], new tokens [60, 70]
        state2, _ = runner.forward_prefill("req-2", [10, 20, 30, 60, 70])

        # Should have matched 3 tokens from radix cache
        assert state2.prefix_len == 3
        # Only 2 new slots allocated (for tokens 60, 70)
        assert runner._allocator.available_size == avail_before - 2
        # Total slots = 3 (reused) + 2 (new) = 5
        assert len(state2.all_slot_indices) == 5

        runner.finish_request(state2)

    def test_full_prefix_reuse(self):
        """Second request with identical prompt reuses everything."""
        runner, _ = _make_runner()
        prompt = [10, 20, 30]

        state1, _ = runner.forward_prefill("req-1", prompt)
        runner.finish_request(state1)

        avail_before = runner._allocator.available_size
        state2, _ = runner.forward_prefill("req-2", prompt)

        # Full hit: no new slots allocated (last token reuses existing slot)
        assert runner._allocator.available_size == avail_before
        assert len(state2.all_slot_indices) == 3

        runner.finish_request(state2)

    def test_prefix_reuse_with_decode(self):
        """Prefix reuse works after first request has decoded tokens."""
        runner, _ = _make_runner(pool_size=100)

        # Request 1: prefill + decode
        state1, _ = runner.forward_prefill("req-1", [10, 20, 30])
        runner.forward_decode(state1, 40)
        runner.forward_decode(state1, 50)
        runner.finish_request(state1)

        # Request 2: same prefix, different suffix
        state2, _ = runner.forward_prefill("req-2", [10, 20, 30, 60])

        # Should reuse [10, 20, 30] from cache
        assert state2.prefix_len == 3
        assert len(state2.all_slot_indices) == 4

        runner.finish_request(state2)

    def test_no_slot_leaks(self):
        """Multiple generate calls don't leak pool slots."""
        runner, _ = _make_runner(pool_size=200)
        avail_start = runner._allocator.available_size

        # Run two generate calls
        runner.generate([1, 2, 3], max_tokens=2)  # 3 + 2 = 5 slots
        runner.generate([4, 5, 6], max_tokens=2)  # 3 + 2 = 5 slots

        # Slots are in the radix cache, not freed
        used = avail_start - runner._allocator.available_size
        assert used == 10

        # But eviction can reclaim them
        runner._radix_cache.evict(
            __import__(
                "sglang_mlx.srt.mem_cache.base_prefix_cache", fromlist=["EvictParams"]
            ).EvictParams(num_tokens=200)
        )
        assert runner._allocator.available_size == avail_start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
