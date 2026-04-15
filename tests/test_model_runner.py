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

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_LAYERS = 2
N_KV_HEADS = 4
HEAD_DIM = 8
VOCAB_SIZE = 32
HIDDEN_SIZE = 32
NUM_ATTENTION_HEADS = 4
# Use block_size=1 so 1 block = 1 token — same granularity as the old slot model,
# making test assertions straightforward.
NUM_BLOCKS = 64
BLOCK_SIZE = 1


# ── Mock model ────────────────────────────────────────────────────────────────


class MockModel:
    """Minimal model that simulates mlx-lm forward pass without KV interaction.

    Returns zero logits so that argmax always selects token 0.
    The cache is not exercised here — KV mechanics are tested separately in
    test_memory_pool.py.
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
        self._vocab_size = vocab_size
        self._call_count = 0

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        B, S = inputs.shape
        self._call_count += 1
        return mx.zeros((B, S, self._vocab_size))


def _make_args(**kwargs):
    import types

    return types.SimpleNamespace(**kwargs)


def _make_runner(
    num_blocks: int = NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    num_layers: int = NUM_LAYERS,
    n_kv_heads: int = N_KV_HEADS,
    head_dim: int = HEAD_DIM,
    vocab_size: int = VOCAB_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    num_attention_heads: int = NUM_ATTENTION_HEADS,
) -> tuple[ModelRunner, MockModel]:
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
        model,
        tokenizer=None,
        config=config,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=mx.float32,
    )
    return runner, model


# ── TestModelConfig ───────────────────────────────────────────────────────────


class TestModelConfig:
    def test_from_model_args(self):
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
        model = MockModel()
        del model.args.head_dim
        config = _extract_model_config(model)
        assert config.head_dim == HIDDEN_SIZE // NUM_ATTENTION_HEADS

    def test_config_dict_fallback(self):
        model = MockModel()
        del model.args.vocab_size
        config = _extract_model_config(model, config_dict={"vocab_size": 128})
        assert config.vocab_size == 128

    def test_num_layers_from_model_layers(self):
        model = MockModel(num_layers=4)
        model.args.num_hidden_layers = 99
        config = _extract_model_config(model)
        assert config.num_layers == 4

    def test_kv_heads_defaults_to_attention_heads(self):
        model = MockModel()
        del model.args.num_key_value_heads
        config = _extract_model_config(model)
        assert config.n_kv_heads == NUM_ATTENTION_HEADS

    def test_sliding_window(self):
        model = MockModel()
        model.args.sliding_window = 512
        config = _extract_model_config(model)
        assert config.sliding_window == 512


# ── TestRequestState ──────────────────────────────────────────────────────────


class TestRequestState:
    def test_seq_len(self):
        state = RequestState(
            request_id="r1",
            prompt_tokens=[1, 2, 3],
            output_tokens=[4, 5],
            all_block_ids=[10, 11, 12, 13, 14],
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
            all_block_ids=[10, 11, 12],
            radix_node=None,
            prefix_len=0,
            kv_caches=[],
        )
        assert state.total_cached_len == 3


# ── TestForwardPrefill ────────────────────────────────────────────────────────


class TestForwardPrefill:
    def test_cold_start(self):
        runner, model = _make_runner()
        prompt = [10, 20, 30, 40]

        state, result = runner.forward_prefill("req-1", prompt)

        assert result.forward_mode == ForwardMode.PREFILL
        assert result.logits.shape == (VOCAB_SIZE,)
        assert state.prompt_tokens == prompt
        assert state.output_tokens == []
        assert state.prefix_len == 0
        assert len(state.all_block_ids) == 4  # block_size=1 → 1 block per token
        assert len(state.kv_caches) == NUM_LAYERS
        assert model._call_count == 1

    def test_block_allocation(self):
        runner, _ = _make_runner(num_blocks=32)
        prompt = [1, 2, 3, 4, 5]
        avail_before = runner._allocator.available_blocks

        state, _ = runner.forward_prefill("req-1", prompt)

        assert runner._allocator.available_blocks == avail_before - 5
        assert len(state.all_block_ids) == 5

    def test_radix_insert(self):
        runner, _ = _make_runner()
        prompt = [10, 20, 30]
        runner.forward_prefill("req-1", prompt)

        match = runner._radix_cache.match_prefix(prompt)
        assert len(match.device_indices) == 3

    def test_radix_lock(self):
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])

        assert state.radix_node is not None
        assert state.radix_node.lock_ref > 0

    def test_active_request_tracking(self):
        runner, _ = _make_runner()
        runner.forward_prefill("req-1", [1, 2, 3])
        assert "req-1" in runner._active_requests

    def test_full_cache_hit(self):
        runner, model = _make_runner()
        prompt = [10, 20, 30]

        state1, _ = runner.forward_prefill("req-1", prompt)
        runner.finish_request(state1)

        state2, result = runner.forward_prefill("req-2", prompt)

        assert result.forward_mode == ForwardMode.PREFILL
        assert result.logits.shape == (VOCAB_SIZE,)
        assert len(state2.all_block_ids) == 3
        assert model._call_count == 2


# ── TestForwardDecode ─────────────────────────────────────────────────────────


class TestForwardDecode:
    def test_single_decode_step(self):
        runner, model = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])
        blocks_before = len(state.all_block_ids)
        avail_before = runner._allocator.available_blocks

        result = runner.forward_decode(state, token_id=42)

        assert result.forward_mode == ForwardMode.DECODE
        assert result.logits.shape == (VOCAB_SIZE,)
        assert len(state.all_block_ids) == blocks_before + 1  # block_size=1
        assert runner._allocator.available_blocks == avail_before - 1
        assert state.output_tokens == [42]
        assert model._call_count == 2

    def test_multiple_decode_steps(self):
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20])

        for token_id in [30, 40, 50]:
            runner.forward_decode(state, token_id)

        assert state.output_tokens == [30, 40, 50]
        assert len(state.all_block_ids) == 5  # block_size=1 → 1 block per token
        assert state.seq_len == 5

    def test_radix_insert_after_decode(self):
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20])
        runner.forward_decode(state, token_id=30)

        match = runner._radix_cache.match_prefix([10, 20, 30])
        assert len(match.device_indices) == 3

    def test_kv_cache_offset_grows(self):
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])
        runner.forward_decode(state, token_id=40)
        assert state.seq_len == 4


# ── TestFinishRequest ─────────────────────────────────────────────────────────


class TestFinishRequest:
    def test_unlock(self):
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])
        node = state.radix_node
        assert node.lock_ref > 0

        runner.finish_request(state)
        assert node.lock_ref == 0

    def test_removes_from_active(self):
        runner, _ = _make_runner()
        state, _ = runner.forward_prefill("req-1", [10, 20, 30])
        assert "req-1" in runner._active_requests

        runner.finish_request(state)
        assert "req-1" not in runner._active_requests

    def test_blocks_remain_in_radix(self):
        runner, _ = _make_runner()
        prompt = [10, 20, 30]
        state, _ = runner.forward_prefill("req-1", prompt)
        runner.finish_request(state)

        match = runner._radix_cache.match_prefix(prompt)
        assert len(match.device_indices) == 3


# ── TestGenerate ──────────────────────────────────────────────────────────────


class TestGenerate:
    def test_basic_generate(self):
        runner, model = _make_runner()
        tokens = runner.generate([10, 20, 30], max_tokens=3)

        assert tokens == [0, 0, 0]
        assert model._call_count == 4  # 1 prefill + 3 decode

    def test_stop_token(self):
        runner, _ = _make_runner()
        tokens = runner.generate([10, 20, 30], max_tokens=100, stop_tokens={0})
        assert tokens == [0]

    def test_custom_sampler(self):
        runner, _ = _make_runner()
        tokens = runner.generate([10, 20, 30], max_tokens=2, sampler=lambda _: 5)
        assert tokens == [5, 5]

    def test_cleanup_after_generate(self):
        runner, _ = _make_runner()
        runner.generate([10, 20, 30], max_tokens=2)
        assert len(runner._active_requests) == 0

    def test_blocks_allocated_correctly(self):
        runner, _ = _make_runner(num_blocks=100)
        avail_before = runner._allocator.available_blocks

        runner.generate([10, 20, 30], max_tokens=5)

        used = avail_before - runner._allocator.available_blocks
        assert used == 8  # 3 prompt + 5 generated (block_size=1)


# ── TestPrefixReuse ───────────────────────────────────────────────────────────


class TestPrefixReuse:
    def test_shared_prefix(self):
        runner, model = _make_runner(num_blocks=100)

        state1, _ = runner.forward_prefill("req-1", [10, 20, 30, 40, 50])
        runner.finish_request(state1)

        avail_before = runner._allocator.available_blocks

        state2, _ = runner.forward_prefill("req-2", [10, 20, 30, 60, 70])

        assert state2.prefix_len == 3  # [10,20,30] reused (block_size=1 → 3 blocks)
        assert runner._allocator.available_blocks == avail_before - 2
        assert len(state2.all_block_ids) == 5

        runner.finish_request(state2)

    def test_full_prefix_reuse(self):
        runner, _ = _make_runner()
        prompt = [10, 20, 30]

        state1, _ = runner.forward_prefill("req-1", prompt)
        runner.finish_request(state1)

        avail_before = runner._allocator.available_blocks
        state2, _ = runner.forward_prefill("req-2", prompt)

        assert runner._allocator.available_blocks == avail_before
        assert len(state2.all_block_ids) == 3

        runner.finish_request(state2)

    def test_prefix_reuse_with_decode(self):
        runner, _ = _make_runner(num_blocks=100)

        state1, _ = runner.forward_prefill("req-1", [10, 20, 30])
        runner.forward_decode(state1, 40)
        runner.forward_decode(state1, 50)
        runner.finish_request(state1)

        state2, _ = runner.forward_prefill("req-2", [10, 20, 30, 60])

        assert state2.prefix_len == 3
        assert len(state2.all_block_ids) == 4

        runner.finish_request(state2)

    def test_no_block_leaks(self):
        runner, _ = _make_runner(num_blocks=200)
        avail_start = runner._allocator.available_blocks

        runner.generate([1, 2, 3], max_tokens=2)
        runner.generate([4, 5, 6], max_tokens=2)

        used = avail_start - runner._allocator.available_blocks
        assert used == 10  # (3+2) + (3+2) = 10 blocks (block_size=1)

        from sglang_mlx.srt.mem_cache.base_prefix_cache import EvictParams

        runner._radix_cache.evict(EvictParams(num_tokens=200))
        assert runner._allocator.available_blocks == avail_start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
