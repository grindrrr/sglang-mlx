"""End-to-end tests for ModelRunner with a paged-attention-aware model.

Uses TinyPagedModel — a minimal 1-layer transformer that actually calls
store() + compute_paged_attn() during decode and update_and_fetch() during
prefill, so the full Metal pipeline is exercised.

Key tests:
- Prefill logits match naive full-sequence attention
- Decode logits match naive attention over the same K/V context
- Prefix-reuse produces numerically identical output to a cold-start run

All tests require the compiled Metal extension and are skipped without it.
"""

from __future__ import annotations

import types

import mlx.core as mx
import numpy as np
import pytest

from sglang_mlx.model_runner import ModelConfig, ModelRunner
from sglang_mlx.srt.mem_cache.paged_pool import PagedKVCache

try:
    from sglang_mlx import _ext  # noqa: F401

    HAS_EXT = True
except ImportError:
    HAS_EXT = False

needs_ext = pytest.mark.skipif(not HAS_EXT, reason="Metal extension (_ext) not built")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _naive_sdpa(q, k, v, scale):
    """Full scaled dot-product attention (no causal mask).

    q: (B, n_heads, S_q, D)
    k: (B, n_kv_heads, S_k, D)
    v: (B, n_kv_heads, S_k, D)
    Returns: (B, n_heads, S_q, D)
    """
    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    if n_heads != n_kv_heads:
        group = n_heads // n_kv_heads
        k = mx.repeat(k, group, axis=1)
        v = mx.repeat(v, group, axis=1)
    # (B, n_heads, S_q, S_k)
    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return mx.matmul(weights, v)


# ── TinyPagedModel ────────────────────────────────────────────────────────────


class TinyPagedModel:
    """Minimal 1-layer model that exercises the full paged attention pipeline.

    Properties:
    - Deterministic embedding table (seeded, no positional encoding).
    - Q = K = V = linear slice of the embedding (no learned projection weights).
    - Prefill (S > 1): update_and_fetch gathers prefix + new K/V, then naive SDPA.
    - Decode (S == 1): store() + compute_paged_attn() via Metal kernel.
    - No cache (cache=None): naive SDPA over full sequence (reference path).

    Because K/V are pure functions of token IDs (no positional encoding,
    no context-dependent computation), the stored blocks contain exactly the
    same values as a fresh recomputation — making prefix reuse comparisons valid.

    Uses head_dim=32, block_size=8 — the minimum sizes supported by the
    Metal paged attention kernel.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        n_heads: int = 2,
        n_kv_heads: int = 2,
        head_dim: int = 32,  # minimum kernel-supported head size
    ):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.vocab_size = vocab_size

        D = n_heads * head_dim
        mx.random.seed(42)
        # Use float16 to match the pool dtype; K/V stored in cache are float16
        self.embed = (mx.random.normal((vocab_size, D)) * 0.1).astype(mx.float16)
        self.Wout = (mx.random.normal((D, vocab_size)) * 0.1).astype(mx.float16)

        self._args = types.SimpleNamespace(
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            hidden_size=D,
            num_attention_heads=n_heads,
        )
        self.layers = [None]  # 1 layer — ModelRunner reads len(model.layers)

    @property
    def args(self):
        return self._args

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        B, S = input_ids.shape
        D = self.n_heads * self.head_dim
        KVD = self.n_kv_heads * self.head_dim

        x = self.embed[input_ids]  # (B, S, D)

        # Q/K/V: simple reshape of the embedding (no learned projection)
        # Q: (B, n_heads,    S, head_dim)
        # K: (B, n_kv_heads, S, head_dim)  [first KVD dims of embedding]
        q = x.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (
            x[..., :KVD]
            .reshape(B, S, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            x[..., :KVD]
            .reshape(B, S, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        if cache is None:
            # Reference path: naive full-sequence attention
            attn = _naive_sdpa(q, k, v, self.scale)

        elif S == 1:
            # Decode path: store new token KV, then paged attention over full context
            c: PagedKVCache = cache[0]
            c.store(k, v, layer_idx=0)
            attn = c.compute_paged_attn(q, layer_idx=0, scale=self.scale)

        else:
            # Prefill path: update_and_fetch stores new KV and gathers full context
            c: PagedKVCache = cache[0]
            k_full, v_full = c.update_and_fetch(
                k, v
            )  # (B, n_kv_heads, total, head_dim)
            attn = _naive_sdpa(q, k_full, v_full, self.scale)

        # (B, n_heads, S, D) → (B, S, vocab)
        out = attn.transpose(0, 2, 1, 3).reshape(B, S, D)
        return out @ self.Wout


def _make_runner(
    model: TinyPagedModel, num_blocks: int = 200, block_size: int = 8
) -> ModelRunner:
    config = ModelConfig(
        num_layers=1,
        n_kv_heads=model.n_kv_heads,
        head_dim=model.head_dim,
        vocab_size=model.vocab_size,
        hidden_size=model.n_heads * model.head_dim,
        num_attention_heads=model.n_heads,
    )
    # Must use float16: matches _KEY_X=8 x-interleaving in Metal kernel
    return ModelRunner(
        model,
        tokenizer=None,
        config=config,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=mx.float16,
    )


def _last_logits_naive(model: TinyPagedModel, token_ids: list[int]) -> np.ndarray:
    """Run the model with no cache and return logits at the last position."""
    out = model(mx.array([token_ids]), cache=None)  # (1, S, vocab)
    mx.eval(out)
    return np.array(out[0, -1, :])


# ── Prefill correctness ───────────────────────────────────────────────────────


@needs_ext
class TestPrefillMatchesNaive:
    """Prefill via paged cache gives same last-token logits as naive attention."""

    def test_cold_start_prefill(self):
        model = TinyPagedModel()
        runner = _make_runner(model)
        prompt = [1, 2, 3, 4, 5]

        state, result = runner.forward_prefill("req-1", prompt)
        paged_logits = np.array(result.logits)
        runner.finish_request(state)

        ref_logits = _last_logits_naive(model, prompt)
        np.testing.assert_allclose(paged_logits, ref_logits, atol=2e-3)

    def test_prefill_single_token(self):
        model = TinyPagedModel()
        runner = _make_runner(model)
        prompt = [7]

        state, result = runner.forward_prefill("req-1", prompt)
        paged_logits = np.array(result.logits)
        runner.finish_request(state)

        ref_logits = _last_logits_naive(model, prompt)
        np.testing.assert_allclose(paged_logits, ref_logits, atol=2e-3)

    def test_full_cache_hit_prefill(self):
        """Identical prompt reuses blocks and returns correct logits."""
        model = TinyPagedModel()
        runner = _make_runner(model)
        prompt = [3, 5, 7, 9]

        state1, _ = runner.forward_prefill("req-1", prompt)
        runner.finish_request(state1)

        state2, result2 = runner.forward_prefill("req-2", prompt)
        paged_logits = np.array(result2.logits)
        runner.finish_request(state2)

        ref_logits = _last_logits_naive(model, prompt)
        np.testing.assert_allclose(paged_logits, ref_logits, atol=2e-3)


# ── Decode correctness ────────────────────────────────────────────────────────


@needs_ext
class TestDecodeMatchesNaive:
    """Paged decode gives same logits as naive full-sequence attention."""

    ATOL = 3e-3

    def test_single_decode_step(self):
        model = TinyPagedModel()
        runner = _make_runner(model)
        context = [1, 2, 3, 4]
        decode_token = 5

        state, _ = runner.forward_prefill("req-1", context)
        result = runner.forward_decode(state, decode_token)
        paged_logits = np.array(result.logits)
        runner.finish_request(state)

        ref_logits = _last_logits_naive(model, context + [decode_token])
        np.testing.assert_allclose(paged_logits, ref_logits, atol=self.ATOL)

    def test_multiple_decode_steps(self):
        """Each step's logits must match naive attention over the growing context."""
        model = TinyPagedModel()
        runner = _make_runner(model)
        context = [10, 11, 12]
        decode_tokens = [20, 21, 22, 23, 24]

        state, _ = runner.forward_prefill("req-1", context)
        running = list(context)

        for token in decode_tokens:
            result = runner.forward_decode(state, token)
            running.append(token)

            paged_logits = np.array(result.logits)
            ref_logits = _last_logits_naive(model, running)
            np.testing.assert_allclose(
                paged_logits,
                ref_logits,
                atol=self.ATOL,
                err_msg=f"Mismatch after decoding token {token}",
            )

        runner.finish_request(state)

    def test_decode_crosses_block_boundary(self):
        """Growing sequence that allocates a new block mid-generation."""
        model = TinyPagedModel()
        # block_size=8: context of 5 fills block 0 partially,
        # decode tokens cross into block 1
        runner = _make_runner(model, block_size=8)
        context = [1, 2, 3, 4, 5]  # 5 tokens in block 0 (3 slots free)
        decode_tokens = [
            6,
            7,
            8,
            9,
            10,
            11,
        ]  # tokens 6-8 fill block 0; 9-11 go into block 1

        state, _ = runner.forward_prefill("req-1", context)
        running = list(context)

        for token in decode_tokens:
            result = runner.forward_decode(state, token)
            running.append(token)

            paged_logits = np.array(result.logits)
            ref_logits = _last_logits_naive(model, running)
            np.testing.assert_allclose(paged_logits, ref_logits, atol=self.ATOL)

        runner.finish_request(state)


# ── Prefix reuse correctness ─────────────────────────────────────────────────


@needs_ext
class TestPrefixReuseMatchesColdStart:
    """Prefix reuse (radix cache hit) produces identical output to a cold-start run.

    This is the critical end-to-end test: it validates that K/V values stored
    in shared blocks during req-1 are correctly reused in req-2, and that the
    paged attention over those blocks produces the same result as computing
    attention from scratch over the full sequence.
    """

    ATOL = 5e-3  # two Metal kernel passes (prefill + decode) may compound errors

    def test_shared_prefix_decode_matches_cold_start(self):
        """req-2 reuses req-1's prefix; its decode logits match fresh computation."""
        model = TinyPagedModel()
        runner = _make_runner(model, num_blocks=200, block_size=8)

        shared_prefix = list(range(1, 17))  # 16 tokens = 2 blocks of 8
        suffix1 = [20, 21, 22, 23, 24, 25, 26, 27]  # 8 tokens = 1 block
        suffix2 = [30, 31, 32, 33, 34, 35, 36, 37]  # 8 tokens = 1 block
        decode_token = 50

        # Req-1: populate shared prefix in the radix cache
        state1, _ = runner.forward_prefill("req-1", shared_prefix + suffix1)
        runner.finish_request(state1)

        # Req-2: hits shared prefix, processes only suffix2 as new tokens
        state2, _ = runner.forward_prefill("req-2", shared_prefix + suffix2)
        assert state2.prefix_len == len(shared_prefix), (
            f"Expected prefix_len={len(shared_prefix)}, got {state2.prefix_len}"
        )

        result2 = runner.forward_decode(state2, decode_token)
        paged_logits = np.array(result2.logits)
        runner.finish_request(state2)

        # Cold-start reference: process full sequence with no cache
        ref_logits = _last_logits_naive(model, shared_prefix + suffix2 + [decode_token])
        np.testing.assert_allclose(paged_logits, ref_logits, atol=self.ATOL)

    def test_full_prefix_reuse_prefill_matches_cold_start(self):
        """When entire prompt is a cache hit, prefill logits still match naive."""
        model = TinyPagedModel()
        runner = _make_runner(model, num_blocks=200, block_size=8)
        prompt = [5, 6, 7, 8]

        state1, _ = runner.forward_prefill("req-1", prompt)
        runner.finish_request(state1)

        state2, result2 = runner.forward_prefill("req-2", prompt)
        paged_logits = np.array(result2.logits)
        runner.finish_request(state2)

        ref_logits = _last_logits_naive(model, prompt)
        np.testing.assert_allclose(paged_logits, ref_logits, atol=self.ATOL)

    def test_prefix_reuse_multiple_decode_steps(self):
        """Multiple decode steps after a prefix hit all match naive computation."""
        model = TinyPagedModel()
        runner = _make_runner(model, num_blocks=200, block_size=8)

        shared_prefix = list(range(1, 9))  # 8 tokens = 1 block
        suffix = [10, 11, 12, 13, 14, 15, 16, 17]  # 8 tokens = 1 block
        decode_tokens = [20, 21, 22]

        # Req-1: populate cache
        state1, _ = runner.forward_prefill("req-1", shared_prefix + suffix[:4])
        runner.finish_request(state1)

        # Req-2: reuse prefix
        state2, _ = runner.forward_prefill("req-2", shared_prefix + suffix)
        running = list(shared_prefix + suffix)

        for token in decode_tokens:
            result = runner.forward_decode(state2, token)
            running.append(token)

            paged_logits = np.array(result.logits)
            ref_logits = _last_logits_naive(model, running)
            np.testing.assert_allclose(
                paged_logits,
                ref_logits,
                atol=self.ATOL,
                err_msg=f"Mismatch at decode token {token}",
            )

        runner.finish_request(state2)

    def test_no_block_reuse_without_common_prefix(self):
        """Requests with different prefixes don't interfere with each other."""
        model = TinyPagedModel()
        runner = _make_runner(model, num_blocks=200, block_size=8)

        # Two completely different sequences
        seq_a = [1, 2, 3, 4, 5]
        seq_b = [6, 7, 8, 9, 10]
        decode_token = 99

        state_a, _ = runner.forward_prefill("req-a", seq_a)
        result_a = runner.forward_decode(state_a, decode_token)
        paged_a = np.array(result_a.logits)
        runner.finish_request(state_a)

        state_b, _ = runner.forward_prefill("req-b", seq_b)
        result_b = runner.forward_decode(state_b, decode_token)
        paged_b = np.array(result_b.logits)
        runner.finish_request(state_b)

        ref_a = _last_logits_naive(model, seq_a + [decode_token])
        ref_b = _last_logits_naive(model, seq_b + [decode_token])

        np.testing.assert_allclose(paged_a, ref_a, atol=self.ATOL)
        np.testing.assert_allclose(paged_b, ref_b, atol=self.ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
