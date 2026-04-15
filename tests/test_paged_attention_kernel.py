"""Tests for Metal paged attention kernels.

Validates reshape_and_cache (scatter-write) and paged_attention_v1/v2
(attention computation) against a numpy naive reference.

All tests require the compiled Metal extension (_ext) and are skipped
when it is not available.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

try:
    from sglang_mlx import _ext  # noqa: F401

    HAS_EXT = True
except ImportError:
    HAS_EXT = False

needs_ext = pytest.mark.skipif(not HAS_EXT, reason="Metal extension (_ext) not built")

_KEY_X = 8  # x-interleaving factor for key cache


# ── Reference implementation ──────────────────────────────────────────────────


def naive_attention(q_np, keys_np, values_np, scale, n_kv_heads):
    """Exact softmax attention for a single query vector.

    Args:
        q_np:     (n_heads, head_dim)
        keys_np:  (n_tokens, n_kv_heads, head_dim)
        values_np:(n_tokens, n_kv_heads, head_dim)
        scale:    attention scale (1/sqrt(head_dim))
        n_kv_heads: number of KV heads (for GQA broadcasting)

    Returns:
        (n_heads, head_dim) float32 numpy array.
    """
    n_heads = q_np.shape[0]
    group = n_heads // n_kv_heads
    out = np.zeros_like(q_np)
    for h in range(n_heads):
        kv_h = h // group
        k_h = keys_np[:, kv_h, :]  # (n_tokens, D)
        v_h = values_np[:, kv_h, :]
        scores = (q_np[h] @ k_h.T) * scale  # (n_tokens,)
        scores -= scores.max()  # numerical stability
        weights = np.exp(scores)
        weights /= weights.sum()
        out[h] = weights @ v_h
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_empty_cache(num_blocks, n_kv_heads, head_dim, block_size):
    # _KEY_X=8 matches float16 layout (16 / sizeof(half) = 8).
    # The Metal kernel uses x = 16/sizeof(CACHE_T), so the cache must be float16.
    key_cache = mx.zeros(
        (num_blocks, n_kv_heads, head_dim // _KEY_X, block_size, _KEY_X),
        dtype=mx.float16,
    )
    val_cache = mx.zeros(
        (num_blocks, n_kv_heads, head_dim, block_size), dtype=mx.float16
    )
    return key_cache, val_cache


def fill_cache(keys_np, values_np, block_size, n_kv_heads, head_dim, num_blocks):
    """Write tokens into a block cache via reshape_and_cache.

    Tokens are placed in contiguous blocks starting at block 0.
    Returns (key_cache, val_cache, block_tables, seq_lens) ready for
    paged_attention_v1.

    Note: keys/values are stored as float16 — the only dtype compatible with
    the Python _KEY_X=8 x-interleaving and the Metal kernel's x=16/sizeof(half)=8.
    """
    n_tokens = keys_np.shape[0]
    key_cache, val_cache = make_empty_cache(
        num_blocks, n_kv_heads, head_dim, block_size
    )

    keys = mx.array(keys_np.astype(np.float16), dtype=mx.float16)
    values = mx.array(values_np.astype(np.float16), dtype=mx.float16)
    # slot i = block (i // block_size) * block_size + (i % block_size) = i
    slot_mapping = mx.array(list(range(n_tokens)), dtype=mx.int64)

    new_kc, new_vc = _ext.reshape_and_cache(
        keys, values, key_cache, val_cache, slot_mapping, "auto", 1.0, 1.0
    )
    mx.eval(new_kc, new_vc)

    n_blocks_used = math.ceil(n_tokens / block_size)
    block_tables = mx.array([list(range(n_blocks_used))], dtype=mx.int32)
    seq_lens = mx.array([n_tokens], dtype=mx.int32)
    return new_kc, new_vc, block_tables, seq_lens


# ── reshape_and_cache ─────────────────────────────────────────────────────────


@needs_ext
class TestReshapeAndCache:
    N_KV_HEADS = 2
    HEAD_DIM = 32
    BLOCK_SIZE = 8
    NUM_BLOCKS = 16

    def test_single_token_value_written_correctly(self):
        rng = np.random.default_rng(0)
        keys_np = rng.standard_normal((1, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        values_np = rng.standard_normal((1, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        _, vc, _, _ = fill_cache(
            keys_np,
            values_np,
            self.BLOCK_SIZE,
            self.N_KV_HEADS,
            self.HEAD_DIM,
            self.NUM_BLOCKS,
        )

        # val_cache[block=0, head, :, pos=0] == values_np[token=0, head, :]
        vc_np = np.array(vc)  # (num_blocks, n_kv_heads, head_dim, block_size)
        for h in range(self.N_KV_HEADS):
            # float16 has ~3 decimal digits; atol accounts for rounding
            np.testing.assert_allclose(vc_np[0, h, :, 0], values_np[0, h, :], atol=5e-4)

    def test_multiple_tokens_fill_one_block(self):
        rng = np.random.default_rng(1)
        n_tokens = self.BLOCK_SIZE  # exactly fills one block
        keys_np = rng.standard_normal(
            (n_tokens, self.N_KV_HEADS, self.HEAD_DIM)
        ).astype(np.float32)
        values_np = rng.standard_normal(
            (n_tokens, self.N_KV_HEADS, self.HEAD_DIM)
        ).astype(np.float32)

        _, vc, _, _ = fill_cache(
            keys_np,
            values_np,
            self.BLOCK_SIZE,
            self.N_KV_HEADS,
            self.HEAD_DIM,
            self.NUM_BLOCKS,
        )
        vc_np = np.array(vc)

        # Compare against float16-rounded reference: cache stores float16, so reference
        # must be rounded to the same precision before comparing.
        values_f16 = values_np.astype(np.float16)
        for pos in range(n_tokens):
            for h in range(self.N_KV_HEADS):
                np.testing.assert_allclose(
                    vc_np[0, h, :, pos],
                    values_f16[pos, h, :],
                    atol=1e-6,
                    err_msg=f"Mismatch at pos={pos}, head={h}",
                )

    def test_tokens_span_two_blocks(self):
        """Tokens split across block boundary are stored in the correct positions."""
        rng = np.random.default_rng(2)
        n_tokens = self.BLOCK_SIZE * 2  # exactly 2 blocks
        keys_np = rng.standard_normal(
            (n_tokens, self.N_KV_HEADS, self.HEAD_DIM)
        ).astype(np.float32)
        values_np = rng.standard_normal(
            (n_tokens, self.N_KV_HEADS, self.HEAD_DIM)
        ).astype(np.float32)

        _, vc, _, _ = fill_cache(
            keys_np,
            values_np,
            self.BLOCK_SIZE,
            self.N_KV_HEADS,
            self.HEAD_DIM,
            self.NUM_BLOCKS,
        )
        vc_np = np.array(vc)

        values_f16 = values_np.astype(np.float16)
        bs = self.BLOCK_SIZE
        for h in range(self.N_KV_HEADS):
            for pos in range(bs):
                np.testing.assert_allclose(
                    vc_np[0, h, :, pos], values_f16[pos, h, :], atol=1e-6
                )
                np.testing.assert_allclose(
                    vc_np[1, h, :, pos], values_f16[bs + pos, h, :], atol=1e-6
                )

    def test_partial_block_at_end(self):
        """Last block need not be full — only written slots are non-zero."""
        rng = np.random.default_rng(3)
        n_tokens = 5  # 5 tokens in a block of size 8 → positions 5-7 stay zero
        keys_np = rng.standard_normal(
            (n_tokens, self.N_KV_HEADS, self.HEAD_DIM)
        ).astype(np.float32)
        values_np = rng.standard_normal(
            (n_tokens, self.N_KV_HEADS, self.HEAD_DIM)
        ).astype(np.float32)

        _, vc, _, _ = fill_cache(
            keys_np,
            values_np,
            self.BLOCK_SIZE,
            self.N_KV_HEADS,
            self.HEAD_DIM,
            self.NUM_BLOCKS,
        )
        vc_np = np.array(vc)

        values_f16 = values_np.astype(np.float16)
        for pos in range(n_tokens):
            for h in range(self.N_KV_HEADS):
                np.testing.assert_allclose(
                    vc_np[0, h, :, pos], values_f16[pos, h, :], atol=1e-6
                )
        # Positions 5-7 (unused) should still be zero
        np.testing.assert_allclose(vc_np[0, :, :, n_tokens:], 0.0, atol=1e-8)


# ── paged_attention_v1 ────────────────────────────────────────────────────────


@needs_ext
class TestPagedAttentionV1:
    # Minimum supported dimensions: head_size ∈ {32,64,...}, block_size ∈ {8,16,32}
    N_HEADS = 4
    N_KV_HEADS = 4
    HEAD_DIM = 32
    BLOCK_SIZE = 8
    NUM_BLOCKS = 32
    SCALE = HEAD_DIM**-0.5
    # float16 K/V round-trips through the cache with ~3 decimal digits of precision.
    # The kernel accumulates in float32, so errors are dominated by fp16 rounding.
    ATOL = 5e-3

    def _run_paged(self, q_np, keys_np, values_np, block_size=None, n_kv_heads=None):
        bs = block_size or self.BLOCK_SIZE
        nkv = n_kv_heads or self.N_KV_HEADS
        n_tokens = keys_np.shape[0]

        kc, vc, block_tables, seq_lens = fill_cache(
            keys_np, values_np, bs, nkv, self.HEAD_DIM, self.NUM_BLOCKS
        )
        # Query must be float16 to match cache dtype and kernel template
        # T=half, CACHE_T=half
        q = mx.array(
            q_np[None].astype(np.float16), dtype=mx.float16
        )  # (1, n_heads, head_dim)
        out = _ext.paged_attention_v1(
            q,
            kc,
            vc,
            block_tables,
            seq_lens,
            nkv,
            self.SCALE,
            bs,
            n_tokens,
            None,
            "auto",
            1.0,
            1.0,
        )
        mx.eval(out)
        return np.array(out[0]).astype(np.float32)  # (n_heads, head_dim)

    def _ref(self, q_np, k_np, v_np):
        # Round to float16 to match what the cache stores, then compute in float64.
        return naive_attention(
            q_np.astype(np.float16).astype(np.float64),
            k_np.astype(np.float16).astype(np.float64),
            v_np.astype(np.float16).astype(np.float64),
            self.SCALE,
            self.N_KV_HEADS,
        )

    def test_single_block_matches_naive(self):
        rng = np.random.default_rng(10)
        n_tokens = self.BLOCK_SIZE  # exactly one block
        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        paged = self._run_paged(q_np, k_np, v_np)
        ref = self._ref(q_np, k_np, v_np)

        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_multi_block_matches_naive(self):
        rng = np.random.default_rng(11)
        n_tokens = self.BLOCK_SIZE * 3  # 3 full blocks
        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        paged = self._run_paged(q_np, k_np, v_np)
        ref = self._ref(q_np, k_np, v_np)
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_single_token_context(self):
        """Edge case: attending to exactly one cached token."""
        rng = np.random.default_rng(12)
        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((1, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((1, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        paged = self._run_paged(q_np, k_np, v_np)
        ref = self._ref(q_np, k_np, v_np)
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_partial_block_matches_naive(self):
        """Sequence that does not fill the last block evenly."""
        rng = np.random.default_rng(13)
        n_tokens = self.BLOCK_SIZE + 3  # 1 full block + 3 tokens in second block
        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        paged = self._run_paged(q_np, k_np, v_np)
        ref = self._ref(q_np, k_np, v_np)
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_gqa_matches_naive(self):
        """Grouped-query attention: n_heads=4, n_kv_heads=2."""
        N_HEADS, N_KV_HEADS = 4, 2
        rng = np.random.default_rng(14)
        n_tokens = self.BLOCK_SIZE * 2
        q_np = rng.standard_normal((N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        paged = self._run_paged(q_np, k_np, v_np, n_kv_heads=N_KV_HEADS)
        ref = naive_attention(
            q_np.astype(np.float16).astype(np.float64),
            k_np.astype(np.float16).astype(np.float64),
            v_np.astype(np.float16).astype(np.float64),
            self.SCALE,
            N_KV_HEADS,
        )
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_long_sequence_matches_naive(self):
        """Long sequence spanning many blocks."""
        rng = np.random.default_rng(15)
        n_tokens = self.BLOCK_SIZE * 8  # 8 blocks
        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        paged = self._run_paged(q_np, k_np, v_np)
        ref = self._ref(q_np, k_np, v_np)
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_block_size_16_matches_naive(self):
        """Larger block size supported by the kernel."""
        rng = np.random.default_rng(16)
        n_tokens = 32  # 2 blocks of size 16
        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        paged = self._run_paged(q_np, k_np, v_np, block_size=16)
        ref = self._ref(q_np, k_np, v_np)
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)


# ── paged_attention_v2 ────────────────────────────────────────────────────────


@needs_ext
class TestPagedAttentionV2:
    """paged_attention_v2 (two-pass) output must equal v1 (single-pass)."""

    N_HEADS = 4
    N_KV_HEADS = 4
    HEAD_DIM = 32
    BLOCK_SIZE = 8
    NUM_BLOCKS = 32
    SCALE = HEAD_DIM**-0.5
    ATOL = 1e-4  # v1 vs v2 should agree very tightly

    def _run_both(self, n_tokens, seed):
        rng = np.random.default_rng(seed)
        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        kc, vc, block_tables, seq_lens = fill_cache(
            k_np, v_np, self.BLOCK_SIZE, self.N_KV_HEADS, self.HEAD_DIM, self.NUM_BLOCKS
        )
        q = mx.array(q_np[None].astype(np.float16), dtype=mx.float16)

        out_v1 = _ext.paged_attention_v1(
            q,
            kc,
            vc,
            block_tables,
            seq_lens,
            self.N_KV_HEADS,
            self.SCALE,
            self.BLOCK_SIZE,
            n_tokens,
            None,
            "auto",
            1.0,
            1.0,
        )
        out_v2 = _ext.paged_attention_v2(
            q,
            kc,
            vc,
            block_tables,
            seq_lens,
            self.N_KV_HEADS,
            self.SCALE,
            self.BLOCK_SIZE,
            n_tokens,
            16,
            None,
            "auto",
            1.0,
            1.0,
        )[0]  # first element is the output tensor

        mx.eval(out_v1, out_v2)
        return np.array(out_v1[0]), np.array(out_v2[0])

    def test_v2_matches_v1_short(self):
        v1, v2 = self._run_both(n_tokens=self.BLOCK_SIZE, seed=20)
        np.testing.assert_allclose(v1, v2, atol=self.ATOL)

    def test_v2_matches_v1_long(self):
        v1, v2 = self._run_both(n_tokens=self.BLOCK_SIZE * 4, seed=21)
        np.testing.assert_allclose(v1, v2, atol=self.ATOL)

    def test_v2_matches_naive(self):
        rng = np.random.default_rng(22)
        n_tokens = self.BLOCK_SIZE * 2
        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        kc, vc, block_tables, seq_lens = fill_cache(
            k_np, v_np, self.BLOCK_SIZE, self.N_KV_HEADS, self.HEAD_DIM, self.NUM_BLOCKS
        )
        q = mx.array(q_np[None].astype(np.float16), dtype=mx.float16)
        out_v2 = _ext.paged_attention_v2(
            q,
            kc,
            vc,
            block_tables,
            seq_lens,
            self.N_KV_HEADS,
            self.SCALE,
            self.BLOCK_SIZE,
            n_tokens,
            16,
            None,
            "auto",
            1.0,
            1.0,
        )[0]
        mx.eval(out_v2)

        paged = np.array(out_v2[0]).astype(np.float32)
        ref = naive_attention(
            q_np.astype(np.float16).astype(np.float64),
            k_np.astype(np.float16).astype(np.float64),
            v_np.astype(np.float16).astype(np.float64),
            self.SCALE,
            self.N_KV_HEADS,
        )
        np.testing.assert_allclose(paged, ref, atol=5e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
