"""Tests for the paged KV memory pool: block allocator and cache adapter."""

import mlx.core as mx
import numpy as np
import pytest

from sglang_mlx.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang_mlx.srt.mem_cache.paged_pool import (
    BlockAllocator,
    PagedKVCache,
    PagedMHAPool,
)
from sglang_mlx.srt.mem_cache.radix_cache import RadixCache

try:
    from sglang_mlx import _ext  # noqa: F401

    HAS_EXT = True
except ImportError:
    HAS_EXT = False

needs_ext = pytest.mark.skipif(not HAS_EXT, reason="Metal extension (_ext) not built")


# ── BlockAllocator ────────────────────────────────────────────────────────────


class TestBlockAllocator:
    def test_basic_alloc_free(self):
        alloc = BlockAllocator(num_blocks=10, block_size=1)
        assert alloc.available_blocks == 10

        blocks = alloc.alloc(3)
        assert len(blocks) == 3
        assert alloc.available_blocks == 7

        alloc.free(blocks)
        assert alloc.available_blocks == 10

    def test_alloc_zero_returns_empty(self):
        alloc = BlockAllocator(num_blocks=5, block_size=1)
        assert alloc.alloc(0) == []
        assert alloc.available_blocks == 5

    def test_alloc_overflow_raises(self):
        alloc = BlockAllocator(num_blocks=5, block_size=1)
        with pytest.raises(MemoryError, match="Cannot allocate"):
            alloc.alloc(6)

    def test_alloc_exact_capacity(self):
        alloc = BlockAllocator(num_blocks=5, block_size=1)
        blocks = alloc.alloc(5)
        assert len(blocks) == 5
        assert alloc.available_blocks == 0

    def test_free_then_realloc(self):
        alloc = BlockAllocator(num_blocks=5, block_size=1)
        b1 = alloc.alloc(3)
        alloc.free(b1)
        b2 = alloc.alloc(5)
        assert len(b2) == 5

    def test_clear_resets(self):
        alloc = BlockAllocator(num_blocks=10, block_size=1)
        alloc.alloc(10)
        assert alloc.available_blocks == 0
        alloc.clear()
        assert alloc.available_blocks == 10

    def test_invalid_num_blocks(self):
        with pytest.raises(ValueError):
            BlockAllocator(num_blocks=0, block_size=1)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            BlockAllocator(num_blocks=10, block_size=3)  # not a power of 2

    def test_no_duplicate_blocks(self):
        alloc = BlockAllocator(num_blocks=50, block_size=1)
        b1 = alloc.alloc(20)
        b2 = alloc.alloc(20)
        all_blocks = b1 + b2
        assert len(set(all_blocks)) == len(all_blocks)

    def test_alloc_for_tokens(self):
        alloc = BlockAllocator(num_blocks=10, block_size=4)
        # 5 tokens needs ceil(5/4) = 2 blocks
        blocks = alloc.alloc_for_tokens(5)
        assert len(blocks) == 2
        assert alloc.available_blocks == 8

    def test_free_deduplicates(self):
        """free() should handle repeated block IDs gracefully."""
        alloc = BlockAllocator(num_blocks=10, block_size=1)
        blocks = alloc.alloc(3)
        before = alloc.available_blocks
        # Pass repeated IDs (as occurs with token-granularity radix cache storage)
        alloc.free(blocks + blocks)
        # Should only free each block once
        assert alloc.available_blocks == before + 3


# ── PagedMHAPool ──────────────────────────────────────────────────────────────


class TestPagedMHAPool:
    NUM_LAYERS = 2
    N_KV_HEADS = 4
    HEAD_DIM = 8  # must be divisible by 8 (_KEY_X)
    NUM_BLOCKS = 16
    BLOCK_SIZE = 2

    def _make_pool(self) -> PagedMHAPool:
        return PagedMHAPool(
            num_blocks=self.NUM_BLOCKS,
            num_layers=self.NUM_LAYERS,
            n_kv_heads=self.N_KV_HEADS,
            head_dim=self.HEAD_DIM,
            block_size=self.BLOCK_SIZE,
            dtype=mx.float32,
        )

    def test_init_shapes(self):
        pool = self._make_pool()
        assert len(pool.key_cache) == self.NUM_LAYERS
        assert len(pool.val_cache) == self.NUM_LAYERS

        # key: (num_blocks, n_kv_heads, head_dim//x, block_size, x)
        x = 8
        assert pool.key_cache[0].shape == (
            self.NUM_BLOCKS,
            self.N_KV_HEADS,
            self.HEAD_DIM // x,
            self.BLOCK_SIZE,
            x,
        )
        # val: (num_blocks, n_kv_heads, head_dim, block_size)
        assert pool.val_cache[0].shape == (
            self.NUM_BLOCKS,
            self.N_KV_HEADS,
            self.HEAD_DIM,
            self.BLOCK_SIZE,
        )

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim must be divisible"):
            PagedMHAPool(
                num_blocks=4,
                num_layers=1,
                n_kv_heads=2,
                head_dim=7,
                block_size=2,  # 7 % 8 != 0
            )

    @needs_ext
    def test_store_changes_cache(self):
        pool = self._make_pool()
        keys = mx.ones((1, self.N_KV_HEADS, self.HEAD_DIM), dtype=mx.float32)
        values = mx.ones((1, self.N_KV_HEADS, self.HEAD_DIM), dtype=mx.float32)
        slot_mapping = mx.array([0], dtype=mx.int64)  # block 0, position 0

        pool.store(0, keys, values, slot_mapping)
        # After store, cache should have been updated (non-zero)
        mx.eval(pool.val_cache[0])
        val = np.array(pool.val_cache[0], copy=False)
        assert val.sum() > 0


# ── PagedKVCache ──────────────────────────────────────────────────────────────


class TestPagedKVCache:
    NUM_LAYERS = 2
    N_KV_HEADS = 4
    HEAD_DIM = 8
    NUM_BLOCKS = 32
    BLOCK_SIZE = 2

    def _make_pool(self) -> PagedMHAPool:
        return PagedMHAPool(
            num_blocks=self.NUM_BLOCKS,
            num_layers=self.NUM_LAYERS,
            n_kv_heads=self.N_KV_HEADS,
            head_dim=self.HEAD_DIM,
            block_size=self.BLOCK_SIZE,
            dtype=mx.float32,
        )

    def test_offset_reflects_prefix_tokens(self):
        pool = self._make_pool()
        cache = PagedKVCache(pool, layer_idx=0)
        cache.set_blocks(prefix_blocks=[0, 1], new_blocks=[2], prefix_tokens=4)
        assert cache.offset == 4

    def test_offset_zero_cold_start(self):
        pool = self._make_pool()
        cache = PagedKVCache(pool, layer_idx=0)
        cache.set_blocks(prefix_blocks=[], new_blocks=[0], prefix_tokens=0)
        assert cache.offset == 0

    def test_make_mask_decode_returns_none(self):
        pool = self._make_pool()
        cache = PagedKVCache(pool, layer_idx=0)
        cache.set_blocks(prefix_blocks=[0, 1], new_blocks=[2], prefix_tokens=4)
        assert cache.make_mask(N=1) is None

    def test_make_mask_prefill_causal(self):
        pool = self._make_pool()
        cache = PagedKVCache(pool, layer_idx=0)
        cache.set_blocks(prefix_blocks=[], new_blocks=[0, 1], prefix_tokens=0)
        assert cache.make_mask(N=4) == "causal"

    @needs_ext
    def test_update_and_fetch_shape(self):
        """update_and_fetch returns correct shape for prefix + new tokens."""
        pool = self._make_pool()
        cache = PagedKVCache(pool, layer_idx=0)
        # 2 blocks = 4 tokens of prefix + 1 new block (2 new tokens)
        cache.set_blocks(prefix_blocks=[], new_blocks=[0, 1], prefix_tokens=0)

        shape = (1, self.N_KV_HEADS, 4, self.HEAD_DIM)
        k_in = mx.zeros(shape, dtype=mx.float32)
        v_in = mx.zeros(shape, dtype=mx.float32)
        k_out, v_out = cache.update_and_fetch(k_in, v_in)

        assert k_out.shape == shape
        assert v_out.shape == shape


# ── Integration: RadixCache + BlockAllocator ──────────────────────────────────


class TestRadixCacheAllocatorIntegration:
    """Tests that eviction frees blocks back to the allocator."""

    def test_eviction_frees_blocks(self):
        alloc = BlockAllocator(num_blocks=20, block_size=1)
        cache = RadixCache()
        cache.set_allocator(alloc)

        blocks1 = alloc.alloc(5)
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5], value=blocks1))

        blocks2 = alloc.alloc(5)
        cache.insert(InsertParams(key=[6, 7, 8, 9, 10], value=blocks2))

        assert alloc.available_blocks == 10

        cache.evict(EvictParams(num_tokens=100))

        assert alloc.available_blocks == 20

    def test_eviction_partial(self):
        alloc = BlockAllocator(num_blocks=20, block_size=1)
        cache = RadixCache()
        cache.set_allocator(alloc)

        blocks1 = alloc.alloc(3)
        cache.insert(InsertParams(key=[1, 2, 3], value=blocks1))

        blocks2 = alloc.alloc(3)
        cache.insert(InsertParams(key=[4, 5, 6], value=blocks2))

        cache.match_prefix([4, 5, 6])  # make [4,5,6] most recent

        before = alloc.available_blocks
        cache.evict(EvictParams(num_tokens=3))
        after = alloc.available_blocks

        assert after - before == 3

    def test_no_allocator_attached(self):
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))
        cache.evict(EvictParams(num_tokens=100))
        result = cache.match_prefix([1, 2, 3, 4, 5])
        assert len(result.device_indices) == 0

    def test_prefix_sharing(self):
        alloc = BlockAllocator(num_blocks=50, block_size=1)
        cache = RadixCache()
        cache.set_allocator(alloc)

        blocks_r1 = alloc.alloc(5)
        cache.insert(InsertParams(key=[1, 2, 3, 10, 11], value=blocks_r1))

        result = cache.match_prefix([1, 2, 3, 20, 21])
        assert len(result.device_indices) == 3  # [1,2,3] matched

        blocks_r2_new = alloc.alloc(2)
        cache.insert(
            InsertParams(
                key=[1, 2, 3, 20, 21],
                value=result.device_indices + blocks_r2_new,
            )
        )

        assert alloc.available_blocks == 50 - 5 - 2


# ── PagedKVCache.compute_paged_attn ──────────────────────────────────────────


def _naive_attention_np(q_np, keys_np, values_np, scale, n_kv_heads):
    """Single-query softmax attention reference.

    q_np:     (n_heads, head_dim)
    keys_np:  (n_tokens, n_kv_heads, head_dim)
    values_np:(n_tokens, n_kv_heads, head_dim)
    Returns:  (n_heads, head_dim)
    """
    n_heads = q_np.shape[0]
    group = n_heads // n_kv_heads
    out = np.zeros_like(q_np)
    for h in range(n_heads):
        kv_h = h // group
        k_h = keys_np[:, kv_h, :]
        v_h = values_np[:, kv_h, :]
        scores = (q_np[h] @ k_h.T) * scale
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        out[h] = weights @ v_h
    return out


@needs_ext
class TestComputePagedAttn:
    """Tests that PagedKVCache.compute_paged_attn matches naive attention."""

    # Must use kernel-supported sizes: head_size ∈ {32,64,...}, block_size ∈ {8,16,32}
    N_HEADS = 4
    N_KV_HEADS = 4
    HEAD_DIM = 32
    NUM_BLOCKS = 32
    BLOCK_SIZE = 8
    ATOL = 2e-3

    def _make_pool(self, block_size=None):
        # Must use float16: _KEY_X=8 matches 16/sizeof(half)=8 for the Metal kernel.
        return PagedMHAPool(
            num_blocks=self.NUM_BLOCKS,
            num_layers=1,
            n_kv_heads=self.N_KV_HEADS,
            head_dim=self.HEAD_DIM,
            block_size=block_size or self.BLOCK_SIZE,
            dtype=mx.float16,
        )

    def _to_cache_fmt(self, kv_np):
        """Convert (n_tokens, n_kv_heads, head_dim) →
        (1, n_kv_heads, n_tokens, head_dim)."""
        return mx.array(kv_np.astype(np.float16), dtype=mx.float16).transpose(1, 0, 2)[
            None
        ]

    def _to_query_fmt(self, q_np):
        """Convert (n_heads, head_dim) → (1, n_heads, 1, head_dim)."""
        return mx.array(q_np.astype(np.float16), dtype=mx.float16)[None, :, None, :]

    def test_single_block_matches_naive(self):
        pool = self._make_pool()
        rng = np.random.default_rng(40)
        scale = self.HEAD_DIM**-0.5
        n_tokens = self.BLOCK_SIZE  # exactly fills one block

        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        # Store all tokens as a prefill pass
        cache = PagedKVCache(pool, layer_idx=0)
        cache.set_blocks(prefix_blocks=[], new_blocks=[0], prefix_tokens=0)
        cache.store(self._to_cache_fmt(k_np), self._to_cache_fmt(v_np), layer_idx=0)
        mx.eval(pool.key_cache[0], pool.val_cache[0])

        # Decode: treat all stored tokens as prefix
        cache.set_blocks(prefix_blocks=[0], new_blocks=[], prefix_tokens=n_tokens)
        attn_out = cache.compute_paged_attn(
            self._to_query_fmt(q_np), layer_idx=0, scale=scale
        )
        mx.eval(attn_out)

        paged = np.array(attn_out[0, :, 0, :]).astype(np.float32)
        # Round K/V to float16 to match what was stored in the cache
        ref = _naive_attention_np(
            q_np.astype(np.float16).astype(np.float64),
            k_np.astype(np.float16).astype(np.float64),
            v_np.astype(np.float16).astype(np.float64),
            scale,
            self.N_KV_HEADS,
        )
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_multi_block_matches_naive(self):
        """Sequence spanning three blocks of size 8."""
        pool = self._make_pool()
        rng = np.random.default_rng(41)
        scale = self.HEAD_DIM**-0.5
        n_tokens = self.BLOCK_SIZE * 3  # 3 full blocks

        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_tokens, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        cache = PagedKVCache(pool, layer_idx=0)
        cache.set_blocks(prefix_blocks=[], new_blocks=[0, 1, 2], prefix_tokens=0)
        cache.store(self._to_cache_fmt(k_np), self._to_cache_fmt(v_np), layer_idx=0)
        mx.eval(pool.key_cache[0], pool.val_cache[0])

        cache.set_blocks(prefix_blocks=[0, 1, 2], new_blocks=[], prefix_tokens=n_tokens)
        attn_out = cache.compute_paged_attn(
            self._to_query_fmt(q_np), layer_idx=0, scale=scale
        )
        mx.eval(attn_out)

        paged = np.array(attn_out[0, :, 0, :]).astype(np.float32)
        # Round K/V to float16 to match what was stored in the cache
        ref = _naive_attention_np(
            q_np.astype(np.float16).astype(np.float64),
            k_np.astype(np.float16).astype(np.float64),
            v_np.astype(np.float16).astype(np.float64),
            scale,
            self.N_KV_HEADS,
        )
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_with_prefix_blocks(self):
        """K/V split across prefix blocks (req1) and new blocks (req2) matches naive.

        Simulates: req1 stores tokens 0..15 in blocks [0,1]. req2 reuses those
        blocks as prefix, stores tokens 16..23 in block [2], then queries.
        The paged attention should match naive over all 24 tokens.
        """
        pool = self._make_pool()
        rng = np.random.default_rng(42)
        scale = self.HEAD_DIM**-0.5
        n_prefix, n_new = self.BLOCK_SIZE * 2, self.BLOCK_SIZE
        n_total = n_prefix + n_new

        q_np = rng.standard_normal((self.N_HEADS, self.HEAD_DIM)).astype(np.float32)
        k_np = rng.standard_normal((n_total, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_total, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        # Req1: store prefix tokens in blocks 0 and 1
        cache1 = PagedKVCache(pool, layer_idx=0)
        cache1.set_blocks(prefix_blocks=[], new_blocks=[0, 1], prefix_tokens=0)
        cache1.store(
            self._to_cache_fmt(k_np[:n_prefix]),
            self._to_cache_fmt(v_np[:n_prefix]),
            layer_idx=0,
        )
        mx.eval(pool.key_cache[0], pool.val_cache[0])

        # Req2: reuse blocks 0,1 as prefix, store new tokens in block 2
        cache2 = PagedKVCache(pool, layer_idx=0)
        cache2.set_blocks(prefix_blocks=[0, 1], new_blocks=[2], prefix_tokens=n_prefix)
        cache2.store(
            self._to_cache_fmt(k_np[n_prefix:]),
            self._to_cache_fmt(v_np[n_prefix:]),
            layer_idx=0,
        )
        mx.eval(pool.key_cache[0], pool.val_cache[0])

        # Decode over full context
        cache2.set_blocks(prefix_blocks=[0, 1, 2], new_blocks=[], prefix_tokens=n_total)
        attn_out = cache2.compute_paged_attn(
            self._to_query_fmt(q_np), layer_idx=0, scale=scale
        )
        mx.eval(attn_out)

        paged = np.array(attn_out[0, :, 0, :]).astype(np.float32)
        # Round K/V to float16 to match what was stored in the cache
        ref = _naive_attention_np(
            q_np.astype(np.float16).astype(np.float64),
            k_np.astype(np.float16).astype(np.float64),
            v_np.astype(np.float16).astype(np.float64),
            scale,
            self.N_KV_HEADS,
        )
        np.testing.assert_allclose(paged, ref, atol=self.ATOL)

    def test_decode_step_by_step(self):
        """5 consecutive decode steps — matches naive over growing context."""
        pool = self._make_pool()
        rng = np.random.default_rng(43)
        scale = self.HEAD_DIM**-0.5
        n_context = 3
        n_steps = 5
        n_total = n_context + n_steps  # 8 tokens, fits in one block of size 8

        # Pre-generate all K/V/Q so we can compare each step
        k_np = rng.standard_normal((n_total, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        v_np = rng.standard_normal((n_total, self.N_KV_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )
        q_np = rng.standard_normal((n_total, self.N_HEADS, self.HEAD_DIM)).astype(
            np.float32
        )

        # One block of size 8 covers all 8 tokens
        all_blocks = [0]
        cache = PagedKVCache(pool, layer_idx=0)

        # Store context tokens (prefill simulation)
        cache.set_blocks(prefix_blocks=[], new_blocks=all_blocks, prefix_tokens=0)
        cache.store(
            self._to_cache_fmt(k_np[:n_context]),
            self._to_cache_fmt(v_np[:n_context]),
            layer_idx=0,
        )
        mx.eval(pool.key_cache[0], pool.val_cache[0])

        for step in range(n_steps):
            total_so_far = n_context + step

            # Simulate the decode setup from ModelRunner
            cache.set_blocks(
                prefix_blocks=all_blocks,
                new_blocks=[],
                prefix_tokens=total_so_far,
            )
            # Store the new decode token
            k_new = k_np[total_so_far : total_so_far + 1]
            v_new = v_np[total_so_far : total_so_far + 1]
            cache.store(
                self._to_cache_fmt(k_new), self._to_cache_fmt(v_new), layer_idx=0
            )

            q_step = q_np[total_so_far]
            attn_out = cache.compute_paged_attn(
                self._to_query_fmt(q_step), layer_idx=0, scale=scale
            )
            mx.eval(attn_out)

            paged = np.array(attn_out[0, :, 0, :])
            ref = _naive_attention_np(
                q_step,
                k_np[: total_so_far + 1],
                v_np[: total_so_far + 1],
                scale,
                self.N_KV_HEADS,
            )
            np.testing.assert_allclose(
                paged, ref, atol=self.ATOL, err_msg=f"Mismatch at decode step {step}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
