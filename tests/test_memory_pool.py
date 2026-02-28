"""Tests for the KV memory pool: allocator, MLX buffers, and cache adapter."""

import mlx.core as mx
import numpy as np
import pytest

from sglang_mlx.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang_mlx.srt.mem_cache.memory_pool import (
    MHATokenPool,
    PooledKVCache,
    TokenPoolAllocator,
)
from sglang_mlx.srt.mem_cache.radix_cache import RadixCache

# ---- TokenPoolAllocator ----


class TestTokenPoolAllocator:
    def test_basic_alloc_free(self):
        alloc = TokenPoolAllocator(pool_size=10)
        assert alloc.available_size == 10

        slots = alloc.alloc(3)
        assert len(slots) == 3
        assert alloc.available_size == 7

        alloc.free(slots)
        assert alloc.available_size == 10

    def test_slot_zero_reserved(self):
        """Slot 0 must never be allocated (padding convention)."""
        alloc = TokenPoolAllocator(pool_size=100)
        all_slots = alloc.alloc(100)
        assert 0 not in all_slots
        assert all(s >= 1 for s in all_slots)

    def test_alloc_zero_returns_empty(self):
        alloc = TokenPoolAllocator(pool_size=5)
        assert alloc.alloc(0) == []
        assert alloc.available_size == 5

    def test_alloc_overflow_raises(self):
        alloc = TokenPoolAllocator(pool_size=5)
        with pytest.raises(MemoryError, match="Cannot allocate"):
            alloc.alloc(6)

    def test_alloc_exact_capacity(self):
        alloc = TokenPoolAllocator(pool_size=5)
        slots = alloc.alloc(5)
        assert len(slots) == 5
        assert alloc.available_size == 0

    def test_free_then_realloc(self):
        alloc = TokenPoolAllocator(pool_size=5)
        s1 = alloc.alloc(3)
        alloc.free(s1)
        s2 = alloc.alloc(5)
        assert len(s2) == 5

    def test_clear_resets(self):
        alloc = TokenPoolAllocator(pool_size=10)
        alloc.alloc(10)
        assert alloc.available_size == 0
        alloc.clear()
        assert alloc.available_size == 10

    def test_invalid_pool_size(self):
        with pytest.raises(ValueError):
            TokenPoolAllocator(pool_size=0)
        with pytest.raises(ValueError):
            TokenPoolAllocator(pool_size=-1)

    def test_no_duplicate_slots(self):
        """All allocated slots should be unique."""
        alloc = TokenPoolAllocator(pool_size=50)
        s1 = alloc.alloc(20)
        s2 = alloc.alloc(20)
        all_slots = s1 + s2
        assert len(set(all_slots)) == len(all_slots)


# ---- MHATokenPool ----


class TestMHATokenPool:
    """Tests for the MLX KV buffer pool."""

    NUM_LAYERS = 2
    N_KV_HEADS = 4
    HEAD_DIM = 8
    POOL_SIZE = 32

    def _make_pool(self) -> MHATokenPool:
        return MHATokenPool(
            pool_size=self.POOL_SIZE,
            num_layers=self.NUM_LAYERS,
            n_kv_heads=self.N_KV_HEADS,
            head_dim=self.HEAD_DIM,
            dtype=mx.float32,  # float32 for easier value checking
        )

    def _random_kv(self, seq_len: int) -> tuple[mx.array, mx.array]:
        """Return (keys, values) in mlx-lm format: (1, H, S, D)."""
        shape = (1, self.N_KV_HEADS, seq_len, self.HEAD_DIM)
        k = mx.random.normal(shape)
        v = mx.random.normal(shape)
        return k, v

    def test_store_fetch_roundtrip(self):
        pool = self._make_pool()
        k, v = self._random_kv(5)
        indices = [1, 2, 3, 4, 5]

        pool.store(0, indices, k, v)
        k_out, v_out = pool.fetch(0, indices)

        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
        np.testing.assert_allclose(
            np.array(k_out, copy=False), np.array(k, copy=False), atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(v_out, copy=False), np.array(v, copy=False), atol=1e-6
        )

    def test_cross_layer_independence(self):
        pool = self._make_pool()
        k0, v0 = self._random_kv(3)
        k1, v1 = self._random_kv(3)
        indices = [1, 2, 3]

        pool.store(0, indices, k0, v0)
        pool.store(1, indices, k1, v1)

        k0_out, _ = pool.fetch(0, indices)
        k1_out, _ = pool.fetch(1, indices)

        # Layer 0 should have layer 0's data, not layer 1's
        np.testing.assert_allclose(
            np.array(k0_out, copy=False), np.array(k0, copy=False), atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(k1_out, copy=False), np.array(k1, copy=False), atol=1e-6
        )

    def test_non_contiguous_indices(self):
        """Slots don't have to be contiguous; gather should still work."""
        pool = self._make_pool()
        k, v = self._random_kv(3)
        indices = [5, 10, 20]

        pool.store(0, indices, k, v)
        k_out, v_out = pool.fetch(0, indices)

        np.testing.assert_allclose(
            np.array(k_out, copy=False), np.array(k, copy=False), atol=1e-6
        )

    def test_overwrite_existing_slot(self):
        pool = self._make_pool()
        k1, v1 = self._random_kv(1)
        k2, v2 = self._random_kv(1)

        pool.store(0, [1], k1, v1)
        pool.store(0, [1], k2, v2)

        k_out, v_out = pool.fetch(0, [1])
        np.testing.assert_allclose(
            np.array(k_out, copy=False), np.array(k2, copy=False), atol=1e-6
        )

    def test_fetch_subset_of_stored(self):
        """Fetch a subset of previously stored indices."""
        pool = self._make_pool()
        k, v = self._random_kv(5)
        indices = [1, 2, 3, 4, 5]
        pool.store(0, indices, k, v)

        # Fetch only first 3
        k_out, v_out = pool.fetch(0, [1, 2, 3])
        assert k_out.shape == (1, self.N_KV_HEADS, 3, self.HEAD_DIM)

        # Values should match the first 3 tokens
        np.testing.assert_allclose(
            np.array(k_out, copy=False),
            np.array(k[:, :, :3, :], copy=False),
            atol=1e-6,
        )


# ---- PooledKVCache ----


class TestPooledKVCache:
    """Tests for the per-request cache adapter."""

    NUM_LAYERS = 2
    N_KV_HEADS = 4
    HEAD_DIM = 8
    POOL_SIZE = 64

    def _make_pool(self) -> MHATokenPool:
        return MHATokenPool(
            pool_size=self.POOL_SIZE,
            num_layers=self.NUM_LAYERS,
            n_kv_heads=self.N_KV_HEADS,
            head_dim=self.HEAD_DIM,
            dtype=mx.float32,
        )

    def _random_kv(self, seq_len: int) -> tuple[mx.array, mx.array]:
        shape = (1, self.N_KV_HEADS, seq_len, self.HEAD_DIM)
        return mx.random.normal(shape), mx.random.normal(shape)

    def test_cold_start_prefill(self):
        """No prefix match — all tokens are new."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        new_indices = [1, 2, 3, 4, 5]
        cache.set_indices(prefix_indices=[], new_indices=new_indices)

        assert cache.offset == 0  # RoPE starts at 0

        k_in, v_in = self._random_kv(5)
        k_out, v_out = cache.update_and_fetch(k_in, v_in)

        assert k_out.shape == (1, self.N_KV_HEADS, 5, self.HEAD_DIM)
        np.testing.assert_allclose(
            np.array(k_out, copy=False), np.array(k_in, copy=False), atol=1e-6
        )

    def test_decode_with_prefix(self):
        """Prefix cached, decoding one new token at a time."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)

        # Simulate a prefill that already stored 5 tokens
        prefix_indices = [1, 2, 3, 4, 5]
        k_prefix, v_prefix = self._random_kv(5)
        pool.store(0, prefix_indices, k_prefix, v_prefix)

        # Now decode 1 new token
        new_indices = [6]
        cache.set_indices(prefix_indices=prefix_indices, new_indices=new_indices)
        assert cache.offset == 5  # RoPE position for new token

        k_new, v_new = self._random_kv(1)
        k_out, v_out = cache.update_and_fetch(k_new, v_new)

        # Output should have 6 tokens total (5 prefix + 1 new)
        assert k_out.shape == (1, self.N_KV_HEADS, 6, self.HEAD_DIM)

        # Verify prefix portion matches
        np.testing.assert_allclose(
            np.array(k_out[:, :, :5, :], copy=False),
            np.array(k_prefix, copy=False),
            atol=1e-6,
        )
        # Verify new token matches
        np.testing.assert_allclose(
            np.array(k_out[:, :, 5:, :], copy=False),
            np.array(k_new, copy=False),
            atol=1e-6,
        )

    def test_prefix_reuse_offset(self):
        """Prefix match from radix cache — offset should skip recomputation."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)

        prefix_indices = [10, 11, 12]
        k_prefix, v_prefix = self._random_kv(3)
        pool.store(0, prefix_indices, k_prefix, v_prefix)

        new_indices = [20, 21]
        cache.set_indices(prefix_indices=prefix_indices, new_indices=new_indices)

        assert cache.offset == 3  # skip first 3 positions

        k_new, v_new = self._random_kv(2)
        k_out, v_out = cache.update_and_fetch(k_new, v_new)
        assert k_out.shape == (1, self.N_KV_HEADS, 5, self.HEAD_DIM)

    def test_mask_decode_returns_none(self):
        """Single-token decode should return None (no mask needed)."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        cache.set_indices(prefix_indices=[1, 2, 3], new_indices=[4])
        assert cache.make_mask(N=1) is None

    def test_mask_prefill_causal(self):
        """Prefill without window returns 'causal' string (mlx-lm convention)."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        cache.set_indices(prefix_indices=[], new_indices=[1, 2, 3, 4])
        mask = cache.make_mask(N=4)
        assert mask == "causal"

    def test_mask_with_offset(self):
        """Prefill after cached prefix returns 'causal' string."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        cache.set_indices(prefix_indices=[1, 2, 3], new_indices=[4, 5])
        mask = cache.make_mask(N=2)
        assert mask == "causal"

    def test_mask_with_window(self):
        """Sliding window mask returns a boolean array."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        cache.set_indices(prefix_indices=[], new_indices=[1, 2, 3])
        mask = cache.make_mask(N=3, window_size=2)
        assert isinstance(mask, mx.array)
        assert mask.shape == (3, 3)
        m = np.array(mask, copy=False)
        # Position 2 can attend to positions 1,2 (window=2) but not position 0
        assert not m[2, 0]
        assert m[2, 1]
        assert m[2, 2]

    def test_mask_return_array(self):
        """return_array=True with N>1 returns explicit boolean mask."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        cache.set_indices(prefix_indices=[1, 2, 3], new_indices=[4, 5])
        mask = cache.make_mask(N=2, return_array=True)
        assert isinstance(mask, mx.array)
        # Shape: (N, offset+N) = (2, 5)
        assert mask.shape == (2, 5)
        m = np.array(mask, copy=False)
        # First query at position 3: can attend to positions 0..3
        assert m[0, 3]
        assert not m[0, 4]
        # Second query at position 4: can attend to all 5
        assert m[1, 4]

    def test_mask_decode_with_return_array(self):
        """N=1 with return_array=True still returns None (mlx-lm behavior)."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        cache.set_indices(prefix_indices=[1, 2, 3, 4, 5], new_indices=[6])
        mask = cache.make_mask(N=1, return_array=True)
        assert mask is None

    def test_state_property(self):
        """state property should return current cached KV."""
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        cache.set_indices(prefix_indices=[], new_indices=[1, 2, 3])

        k, v = self._random_kv(3)
        cache.update_and_fetch(k, v)

        k_state, v_state = cache.state
        assert k_state.shape == (1, self.N_KV_HEADS, 3, self.HEAD_DIM)

    def test_state_empty_returns_none(self):
        pool = self._make_pool()
        cache = PooledKVCache(pool, layer_idx=0)
        cache.set_indices(prefix_indices=[], new_indices=[])
        k_state, v_state = cache.state
        assert k_state is None
        assert v_state is None


# ---- Integration: RadixCache + Allocator ----


class TestRadixCacheAllocatorIntegration:
    """Tests that eviction properly frees slots back to the allocator."""

    def test_eviction_frees_slots(self):
        alloc = TokenPoolAllocator(pool_size=20)
        cache = RadixCache()
        cache.set_allocator(alloc)

        # Allocate slots and insert into cache
        slots1 = alloc.alloc(5)
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5], value=slots1))

        slots2 = alloc.alloc(5)
        cache.insert(InsertParams(key=[6, 7, 8, 9, 10], value=slots2))

        assert alloc.available_size == 10  # 20 - 5 - 5

        # Evict everything
        cache.evict(EvictParams(num_tokens=100))

        # Slots should be returned to allocator
        assert alloc.available_size == 20

    def test_eviction_partial_free(self):
        alloc = TokenPoolAllocator(pool_size=20)
        cache = RadixCache()
        cache.set_allocator(alloc)

        slots1 = alloc.alloc(3)
        cache.insert(InsertParams(key=[1, 2, 3], value=slots1))

        slots2 = alloc.alloc(3)
        cache.insert(InsertParams(key=[4, 5, 6], value=slots2))

        # Touch [4,5,6] to be most recent
        cache.match_prefix([4, 5, 6])

        before = alloc.available_size
        cache.evict(EvictParams(num_tokens=3))
        after = alloc.available_size

        # Should have freed exactly 3 slots (from LRU leaf)
        assert after - before == 3

    def test_backward_compat_no_allocator(self):
        """Existing tests should work without an allocator attached."""
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))
        cache.evict(EvictParams(num_tokens=100))
        # Should not crash
        result = cache.match_prefix([1, 2, 3, 4, 5])
        assert len(result.device_indices) == 0

    def test_two_requests_sharing_prefix(self):
        """Two requests share a prefix via radix cache; allocator tracks correctly."""
        alloc = TokenPoolAllocator(pool_size=50)
        cache = RadixCache()
        cache.set_allocator(alloc)

        # Request 1: tokens [1,2,3,10,11]
        slots_r1 = alloc.alloc(5)
        cache.insert(InsertParams(key=[1, 2, 3, 10, 11], value=slots_r1))

        # Request 2: same prefix, different suffix
        result = cache.match_prefix([1, 2, 3, 20, 21])
        assert len(result.device_indices) == 3  # matched [1,2,3]

        # Only need to allocate for new tokens [20, 21]
        slots_r2_new = alloc.alloc(2)
        cache.insert(
            InsertParams(
                key=[1, 2, 3, 20, 21],
                value=result.device_indices + slots_r2_new,
            )
        )

        assert alloc.available_size == 50 - 5 - 2  # 43


# ---- End-to-end: allocator + pool + cache adapter ----


class TestEndToEnd:
    """Full pipeline: allocate slots, prefill, decode, prefix reuse."""

    NUM_LAYERS = 2
    N_KV_HEADS = 2
    HEAD_DIM = 4
    POOL_SIZE = 32

    def _setup(self):
        alloc = TokenPoolAllocator(self.POOL_SIZE)
        pool = MHATokenPool(
            self.POOL_SIZE,
            self.NUM_LAYERS,
            self.N_KV_HEADS,
            self.HEAD_DIM,
            dtype=mx.float32,
        )
        cache = RadixCache()
        cache.set_allocator(alloc)
        return alloc, pool, cache

    def _random_kv(self, seq_len):
        shape = (1, self.N_KV_HEADS, seq_len, self.HEAD_DIM)
        return mx.random.normal(shape), mx.random.normal(shape)

    def test_prefill_then_decode(self):
        alloc, pool, radix = self._setup()

        # Prefill: allocate 4 tokens
        prompt = [10, 20, 30, 40]
        slots = alloc.alloc(4)

        # Create per-layer caches and run prefill
        caches = [PooledKVCache(pool, i) for i in range(self.NUM_LAYERS)]
        for c in caches:
            c.set_indices(prefix_indices=[], new_indices=slots)
            assert c.offset == 0

        k_in, v_in = self._random_kv(4)
        for c in caches:
            k_out, v_out = c.update_and_fetch(k_in, v_in)
            assert k_out.shape == (1, self.N_KV_HEADS, 4, self.HEAD_DIM)

        # Insert into radix cache
        radix.insert(InsertParams(key=prompt, value=slots))

        # Decode: 1 new token
        new_slot = alloc.alloc(1)
        for c in caches:
            c.set_indices(prefix_indices=slots, new_indices=new_slot)
            assert c.offset == 4

        k_new, v_new = self._random_kv(1)
        for c in caches:
            k_out, v_out = c.update_and_fetch(k_new, v_new)
            assert k_out.shape == (1, self.N_KV_HEADS, 5, self.HEAD_DIM)

    def test_prefix_reuse_across_requests(self):
        alloc, pool, radix = self._setup()

        # Request 1 prefill
        prompt = [10, 20, 30, 40]
        slots = alloc.alloc(4)
        cache0 = PooledKVCache(pool, layer_idx=0)
        cache0.set_indices(prefix_indices=[], new_indices=slots)
        k_in, v_in = self._random_kv(4)
        cache0.update_and_fetch(k_in, v_in)
        radix.insert(InsertParams(key=prompt, value=slots))

        # Request 2: same prefix, new suffix
        result = radix.match_prefix([10, 20, 30, 40, 50, 60])
        assert len(result.device_indices) == 4  # matched all 4 prefix tokens

        new_slots = alloc.alloc(2)
        cache1 = PooledKVCache(pool, layer_idx=0)
        cache1.set_indices(prefix_indices=result.device_indices, new_indices=new_slots)
        assert cache1.offset == 4  # RoPE offset for new tokens

        k_new, v_new = self._random_kv(2)
        k_out, v_out = cache1.update_and_fetch(k_new, v_new)
        assert k_out.shape == (1, self.N_KV_HEADS, 6, self.HEAD_DIM)

        # The prefix portion should match what request 1 stored
        np.testing.assert_allclose(
            np.array(k_out[:, :, :4, :], copy=False),
            np.array(k_in, copy=False),
            atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
