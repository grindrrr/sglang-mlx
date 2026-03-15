"""Tests for RadixCache."""

import pytest

from sglang_mlx.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang_mlx.srt.mem_cache.radix_cache import RadixCache


class TestBasicOperations:
    def test_insert_and_exact_match(self):
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))

        result = cache.match_prefix([1, 2, 3, 4, 5])
        assert len(result.device_indices) == 5

    def test_prefix_match(self):
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))

        result = cache.match_prefix([1, 2, 3, 4, 5, 6, 7])
        assert len(result.device_indices) == 5

    def test_no_match(self):
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3]))

        result = cache.match_prefix([9, 9, 9])
        assert len(result.device_indices) == 0

    def test_empty_key(self):
        cache = RadixCache()
        result = cache.match_prefix([])
        assert len(result.device_indices) == 0


class TestSharedPrefix:
    def test_shared_system_prompt(self):
        cache = RadixCache()
        system = [1, 2, 3, 4, 5]

        # First request: system + question1
        cache.insert(InsertParams(key=system + [10, 11, 12]))

        # Second request: system + question2 — should match system prefix
        result = cache.match_prefix(system + [20, 21, 22])
        assert len(result.device_indices) == 5  # matched system prompt

    def test_multi_turn_reuse(self):
        cache = RadixCache()

        turn1 = [1, 2, 3, 10, 11]
        cache.insert(InsertParams(key=turn1))

        turn2 = turn1 + [50, 51, 20, 21]
        result = cache.match_prefix(turn2)
        assert len(result.device_indices) == len(turn1)


class TestNodeSplit:
    def test_split_preserves_kv(self):
        """After split, both halves should have valid KV indices."""
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))

        # This forces a split at position 3
        cache.insert(InsertParams(key=[1, 2, 3, 6, 7]))

        # Original sequence should still fully match
        result = cache.match_prefix([1, 2, 3, 4, 5])
        assert len(result.device_indices) == 5

        # New branch should also match
        result = cache.match_prefix([1, 2, 3, 6, 7])
        assert len(result.device_indices) == 5

        # Shared prefix alone
        result = cache.match_prefix([1, 2, 3])
        assert len(result.device_indices) == 3

    def test_split_at_different_positions(self):
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))

        # Split at position 1
        cache.insert(InsertParams(key=[1, 9, 9]))

        result = cache.match_prefix([1, 2, 3, 4, 5])
        assert len(result.device_indices) == 5

        result = cache.match_prefix([1, 9, 9])
        assert len(result.device_indices) == 3


class TestLockRef:
    """Reference counting prevents eviction of in-use nodes."""

    def test_lock_prevents_eviction(self):
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))

        result = cache.match_prefix([1, 2, 3, 4, 5])
        cache.inc_lock_ref(result.last_node)

        # Try to evict — should not evict locked node
        cache.evict(EvictParams(num_tokens=100))

        # Should still be there
        result2 = cache.match_prefix([1, 2, 3, 4, 5])
        assert len(result2.device_indices) == 5

    def test_unlock_allows_eviction(self):
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))

        result = cache.match_prefix([1, 2, 3, 4, 5])
        cache.inc_lock_ref(result.last_node)
        cache.dec_lock_ref(result.last_node)

        # Now eviction should work
        cache.evict(EvictParams(num_tokens=100))

        result2 = cache.match_prefix([1, 2, 3, 4, 5])
        assert len(result2.device_indices) == 0

    def test_lock_walks_to_root(self):
        """Locking a deep node must also lock ancestors."""
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))
        cache.insert(InsertParams(key=[1, 2, 3, 6, 7]))  # splits at position 3

        # Lock the [4,5] branch
        result = cache.match_prefix([1, 2, 3, 4, 5])
        cache.inc_lock_ref(result.last_node)

        # [1,2,3] ancestor should also be locked (not evictable)
        cache.evict(EvictParams(num_tokens=100))

        # [6,7] leaf is evictable, but [1,2,3] shared prefix should survive
        result_shared = cache.match_prefix([1, 2, 3])
        assert len(result_shared.device_indices) == 3


class TestEviction:
    def test_lru_eviction(self):
        cache = RadixCache()

        cache.insert(InsertParams(key=[1, 2, 3]))
        cache.insert(InsertParams(key=[4, 5, 6]))
        cache.insert(InsertParams(key=[7, 8, 9]))

        # Touch [4,5,6] to make it most recently used
        cache.match_prefix([4, 5, 6])

        # Evict 3 tokens — should evict LRU ([1,2,3])
        cache.evict(EvictParams(num_tokens=3))

        # [1,2,3] should be evicted
        result = cache.match_prefix([1, 2, 3])
        assert len(result.device_indices) == 0

        # [4,5,6] should survive (was accessed more recently)
        result = cache.match_prefix([4, 5, 6])
        assert len(result.device_indices) == 3

    def test_evict_cascades_to_parent(self):
        """When a leaf is evicted, its parent may become evictable."""
        cache = RadixCache()
        cache.insert(InsertParams(key=[1, 2, 3, 4, 5]))
        cache.insert(InsertParams(key=[1, 2, 3, 6, 7]))

        # Evict enough to clear both leaves and the shared prefix
        cache.evict(EvictParams(num_tokens=100))

        result = cache.match_prefix([1, 2, 3])
        assert len(result.device_indices) == 0


class TestFewShotScenario:
    def test_few_shot_reuse(self):
        cache = RadixCache()
        examples = list(range(100))  # shared few-shot prefix

        total_reused = 0
        for i in range(10):
            query = examples + [1000 + i, 1001 + i]
            result = cache.match_prefix(query)
            total_reused += len(result.device_indices)
            cache.insert(InsertParams(key=query))

        # First query has 0 reuse, remaining 9 should reuse 100 tokens each
        assert total_reused == 9 * 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
