"""Tests for paged KV memory management."""

import pytest

from sglang_mlx.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang_mlx.srt.mem_cache.paged_pool import (
    BlockAllocator,
    block_table_from_token_blocks,
    expand_blocks_to_tokens,
)
from sglang_mlx.srt.mem_cache.radix_cache import RadixCache


class TestBlockAllocator:
    def test_alloc_free_reserved_block(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)
        blocks = alloc.alloc(2)

        assert len(blocks) == 2
        assert alloc.available_blocks == 2

        alloc.free(blocks)
        assert alloc.available_blocks == 4

    def test_retain_requires_allocated_block(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)

        with pytest.raises(ValueError, match="free block"):
            alloc.retain([0])

    def test_retain_refcount_then_free(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)
        block = alloc.alloc(1)[0]

        alloc.retain([block])
        alloc.retain([block])
        assert alloc.ref_count(block) == 2

        alloc.free([block])
        assert alloc.available_blocks == 3
        assert alloc.ref_count(block) == 1

        alloc.free([block])
        assert alloc.available_blocks == 4
        assert alloc.ref_count(block) == 0

    def test_alloc_for_tokens_rounds_up(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)
        assert len(alloc.alloc_for_tokens(17)) == 2


class TestBlockHelpers:
    def test_expand_blocks_to_tokens(self):
        assert expand_blocks_to_tokens([3, 4], 20, 16) == [3] * 16 + [4] * 4

    def test_block_table_from_token_blocks(self):
        token_blocks = [3] * 16 + [4] * 4
        assert block_table_from_token_blocks(token_blocks, 16) == [3, 4]


class TestRadixAllocatorIntegration:
    def test_insert_retains_and_evict_releases_blocks(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)
        cache = RadixCache()
        cache.set_allocator(alloc)

        blocks = alloc.alloc(1)
        cache.insert(InsertParams(key=list(range(16)), value=blocks * 16))

        assert alloc.ref_count(blocks[0]) == 1
        assert alloc.available_blocks == 3

        cache.evict(EvictParams(num_tokens=16))
        assert alloc.ref_count(blocks[0]) == 0
        assert alloc.available_blocks == 4

    def test_split_retains_shared_partial_block(self):
        alloc = BlockAllocator(num_blocks=8, block_size=16)
        cache = RadixCache()
        cache.set_allocator(alloc)

        blocks = alloc.alloc(2)
        cache.insert(
            InsertParams(
                key=list(range(20)),
                value=[blocks[0]] * 16 + [blocks[1]] * 4,
            )
        )

        # Split inside the second physical block.
        new_block = alloc.alloc(1)[0]
        cache.insert(
            InsertParams(
                key=list(range(18)) + [100, 101],
                value=[blocks[0]] * 16 + [new_block] * 4,
            )
        )

        assert alloc.ref_count(blocks[1]) == 2

        cache.evict(EvictParams(num_tokens=100))
        assert alloc.ref_count(blocks[1]) == 0

    def test_match_split_keeps_eviction_leaf_state_coherent(self):
        alloc = BlockAllocator(num_blocks=8, block_size=16)
        cache = RadixCache()
        cache.set_allocator(alloc)

        blocks = alloc.alloc(2)
        cache.insert(
            InsertParams(
                key=list(range(32)),
                value=[blocks[0]] * 16 + [blocks[1]] * 16,
            )
        )

        cache.match_prefix(list(range(18)) + [100])

        assert all(not node.children for node in cache.evictable_leaves)

        cache.evict(EvictParams(num_tokens=100))
        assert alloc.available_blocks == alloc.num_blocks
