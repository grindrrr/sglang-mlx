"""
The radix tree data structure for managing the KV cache.

Ported from sglang/srt/mem_cache/radix_cache.py.

Key differences from upstream (torch/CUDA):
- Uses plain Python lists for KV pool indices instead of torch.Tensor
- No EAGLE bigram support (not needed for MLX v1)
- No HiCache host offloading (Apple Silicon has unified memory)
- No KV event queue / disaggregation (single-machine only)
- No page_size > 1 support yet (can add later)

The core radix tree algorithm is identical to upstream.
"""

from __future__ import annotations

import heapq
import logging
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_mlx.srt.mem_cache.paged_pool import BlockAllocator

from sglang_mlx.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchResult,
)
from sglang_mlx.srt.mem_cache.cache_init_params import CacheInitParams
from sglang_mlx.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang_mlx.srt.mem_cache.paged_pool import BlockAllocator


class TreeNode:
    counter = 0

    def __init__(self, id: int | None = None, priority: int = 0):
        self.children: dict[int, TreeNode] = defaultdict(TreeNode)
        self.parent: TreeNode | None = None
        self.key: list[int] = []  # token ids stored in this node
        self.value: list[int] | None = None  # KV pool indices
        self.lock_ref: int = 0
        self.last_access_time: float = time.monotonic()
        self.creation_time: float = time.monotonic()
        self.hit_count: int = 0
        self.priority: int = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self) -> bool:
        return self.value is None

    def __lt__(self, other: TreeNode) -> bool:
        return self.last_access_time < other.last_access_time

    def __repr__(self) -> str:
        preview = self.key[:10]
        suffix = "..." if len(self.key) > 10 else ""
        return (
            f"TreeNode(id={self.id}, key={preview}{suffix}, lock_ref={self.lock_ref})"
        )


def _key_match(key0: list[int], key1: list[int]) -> int:
    """Match two token id sequences, return length of common prefix."""
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams | None = None):
        if params is None:
            params = CacheInitParams()

        self.disable = params.disable
        self.page_size = params.page_size
        self.eviction_policy = params.eviction_policy.lower()

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy = PriorityStrategy()
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

        self.allocator: BlockAllocator | None = None
        self.evictable_leaves: set[TreeNode] = set()
        self.reset()

    def reset(self):
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1  # root is always locked
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()

    def set_allocator(self, allocator: BlockAllocator) -> None:
        """Attach a pool allocator so eviction releases committed blocks."""
        self.allocator = allocator

    ##### Public API #####

    def match_prefix(self, key: list[int]) -> MatchResult:
        if self.disable or len(key) == 0:
            return MatchResult(device_indices=[], last_node=self.root_node)

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            # Flatten list of lists into single list
            indices = []
            for v in value:
                indices.extend(v)
        else:
            indices = []

        return MatchResult(device_indices=indices, last_node=last_node)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority or 0

        if value is None:
            # When no explicit value, use token ids as placeholder indices
            value = list(key)

        prefix_len = self._insert_helper(self.root_node, key, value, priority)
        return InsertResult(prefix_len=prefix_len)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if self.allocator is not None:
                self.allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode):
        """Lock a node and all ancestors — prevents eviction."""
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        """Unlock a node and all ancestors — allows eviction."""
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    def total_size(self):
        return self._total_size_helper()

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, node: TreeNode, key: list[int]
    ) -> tuple[list[list[int]], TreeNode]:
        access_time = time.monotonic()
        node.last_access_time = access_time

        if not key:
            return [], node

        child_key = key[0]
        value = []

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = _key_match(child.key, key)

            if prefix_len < len(child.key):
                # Partial match — need to split
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                # Full match of this node
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key[0]

        return value, node

    def _split_node(self, key: list[int], child: TreeNode, split_len: int) -> TreeNode:
        """Split a node at split_len.

        Before: parent -> child[A,B,C,D,E]
        After:  parent -> new_node[A,B,C] -> child[D,E]

        Both new_node and child retain their portion of the KV indices.
        """
        new_node = TreeNode(priority=child.priority)
        new_node.children = {child.key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]  # slice indices

        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]  # slice indices

        if self.allocator is not None:
            # The original node owned one reference for each unique block in
            # child.value before the split. After splitting, only blocks that
            # appear on both sides need an extra reference for the new parent.
            overlap = list(set(new_node.value or []) & set(child.value or []))
            if overlap:
                self.allocator.retain(overlap)

        new_node.parent.children[key[0]] = new_node
        self._update_leaf_status(child)
        self._update_leaf_status(new_node)
        return new_node

    def _insert_helper(
        self, node: TreeNode, key: list[int], value: list[int], priority: int = 0
    ) -> int:
        if priority is None:
            priority = 0

        access_time = time.monotonic()
        node.last_access_time = access_time
        node.priority = max(node.priority, priority)

        if len(key) == 0:
            return 0

        child_key = key[0]
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = _key_match(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                node = new_node
            else:
                node.priority = max(node.priority, priority)

            if len(key):
                child_key = key[0]

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = list(key)  # copy
            new_node.value = list(value)  # copy
            node.children[child_key] = new_node
            if self.allocator is not None:
                self.allocator.retain(new_node.value)
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

        return total_prefix_length

    def _delete_leaf(self, node: TreeNode):
        key = node.key[0]
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        """Maintain the evictable_leaves set incrementally."""
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        # A node is an evictable leaf if all its children are evicted (or it has none)
        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self) -> int:
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value) if current_node.value else 0
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _print_helper(self, node: TreeNode, indent: int):
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))


if __name__ == "__main__":
    tree = RadixCache()

    tree.insert(InsertParams(key=[1, 2, 3]))
    tree.insert(InsertParams(key=[1, 2, 3]))
    tree.insert(InsertParams(key=[1, 2, 4, 5]))
    tree.insert(InsertParams(key=[1, 2, 4, 5, 6, 7]))
    tree.insert(InsertParams(key=[8, 9, 10, 11, 12]))
    tree.pretty_print()

    result = tree.match_prefix([1, 2, 3, 13, 14])
    print(f"match_prefix([1,2,3,13,14]): indices={result.device_indices}")
