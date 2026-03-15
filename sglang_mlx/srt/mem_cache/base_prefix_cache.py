"""
Base class and parameter types for prefix caching.

Adapted from sglang/srt/mem_cache/base_prefix_cache.py.
Key difference: uses plain Python lists for indices instead of torch.Tensor,
since MLX doesn't need the same GPU tensor indirection layer.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, NamedTuple


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices: Indices of the KV cache slots matched by common prefix.
        last_node: The last TreeNode that was matched.
    """

    device_indices: list[int]
    last_node: Any


@dataclasses.dataclass
class InsertParams:
    """Parameters for insert operation."""

    key: list[int]  # token ids
    value: list[int] | None = None  # KV pool indices

    priority: int = 0


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation."""

    prefix_len: int


@dataclasses.dataclass
class EvictParams:
    """Parameters for evict operation."""

    num_tokens: int


@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation."""

    num_tokens_evicted: int = 0


class BasePrefixCache(ABC):
    """Base class for prefix caches."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, key: list[int]) -> MatchResult:
        pass

    @abstractmethod
    def insert(self, params: InsertParams) -> InsertResult:
        pass

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any):
        pass

    @abstractmethod
    def dec_lock_ref(self, node: Any):
        pass

    def evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()
