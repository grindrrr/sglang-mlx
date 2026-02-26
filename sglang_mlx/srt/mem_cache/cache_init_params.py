"""
Initialization parameters for the radix cache.

Adapted from sglang/srt/mem_cache/cache_init_params.py.
Simplified for MLX: no torch distributed, no EAGLE/HiCache/SWA specifics.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class CacheInitParams:
    disable: bool = False
    page_size: int = 1
    eviction_policy: str = "lru"
