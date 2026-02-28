"""
KV memory pool for MLX-based serving.

Provides the bridge between radix tree (which stores slot indices)
and mlx-lm's model attention layers (which expect contiguous KV arrays).

Components:
- TokenPoolAllocator: pure-Python free list for slot management
- MHATokenPool: pre-allocated MLX buffers, one K/V pair per layer
- PooledKVCache: per-request adapter implementing mlx-lm's cache interface

Design: pool + gather strategy. Persistent memory is shared across requests
via the radix cache; a gather copy assembles contiguous arrays for each
attention call. This is accepted as a temporary cost — a future custom Metal
kernel can fuse gather + SDPA to eliminate the copy.
"""

from __future__ import annotations

import mlx.core as mx

# ---------------------------------------------------------------------------
# TokenPoolAllocator — pure Python free list, no MLX dependency
# ---------------------------------------------------------------------------


class TokenPoolAllocator:
    """Manages allocation of token slots in the KV pool.

    Slot 0 is reserved as padding and never allocated.
    """

    def __init__(self, pool_size: int):
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        self._pool_size = pool_size
        # Slots 1..pool_size are allocatable; slot 0 is reserved padding
        self._free_slots: list[int] = list(range(pool_size, 0, -1))  # stack order

    def alloc(self, n: int) -> list[int]:
        """Allocate *n* contiguous-in-list slots.

        Slots are not necessarily contiguous in the underlying pool.

        Returns a list of *n* slot indices.
        Raises MemoryError if fewer than *n* slots are available.
        """
        if n <= 0:
            return []
        if n > len(self._free_slots):
            raise MemoryError(
                f"Cannot allocate {n} slots: only {len(self._free_slots)} available"
            )
        allocated = self._free_slots[-n:]
        del self._free_slots[-n:]
        return allocated

    def free(self, indices: list[int]) -> None:
        """Return *indices* to the free list."""
        self._free_slots.extend(indices)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def clear(self) -> None:
        """Reset all slots to free."""
        self._free_slots = list(range(self._pool_size, 0, -1))


# ---------------------------------------------------------------------------
# MHATokenPool — pre-allocated MLX KV buffers
# ---------------------------------------------------------------------------


class MHATokenPool:
    """Pre-allocated KV cache buffers for all layers.

    Buffer layout per layer: ``(pool_size + 1, n_kv_heads, head_dim)``
    where +1 accounts for the reserved slot 0 padding.
    Token-first layout matches sglang convention; transpositions happen
    in store/fetch to convert to/from mlx-lm's ``(B, n_kv_heads, S, head_dim)`` format.
    """

    # TODO(quant): FP8/quantized KV cache support.
    # TODO(opt): GQA/MQA-specific memory layout optimizations.

    def __init__(
        self,
        pool_size: int,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        shape = (pool_size + 1, n_kv_heads, head_dim)  # +1 for slot 0 padding
        self.k_buffer: list[mx.array] = [
            mx.zeros(shape, dtype=dtype) for _ in range(num_layers)
        ]
        self.v_buffer: list[mx.array] = [
            mx.zeros(shape, dtype=dtype) for _ in range(num_layers)
        ]

    # TODO(perf): Custom Metal paged attention kernel to avoid gather copies.
    #   A Metal kernel could do SDPA directly on scattered pool slots,
    #   eliminating the O(seq_len) gather per layer per decode step.

    # TODO(perf): Contiguous allocation optimization. When slots are contiguous,
    #   use slice indexing (zero-copy view) instead of fancy indexing (copy).

    def store(
        self, layer_idx: int, indices: list[int], keys: mx.array, values: mx.array
    ) -> None:
        """Write KV data into the pool at the given slot indices.

        Args:
            layer_idx: Which transformer layer.
            indices: Pool slot indices, length S.
            keys: Shape ``(1, n_kv_heads, S, head_dim)`` — mlx-lm format.
            values: Shape ``(1, n_kv_heads, S, head_dim)`` — mlx-lm format.
        """
        # Strip batch dim: (1, H, S, D) -> (H, S, D)
        k = keys[0]
        v = values[0]
        # Transpose to token-first: (H, S, D) -> (S, H, D)
        k = mx.transpose(k, axes=(1, 0, 2))
        v = mx.transpose(v, axes=(1, 0, 2))
        # Scatter write into pool
        idx = mx.array(indices)
        self.k_buffer[layer_idx][idx] = k
        self.v_buffer[layer_idx][idx] = v

    def fetch(self, layer_idx: int, indices: list[int]) -> tuple[mx.array, mx.array]:
        """Gather KV data from the pool and return in mlx-lm format.

        Args:
            layer_idx: Which transformer layer.
            indices: Pool slot indices, length T.

        Returns:
            (keys, values) each of shape ``(1, n_kv_heads, T, head_dim)``.
        """
        idx = mx.array(indices)
        # Gather: (T, H, D)
        k = self.k_buffer[layer_idx][idx]
        v = self.v_buffer[layer_idx][idx]
        # Transpose to head-first: (T, H, D) -> (H, T, D)
        k = mx.transpose(k, axes=(1, 0, 2))
        v = mx.transpose(v, axes=(1, 0, 2))
        # Add batch dim: (H, T, D) -> (1, H, T, D)
        k = mx.expand_dims(k, axis=0)
        v = mx.expand_dims(v, axis=0)
        return k, v


# ---------------------------------------------------------------------------
# PooledKVCache — per-request, per-layer adapter for mlx-lm's cache interface
# ---------------------------------------------------------------------------


class PooledKVCache:
    """Adapter that makes the shared KV pool look like mlx-lm's per-request KVCache.

    Models call ``cache.update_and_fetch(keys, values)`` in each attention layer.
    This class stores new tokens into the pool and gathers all (prefix + new)
    tokens back as contiguous arrays.
    """

    # TODO(batch): Support B>1 for continuous batching. Currently assumes B=1.
    #   For batched forward passes, each request has different pool indices.
    #   The scheduler/model runner will need to coordinate this.

    def __init__(self, pool: MHATokenPool, layer_idx: int):
        self._pool = pool
        self._layer_idx = layer_idx
        self._prefix_indices: list[int] = []  # already in pool (from radix match)
        self._new_indices: list[int] = []  # allocated for new tokens
        self._store_offset: int = 0  # tracks how many new tokens stored so far

    def set_indices(self, prefix_indices: list[int], new_indices: list[int]) -> None:
        """Called by the scheduler before forward pass to configure this cache layer.

        Args:
            prefix_indices: Slots already populated in the pool (from radix match).
            new_indices: Freshly allocated slots for new tokens to be computed.
        """
        self._prefix_indices = list(prefix_indices)
        self._new_indices = list(new_indices)
        self._store_offset = 0

    @property
    def offset(self) -> int:
        """RoPE position offset for new tokens.

        The prefix tokens are already computed and stored; new tokens start
        at this position in the sequence.
        """
        return len(self._prefix_indices)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Store new KV and gather all (prefix + new) KV.

        Args:
            keys: Shape ``(1, n_kv_heads, N, head_dim)`` where N is the number
                  of new tokens being processed in this forward step.
            values: Same shape as keys.

        Returns:
            (keys, values) each of shape ``(1, n_kv_heads, total_len, head_dim)``
            where total_len = len(prefix_indices) + len(new_indices).
        """
        # Number of new tokens in this call
        n_new = keys.shape[2]

        # Determine which new slots to write to
        store_indices = self._new_indices[
            self._store_offset : self._store_offset + n_new
        ]
        self._store_offset += n_new

        # Store new K,V into pool
        self._pool.store(self._layer_idx, store_indices, keys, values)

        # Gather ALL K,V (prefix + all new tokens stored so far)
        all_indices = self._prefix_indices + self._new_indices[: self._store_offset]
        return self._pool.fetch(self._layer_idx, all_indices)

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        """Return current cached KV state (for compatibility)."""
        all_indices = self._prefix_indices + self._new_indices[: self._store_offset]
        if not all_indices:
            return None, None
        return self._pool.fetch(self._layer_idx, all_indices)

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        """Create attention mask, delegating to mlx-lm's implementation.

        Called by the model's attention layers via
        ``base.create_attention_mask(h, cache)``.

        Args:
            N: Number of new query tokens.
            return_array: If True and N > 1, return an explicit boolean mask
                instead of the ``"causal"`` shorthand.
            window_size: Sliding window size (None = full attention).

        Returns:
            ``None`` for single-token decode, ``"causal"`` for standard prefill,
            or a boolean ``mx.array`` mask for windowed / explicit cases.
        """
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(
            N, offset=self.offset, return_array=return_array, window_size=window_size
        )
