"""
paged_pool.py — block-based KV cache for paged attention kernels.

Provides the memory management layer that matches vLLM-style paged attention:
- BlockAllocator: allocates fixed-size blocks (groups of tokens), not individual slots.
- PagedMHAPool: pre-allocated KV arrays in the layout expected by paged_attention_v1/v2.
- PagedKVCache: per-request, per-layer adapter that stores via reshape_and_cache
  and provides block_tables + context_lens for paged_attention_v1/v2.

Layout (matches paged_attention_v1/v2 and reshape_and_cache Metal kernels):
  key_cache[layer]: (num_blocks, n_kv_heads, head_dim//x, block_size, x)
  val_cache[layer]: (num_blocks, n_kv_heads, head_dim, block_size)

Prefix sharing (radix cache integration):
  The radix cache stores block IDs at token granularity (one entry per token,
  repeated for all tokens sharing a block). This preserves token-level prefix
  matching while allowing block-level allocation.

Workflow per decode step (layer l):
  1. PagedKVCache.store(keys, values, layer_idx=l)
       → calls reshape_and_cache(keys, values, pool.key_cache[l],
                                 pool.val_cache[l], slot_mapping)
       → writes new token KV into the correct block positions
  2. model patcher calls PagedKVCache.compute_paged_attn(q, layer_idx=l, scale)
       → calls paged_attention_v1(q, pool.key_cache[l], pool.val_cache[l],
                                   block_tables, context_lens, ...)
"""

from __future__ import annotations

import math

mx = None

# ── Load the Metal extension ───────────────────────────────────────────────────

try:
    from sglang_mlx import _ext
except ImportError:
    _ext = None  # extension not built; Metal ops unavailable (pure-Python path only)


def _mx():
    """Import mlx.core lazily so allocator/radix tests do not require Metal."""
    global mx
    if mx is None:
        import mlx.core as _mx_core

        mx = _mx_core
    return mx


def _require_ext() -> None:
    if _ext is None:
        raise ImportError(
            "sglang_mlx._ext not found. Build it first:\n"
            "  pip install -e .[dev]\n"
            "or manually:\n"
            "  cmake -B build . && cmake --build build"
        )


def _key_x_for_dtype(dtype: mx.Dtype) -> int:
    """Return the Metal key-cache interleave factor for a cache dtype."""
    mx_mod = _mx()
    if dtype in (mx_mod.float16, mx_mod.bfloat16):
        return 8
    if dtype == mx_mod.float32:
        return 4
    raise ValueError(f"Unsupported KV cache dtype for paged layout: {dtype}")


def block_table_from_token_blocks(
    token_blocks: list[int],
    block_size: int,
) -> list[int]:
    """Convert token-granular block IDs into one physical block per logical page."""
    if not token_blocks:
        return []
    return [token_blocks[i] for i in range(0, len(token_blocks), block_size)]


def expand_blocks_to_tokens(
    block_ids: list[int],
    n_tokens: int,
    block_size: int,
) -> list[int]:
    """Expand physical block IDs to token-granular radix values."""
    values: list[int] = []
    for i in range(n_tokens):
        values.append(block_ids[i // block_size])
    return values


# ── BlockAllocator ────────────────────────────────────────────────────────────


class BlockAllocator:
    """Free-list allocator for fixed-size KV cache blocks.

    A block holds *block_size* consecutive token positions across all heads.
    Tokens within a block at layer l, kv-head h, position p are stored at
    ``key_cache[l][block_id, h, :, p, :]`` and ``val_cache[l][block_id, h, :, p]``.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0 or (block_size & (block_size - 1)):
            raise ValueError(
                f"block_size must be a positive power of 2, got {block_size}"
            )
        self._num_blocks = num_blocks
        self._block_size = block_size
        # Stack: pop from the back for O(1) alloc/free
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._in_free: set[int] = set(self._free)
        # Allocation removes a block from the free list. Radix insertion calls
        # retain() for committed blocks; eviction calls free() to release refs.
        self._ref_counts: list[int] = [0] * num_blocks

    # ── Allocation ────────────────────────────────────────────────────────────

    def alloc(self, n: int) -> list[int]:
        """Allocate exactly *n* blocks. Raises MemoryError if unavailable."""
        if n <= 0:
            return []
        if n > len(self._free):
            raise MemoryError(
                f"Cannot allocate {n} blocks: only {len(self._free)} available"
            )
        blocks = self._free[-n:]
        del self._free[-n:]
        for block_id in blocks:
            self._in_free.remove(block_id)
        return blocks

    def alloc_for_tokens(self, n_tokens: int) -> list[int]:
        """Allocate enough blocks to hold *n_tokens* tokens."""
        n_blocks = math.ceil(n_tokens / self._block_size)
        return self.alloc(n_blocks)

    def free(self, block_ids: list[int]) -> None:
        """Release one reference to each block and return unreferenced blocks.

        Deduplicates automatically — safe to call with repeated IDs (which
        occurs when block IDs are stored at token granularity in the radix cache).
        """
        for block_id in dict.fromkeys(block_ids):
            self._check_block_id(block_id)
            if block_id in self._in_free:
                continue
            if self._ref_counts[block_id] > 0:
                self._ref_counts[block_id] -= 1
            if self._ref_counts[block_id] == 0:
                self._free.append(block_id)
                self._in_free.add(block_id)

    def retain(self, block_ids: list[int]) -> None:
        """Add one committed reference to each unique block ID."""
        for block_id in dict.fromkeys(block_ids):
            self._check_block_id(block_id)
            if block_id in self._in_free:
                raise ValueError(f"Cannot retain free block {block_id}")
            self._ref_counts[block_id] += 1

    def _check_block_id(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self._num_blocks:
            raise ValueError(f"block_id out of range: {block_id}")

    def clear(self) -> None:
        self._free = list(range(self._num_blocks - 1, -1, -1))
        self._in_free = set(self._free)
        self._ref_counts = [0] * self._num_blocks

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def available_blocks(self) -> int:
        return len(self._free)

    @property
    def available_size(self) -> int:
        """Compatibility alias used by older runner/tests."""
        return self.available_blocks

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def ref_count(self, block_id: int) -> int:
        self._check_block_id(block_id)
        return self._ref_counts[block_id]


# ── PagedMHAPool ──────────────────────────────────────────────────────────────


class PagedMHAPool:
    """Pre-allocated KV cache in the paged attention kernel layout.

    Key cache:   ``(num_blocks, n_kv_heads, head_dim // x, block_size, x)``
    Value cache: ``(num_blocks, n_kv_heads, head_dim, block_size)``

    where ``x = 16 / sizeof(dtype)``.
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: mx.Dtype | None = None,
    ):
        mx_mod = _mx()
        if dtype is None:
            dtype = mx_mod.float16
        key_x = _key_x_for_dtype(dtype)
        if head_dim % key_x != 0:
            raise ValueError(
                f"head_dim must be divisible by {key_x} for key-cache x-interleaving, "
                f"got head_dim={head_dim}"
            )

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.key_x = key_x

        key_shape = (num_blocks, n_kv_heads, head_dim // key_x, block_size, key_x)
        val_shape = (num_blocks, n_kv_heads, head_dim, block_size)

        self.key_cache: list[mx.array] = [
            mx_mod.zeros(key_shape, dtype=dtype) for _ in range(num_layers)
        ]
        self.val_cache: list[mx.array] = [
            mx_mod.zeros(val_shape, dtype=dtype) for _ in range(num_layers)
        ]

    def store(
        self,
        layer_idx: int,
        keys: mx.array,
        values: mx.array,
        slot_mapping: mx.array,
    ) -> None:
        """Write new tokens into the block cache via reshape_and_cache.

        Args:
            layer_idx:    Transformer layer index.
            keys:         Shape ``(n_new_tokens, n_kv_heads, head_dim)``.
            values:       Same shape as keys.
            slot_mapping: Shape ``(n_new_tokens,)`` int32.
                          Each entry = ``block_id * block_size + position_in_block``.
        """
        _require_ext()
        new_kc, new_vc = _ext.reshape_and_cache(
            keys,
            values,
            self.key_cache[layer_idx],
            self.val_cache[layer_idx],
            slot_mapping,
        )
        self.key_cache[layer_idx] = new_kc
        self.val_cache[layer_idx] = new_vc


# ── BatchedPagedKVCache ───────────────────────────────────────────────────────


class BatchedPagedKVCache:
    """Per-layer adapter for batched decode across N sequences.

    Wraps N ``PagedKVCache`` objects (one per sequence) for a single layer and
    provides two batched operations:

    * ``store_each(keys, values, layer_idx)``       — scatter-write each sequence's
      new token into its own paged blocks.  O(N) kernel launches.

    * ``compute_paged_attn_batch(q, layer_idx, scale)``  — run ``paged_attention_v1``
      once for all N sequences using padded ``(N, max_blocks)`` block tables.
      Single kernel launch regardless of batch size.

    Used by ``ModelRunner.forward_decode_batch`` to enable static batching.
    """

    def __init__(self, caches: list[PagedKVCache]):
        if not caches:
            raise ValueError("BatchedPagedKVCache requires at least one cache")
        self._caches = list(caches)

    @property
    def offsets(self) -> list[int]:
        """RoPE position offsets — one per sequence."""
        return [c.offset for c in self._caches]

    @property
    def _layer_idx(self) -> int:
        return self._caches[0]._layer_idx

    def store_each(self, keys: mx.array, values: mx.array, layer_idx: int) -> None:
        """Scatter-write each sequence's new token K/V into its paged blocks.

        Args:
            keys:      ``(B, n_kv_heads, 1, head_dim)`` — one row per sequence.
            values:    Same shape.
            layer_idx: Transformer layer index.
        """
        for i, cache in enumerate(self._caches):
            cache.store(keys[i : i + 1], values[i : i + 1], layer_idx)

    def compute_paged_attn_batch(
        self,
        q: mx.array,
        layer_idx: int,
        scale: float,
    ) -> mx.array:
        """Run paged_attention_v1 for all N sequences in a single kernel call.

        Args:
            q:         ``(B, n_heads, 1, head_dim)`` — one query per sequence.
            layer_idx: Transformer layer index.
            scale:     Attention scale, typically ``1 / sqrt(head_dim)``.

        Returns:
            ``(B, n_heads, 1, head_dim)``
        """
        _require_ext()
        pool = self._caches[0]._pool

        # Build (B, max_blocks) block_tables and (B,) context_lens
        max_blocks = max(len(c._all_blocks) for c in self._caches)
        block_table_rows: list[list[int]] = []
        ctx_lens: list[int] = []
        for c in self._caches:
            blocks = c._all_blocks
            block_table_rows.append(blocks + [0] * (max_blocks - len(blocks)))
            ctx_lens.append(c._total_tokens)

        block_tables = mx.array(block_table_rows, dtype=mx.int32)  # (B, max_blocks)
        context_lens = mx.array(ctx_lens, dtype=mx.int32)  # (B,)

        # (B, n_heads, 1, head_dim) → (B, n_heads, head_dim)
        q_3d = q[:, :, 0, :]

        out_dtype = q_3d.dtype
        pool_dtype = pool.dtype
        if q_3d.dtype != pool_dtype:
            q_3d = q_3d.astype(pool_dtype)

        out_3d = _ext.paged_attention_v1(
            q_3d,
            pool.key_cache[layer_idx],
            pool.val_cache[layer_idx],
            block_tables,
            context_lens,
            pool.n_kv_heads,
            scale,
            pool.block_size,
            max(ctx_lens),
        )

        if out_dtype != pool_dtype:
            out_3d = out_3d.astype(out_dtype)

        return out_3d[:, :, None, :]  # (B, n_heads, 1, head_dim)


# ── PagedKVCache ──────────────────────────────────────────────────────────────


class PagedKVCache:
    """Per-request, per-layer adapter for the paged attention kernels.

    Maintains the sequence's block list and provides two operations:

    * ``store(keys, values, layer_idx)``  — write new token KV into the block cache
      via ``reshape_and_cache``.  O(n_new_tokens), no gather.

    * ``compute_paged_attn(q, layer_idx, scale, max_seq_len)``  — compute decode
      attention via ``paged_attention_v1``.  O(seq_len) on GPU, no gather copy.

    The ``update_and_fetch`` shim is provided for *drop-in compatibility* with
    mlx-lm models that haven't been patched.  It performs the same gather-copy
    as the old contiguous-cache fetch path — use ``store`` + ``compute_paged_attn``
    (via a model patcher) to avoid the gather entirely.
    """

    def __init__(self, pool: PagedMHAPool, layer_idx: int):
        self._pool = pool
        self._layer_idx = layer_idx
        self._prefix_blocks: list[int] = []  # blocks from radix cache hit
        self._new_blocks: list[int] = []  # freshly allocated blocks
        self._prefix_tokens: int = 0  # #tokens in prefix
        self._new_tokens_stored: int = 0  # #new tokens written so far

    def set_blocks(
        self,
        prefix_blocks: list[int],
        new_blocks: list[int],
        prefix_tokens: int,
    ) -> None:
        """Configure this cache layer before a forward pass.

        Args:
            prefix_blocks: Physical block IDs populated from a radix cache hit.
            new_blocks:    Freshly allocated physical block IDs for new tokens.
            prefix_tokens: Number of tokens already in the prefix.
        """
        self._prefix_blocks = list(prefix_blocks)
        self._new_blocks = list(new_blocks)
        self._prefix_tokens = prefix_tokens
        self._new_tokens_stored = 0

    # ── RoPE offset (mlx-lm cache interface) ─────────────────────────────────

    @property
    def offset(self) -> int:
        """RoPE position offset: skip prefix tokens that are already cached."""
        return self._prefix_tokens

    # ── Block table helpers ───────────────────────────────────────────────────

    @property
    def _all_blocks(self) -> list[int]:
        return self._prefix_blocks + self._new_blocks

    @property
    def _total_tokens(self) -> int:
        return self._prefix_tokens + self._new_tokens_stored

    def _make_block_tables(self) -> mx.array:
        """(1, n_blocks) int32 — physical block IDs for this sequence."""
        return mx.array([self._all_blocks], dtype=mx.int32)

    def _make_context_lens(self) -> mx.array:
        """(1,) int32 — total tokens in the sequence."""
        return mx.array([self._total_tokens], dtype=mx.int32)

    def _slot_mapping_for_new(self, n_new: int) -> mx.array:
        """Slot indices for the next *n_new* tokens.

        Slot = block_id * block_size + position_within_block.
        """
        bs = self._pool.block_size
        start = self._prefix_tokens + self._new_tokens_stored
        slots = []
        for i in range(n_new):
            pos = start + i
            logical_block = pos // bs
            pos_in_block = pos % bs
            if logical_block >= len(self._all_blocks):
                raise IndexError(
                    "PagedKVCache has insufficient blocks for new tokens: "
                    f"need logical block {logical_block}, have {len(self._all_blocks)}"
                )
            block_id = self._all_blocks[logical_block]
            slots.append(block_id * bs + pos_in_block)
        return mx.array(slots, dtype=mx.int64)

    # ── Write path ────────────────────────────────────────────────────────────

    def store(self, keys: mx.array, values: mx.array, layer_idx: int) -> None:
        """Write new token KV into the block cache via reshape_and_cache.

        Args:
            keys:      Shape ``(1, n_kv_heads, n_new, head_dim)`` — mlx-lm format.
            values:    Same shape as keys.
            layer_idx: Transformer layer index.
        """
        n_new = keys.shape[2]
        if n_new == 0:
            return
        k = mx.contiguous(
            mx.transpose(keys[0], (1, 0, 2))
        )  # (n_new, n_kv_heads, head_dim)
        v = mx.contiguous(mx.transpose(values[0], (1, 0, 2)))
        # Cast to pool dtype if the model outputs a different dtype (e.g. bfloat16
        # weights produce bfloat16 K/V, but Metal kernel only supports float16).
        pool_dtype = self._pool.dtype
        if k.dtype != pool_dtype:
            k = k.astype(pool_dtype)
            v = v.astype(pool_dtype)

        slot_mapping = self._slot_mapping_for_new(n_new)
        self._pool.store(layer_idx, k, v, slot_mapping)
        self._new_tokens_stored += n_new

    # ── Attention path ────────────────────────────────────────────────────────

    def compute_paged_attn(
        self,
        q: mx.array,
        layer_idx: int,
        scale: float,
        max_seq_len: int | None = None,
    ) -> mx.array:
        """Compute decode attention over all cached tokens via paged_attention_v1.

        Args:
            q:           Shape ``(1, n_heads, 1, head_dim)`` — mlx-lm decode format.
            layer_idx:   Transformer layer index.
            scale:       Attention scale, typically ``1 / sqrt(head_dim)``.
            max_seq_len: Upper bound on sequence length (default: total_tokens).

        Returns:
            Shape ``(1, n_heads, 1, head_dim)`` — same as mlx-lm SDPA output.
        """
        _require_ext()
        pool = self._pool
        max_seq = max_seq_len or self._total_tokens

        # paged_attention_v1 expects q: (num_seqs, n_heads, head_dim)
        q_3d = q[0, :, 0, :][None, :, :]  # (1, n_heads, head_dim)

        # Cast query to pool dtype if needed (e.g. bfloat16 model with float16 cache).
        # The kernel is compiled for T==CACHE_T; we cast and restore dtype on output.
        out_dtype = q_3d.dtype
        pool_dtype = pool.dtype
        if q_3d.dtype != pool_dtype:
            q_3d = q_3d.astype(pool_dtype)

        out_3d = _ext.paged_attention_v1(
            q_3d,
            pool.key_cache[layer_idx],
            pool.val_cache[layer_idx],
            self._make_block_tables(),
            self._make_context_lens(),
            pool.n_kv_heads,
            scale,
            pool.block_size,
            max_seq,
        )
        if out_dtype != pool_dtype:
            out_3d = out_3d.astype(out_dtype)
        # (1, n_heads, head_dim) → (1, n_heads, 1, head_dim)
        return out_3d[:, :, None, :]

    # ── Drop-in shim for unpatched mlx-lm models ─────────────────────────────

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Store new KV and return gathered K, V in mlx-lm format.

        Compatibility shim for mlx-lm models that have not been patched to use
        ``store`` + ``compute_paged_attn`` directly. Stores tokens via
        ``reshape_and_cache`` (correct block layout) then gathers all K, V back
        as contiguous arrays.

        For full elimination of the gather copy, patch the model's attention
        layers to call ``store`` then ``compute_paged_attn`` instead.
        """
        self.store(keys, values, self._layer_idx)

        # Gather all K, V back from the block cache
        pool = self._pool
        bs = pool.block_size
        total = self._total_tokens
        all_blocks = self._all_blocks

        slots = []
        for i in range(total):
            block_id = all_blocks[i // bs]
            pos = i % bs
            slots.append(block_id * bs + pos)
        idx = mx.array(slots, dtype=mx.int32)

        kc = pool.key_cache[self._layer_idx]
        vc = pool.val_cache[self._layer_idx]

        nb, nkv, d_x, bsz, x = kc.shape
        # (nb, nkv, d//x, bs, x) → (nb*bs, nkv, head_dim)
        k_flat = mx.transpose(kc, (0, 3, 1, 2, 4)).reshape(nb * bsz, nkv, d_x * x)
        # (nb, nkv, head_dim, bs) → (nb*bs, nkv, head_dim)
        v_flat = mx.transpose(vc, (0, 3, 1, 2)).reshape(nb * bsz, nkv, d_x * x)

        k_gathered = k_flat[idx]  # (total, nkv, head_dim)
        v_gathered = v_flat[idx]

        # → (1, nkv, total, head_dim)
        k_out = mx.expand_dims(mx.transpose(k_gathered, (1, 0, 2)), 0)
        v_out = mx.expand_dims(mx.transpose(v_gathered, (1, 0, 2)), 0)
        return k_out, v_out

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        """Delegate to mlx-lm's mask helper."""
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(
            N, offset=self.offset, return_array=return_array, window_size=window_size
        )
