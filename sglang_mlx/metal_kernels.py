"""
metal_kernels.py — paged attention Metal kernel integration for sglang-mlx.

Exposes the compiled paged-attention Metal extension (_ext) as a clean Python
API. All operations delegate directly to the C++/Metal kernels built in
paged_attention/ and compiled into sglang_mlx/_ext.

Available operations:
    paged_attention_v1   — single-pass paged attention (decode)
    paged_attention_v2   — two-pass paged attention with partitioning
    reshape_and_cache    — scatter-write new KV tokens into the block cache
    reshape_and_cache_flash — flash-layout variant
    copy_blocks          — copy KV blocks (e.g. for beam search)
    swap_blocks          — swap KV blocks between slots
    convert_fp8          — FP8 ↔ float conversion
"""

from __future__ import annotations

import mlx.core as mx

# ── Load the compiled extension ───────────────────────────────────────────────

try:
    from . import _ext
except ImportError as e:
    raise ImportError(
        "sglang_mlx._ext not found. Build it first:\n"
        "  pip install -e .[dev]\n"
        "or manually:\n"
        "  cmake -B build . && cmake --build build"
    ) from e


# ── Paged Attention ───────────────────────────────────────────────────────────


def paged_attention_v1(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    seq_lens: mx.array,
    num_kv_heads: int,
    scale: float,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: mx.array | None = None,
    kv_cache_dtype: str = "float",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> mx.array:
    """Single-pass paged attention.

    Args:
        query:       Shape ``(num_seqs, num_heads, head_size)``.
        key_cache:   Shape ``(num_blocks, num_kv_heads, head_size//x, block_size, x)``.
        value_cache: Shape ``(num_blocks, num_kv_heads, head_size, block_size)``.
        block_tables: Shape ``(num_seqs, max_blocks_per_seq)`` int32.
        seq_lens:    Shape ``(num_seqs,)`` int32.

    Returns:
        Shape ``(num_seqs, num_heads, head_size)``.
    """
    return _ext.paged_attention_v1(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        num_kv_heads,
        scale,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def paged_attention_v2(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    seq_lens: mx.array,
    num_kv_heads: int,
    scale: float,
    block_size: int,
    max_seq_len: int,
    max_num_partitions: int,
    alibi_slopes: mx.array | None = None,
    kv_cache_dtype: str = "float",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> list[mx.array]:
    """Two-pass paged attention with partitioning for long sequences.

    Returns:
        ``[out, exp_sums, max_logits, tmp_out]``
    """
    return _ext.paged_attention_v2(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        num_kv_heads,
        scale,
        block_size,
        max_seq_len,
        max_num_partitions,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


# ── Cache Operations ──────────────────────────────────────────────────────────


def reshape_and_cache(
    key: mx.array,
    value: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    slot_mapping: mx.array,
    kv_cache_dtype: str = "float",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Scatter-write new KV tokens into the paged block cache.

    Args:
        key:          Shape ``(num_tokens, num_kv_heads, head_size)``.
        value:        Same shape as key.
        key_cache:    Shape ``(num_blocks, num_kv_heads, head_size//x, block_size, x)``.
        value_cache:  Shape ``(num_blocks, num_kv_heads, head_size, block_size)``.
        slot_mapping: Shape ``(num_tokens,)`` int32.
                      Each entry = ``block_id * block_size + position_in_block``.

    Returns:
        ``(new_key_cache, new_value_cache)`` with tokens written in.
    """
    result = _ext.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    return result[0], result[1]


def reshape_and_cache_flash(
    key: mx.array,
    value: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    slot_mapping: mx.array,
    kv_cache_dtype: str = "float",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Flash-layout variant of reshape_and_cache."""
    result = _ext.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    return result[0], result[1]


def copy_blocks(
    key_caches: list[mx.array],
    value_caches: list[mx.array],
    block_mapping: mx.array,
) -> tuple[list[mx.array], list[mx.array]]:
    """Copy KV blocks across layers (e.g. for beam search).

    Returns:
        ``(new_key_caches, new_value_caches)``
    """
    return _ext.copy_blocks(key_caches, value_caches, block_mapping)


def swap_blocks(
    src: mx.array,
    dst: mx.array,
    block_mapping: mx.array,
) -> mx.array:
    """Copy src blocks into dst. Returns new_dst."""
    return _ext.swap_blocks(src, dst, block_mapping)


# ── FP8 Conversion ────────────────────────────────────────────────────────────


def convert_fp8(
    src_cache: mx.array,
    scale: float = 1.0,
    kv_cache_dtype: str = "fp8",
    dst_dtype_str: str = "float16",
) -> mx.array:
    """Convert between FP8 (uint8) and float/half/bfloat16."""
    return _ext.convert_fp8(src_cache, scale, kv_cache_dtype, dst_dtype_str)
