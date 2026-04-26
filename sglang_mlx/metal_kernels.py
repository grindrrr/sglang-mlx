"""
metal_kernels.py — paged attention Metal kernel integration for sglang-mlx.

Exposes the compiled paged-attention Metal extension (_ext) as a clean Python
API. All operations delegate directly to the C++/Metal kernels built in
paged_attention/ and compiled into sglang_mlx/_ext.

Available operations:
    paged_attention_v1   — single-pass paged attention (decode)
    reshape_and_cache    — scatter KV tokens into paged blocks
    copy_blocks          — intra-cache block copying
    swap_blocks          — inter-cache block copying (e.g. host <-> device)
"""

import mlx.core as mx

from . import _ext


def paged_attention_v1(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    num_kv_heads: int,
    scale: float,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: mx.array | None = None,
) -> mx.array:
    """
    Single-pass paged attention kernel.

    Args:
        query: [num_seqs, num_heads, head_size]
        key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        block_tables: [num_seqs, max_num_blocks_per_seq]
        context_lens: [num_seqs]
        num_kv_heads: Number of KV heads
        scale: Attention scaling factor
        block_size: Number of tokens per block
        max_seq_len: Maximum sequence length
        alibi_slopes: Optional ALiBi slopes [num_heads]

    Returns:
        Output tensor [num_seqs, num_heads, head_size]
    """
    return _ext.paged_attention_v1(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        num_kv_heads,
        scale,
        block_size,
        max_seq_len,
        alibi_slopes,
    )


def reshape_and_cache(
    key: mx.array,
    value: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    slot_mapping: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Scatter-write new KV tokens into the paged cache.
    """
    return _ext.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)


def copy_blocks(
    key_caches: list[mx.array],
    value_caches: list[mx.array],
    block_mapping: mx.array,
) -> tuple[list[mx.array], list[mx.array]]:
    """
    Copy blocks within the cache.
    """
    return _ext.copy_blocks(key_caches, value_caches, block_mapping)


def swap_blocks(
    src: mx.array,
    dst: mx.array,
    block_mapping: mx.array,
) -> mx.array:
    """
    Blit blocks between two buffers.
    """
    return _ext.swap_blocks(src, dst, block_mapping)
