import time

import mlx.core as mx
import numpy as np

from sglang_mlx.metal_kernels import paged_attention_v1, reshape_and_cache


def bench_comparison(
    seq_len=2048,
    batch_size=1,
    num_heads=32,
    num_kv_heads=8,
    head_size=128,
    block_size=16,
):
    dtype = mx.float16
    num_blocks = (seq_len + block_size - 1) // block_size + 10
    x = 8
    scale = 1.0 / (head_size**0.5)

    # 1. Setup Paged Cache
    query = mx.random.uniform(shape=(batch_size, num_heads, head_size)).astype(dtype)
    key_cache = mx.zeros(
        (num_blocks, num_kv_heads, head_size // x, block_size, x), dtype=dtype
    )
    value_cache = mx.zeros(
        (num_blocks, num_kv_heads, head_size, block_size), dtype=dtype
    )

    # Fill cache
    k_src = mx.random.uniform(shape=(seq_len, num_kv_heads, head_size)).astype(dtype)
    v_src = mx.random.uniform(shape=(seq_len, num_kv_heads, head_size)).astype(dtype)
    slot_mapping = mx.array(np.arange(seq_len), dtype=mx.int32)
    key_cache, value_cache = reshape_and_cache(
        k_src, v_src, key_cache, value_cache, slot_mapping
    )

    block_tables = mx.array(
        np.array([np.arange(seq_len // block_size) for _ in range(batch_size)]),
        dtype=mx.int32,
    )
    seq_lens = mx.array([seq_len] * batch_size, dtype=mx.int32)
    mx.eval(key_cache, value_cache, block_tables)

    num_iters = 100

    # --- Scenario A: Paged Attention (Zero-Copy) ---
    # Warmup
    for _ in range(10):
        mx.eval(
            paged_attention_v1(
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                num_kv_heads,
                scale,
                block_size,
                seq_len,
            )
        )

    start = time.perf_counter()
    for _ in range(num_iters):
        out = paged_attention_v1(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_kv_heads,
            scale,
            block_size,
            seq_len,
        )
        mx.eval(out)
    paged_lat = (time.perf_counter() - start) / num_iters * 1000

    # --- Scenario B: Gather + Standard Attention (Simulating Radix Cache ---
    # without Paged Attention) ---
    def gather_and_sdpa():
        # 1. THE GATHER: This is what you MUST do if you don't have paged attention
        # We have to pull blocks from the cache and make them contiguous
        nb, nkv, d_x, bsz, x_val = key_cache.shape
        k_flat = mx.transpose(key_cache, (0, 3, 1, 2, 4)).reshape(
            nb * bsz, nkv, d_x * x_val
        )
        v_flat = mx.transpose(value_cache, (0, 3, 1, 2)).reshape(
            nb * bsz, nkv, head_size
        )

        # Logical indices for the tokens in the sequence
        indices = mx.array(np.arange(seq_len), dtype=mx.int32)

        # Real-world gather cost
        k_contig = k_flat[indices].transpose(1, 0, 2)[None, :, :, :]  # (1, nkv, seq, d)
        v_contig = v_flat[indices].transpose(1, 0, 2)[None, :, :, :]

        # 2. Standard SDPA
        q = query.reshape(batch_size, num_heads, 1, head_size)
        return mx.fast.scaled_dot_product_attention(q, k_contig, v_contig, scale=scale)

    # Warmup
    for _ in range(10):
        mx.eval(gather_and_sdpa())

    start = time.perf_counter()
    for _ in range(num_iters):
        mx.eval(gather_and_sdpa())
    gather_lat = (time.perf_counter() - start) / num_iters * 1000

    print(f"Seq Len: {seq_len}")
    print(f"  Paged Attention (Zero-Copy): {paged_lat:.4f} ms")
    print(f"  Gather + Native SDPA:        {gather_lat:.4f} ms")
    print(f"  True Speedup from Paged:     {gather_lat / paged_lat:.2f}x")


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    for s in [1024, 4096, 8192]:
        bench_comparison(seq_len=s)
