import time

import mlx.core as mx
import numpy as np

from sglang_mlx.metal_kernels import paged_attention_v1, reshape_and_cache


def bench_paged_attention(
    batch_size=1,
    num_heads=32,
    num_kv_heads=8,
    head_size=128,
    seq_len=1024,
    block_size=16,
    dtype=mx.float16,
    num_iters=100,
):
    # Setup
    num_blocks = (
        batch_size * seq_len + block_size - 1
    ) // block_size + 10  # extra buffer
    x = 8  # interleaving

    # Kernel expectations:
    # query: (num_seqs, num_heads, head_size)
    # key_cache: (num_blocks, num_kv_heads, head_size//x, block_size, x)
    # value_cache: (num_blocks, num_kv_heads, head_size, block_size)

    query = mx.random.uniform(shape=(batch_size, num_heads, head_size)).astype(dtype)
    key_cache = mx.zeros(
        (num_blocks, num_kv_heads, head_size // x, block_size, x), dtype=dtype
    )
    value_cache = mx.zeros(
        (num_blocks, num_kv_heads, head_size, block_size), dtype=dtype
    )

    # Fill cache with some data
    k_src = mx.random.uniform(
        shape=(batch_size * seq_len, num_kv_heads, head_size)
    ).astype(dtype)
    v_src = mx.random.uniform(
        shape=(batch_size * seq_len, num_kv_heads, head_size)
    ).astype(dtype)
    slot_mapping = mx.array(np.arange(batch_size * seq_len), dtype=mx.int32)

    key_cache, value_cache = reshape_and_cache(
        k_src, v_src, key_cache, value_cache, slot_mapping
    )
    mx.eval(key_cache, value_cache)

    block_tables = mx.array(
        np.array([np.arange(seq_len // block_size) for _ in range(batch_size)]),
        dtype=mx.int32,
    )
    seq_lens = mx.array([seq_len] * batch_size, dtype=mx.int32)
    scale = 1.0 / (head_size**0.5)

    # Warmup
    for _ in range(10):
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

    # Benchmark Paged Attention
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
    end = time.perf_counter()
    paged_lat = (end - start) / num_iters * 1000  # ms

    # Benchmark Baseline (Gather + mx.fast.scaled_dot_product_attention)
    # To be fair, we must gather the blocks into a contiguous KV cache
    # (batch_size, num_kv_heads, seq_len, head_size)

    def baseline():
        # This simulates what a non-paged system has to do: gather blocks
        # into a contiguous tensor. In reality, mlx-lm uses a list of arrays
        # or a large pre-allocated contiguous array. If it's already
        # contiguous, it's faster, but then it's not "paged".
        # Let's assume we have to gather from the paged cache to use standard SDPA.

        # Simple version: just use the k_src/v_src we already have
        # (best case for baseline)
        k = k_src.reshape(batch_size, seq_len, num_kv_heads, head_size).transpose(
            0, 2, 1, 3
        )
        v = v_src.reshape(batch_size, seq_len, num_kv_heads, head_size).transpose(
            0, 2, 1, 3
        )
        q = query.reshape(batch_size, num_heads, 1, head_size)

        # MLX SDPA
        res = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        return res

    # Warmup baseline
    for _ in range(10):
        mx.eval(baseline())

    start = time.perf_counter()
    for _ in range(num_iters):
        mx.eval(baseline())
    end = time.perf_counter()
    baseline_lat = (end - start) / num_iters * 1000  # ms

    print(f"Batch={batch_size}, Seq={seq_len}, Head={head_size}, Block={block_size}")
    print(f"  Paged Attention: {paged_lat:.4f} ms")
    print(f"  MLX SDPA Baseline: {baseline_lat:.4f} ms (approx)")
    print(f"  Speedup: {baseline_lat / paged_lat:.2f}x")


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    for s in [512, 1024, 2048, 4096]:
        bench_paged_attention(seq_len=s)
