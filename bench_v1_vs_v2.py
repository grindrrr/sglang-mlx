"""
bench_v1_vs_v2.py — Find the paged_attention_v1 / v2 crossover point.

v1: single-pass kernel (lower overhead, better for short sequences)
v2: two-pass partitioned kernel (amortises work across warps, better for long sequences)

Sweeps seq_len from 128 to 8192 and reports which kernel wins at each length.
Partition size for v2 is 512 tokens (matching vLLM's default).

Config: 8 attn-heads, 4 KV-heads (GQA 2:1), head_dim=128, block_size=16, float16
"""

from __future__ import annotations

import math
import time

import mlx.core as mx
import numpy as np

try:
    from sglang_mlx.metal_kernels import (
        paged_attention_v1,
        paged_attention_v2,
        reshape_and_cache,
    )
except ImportError:
    print("ERROR: Metal extension not built. Run: pip install -e .")
    raise


# ── Config ─────────────────────────────────────────────────────────────────────

N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 16
_KEY_X = 8
DTYPE = mx.float16

V2_PARTITION_SIZE = 512  # tokens per partition in the two-pass kernel

N_WARMUP = 20
N_ITERS = 100

SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]


# ── Setup helpers ──────────────────────────────────────────────────────────────


def _build_inputs(seq_len: int):
    """Build all arrays needed for one paged attention call."""
    num_blocks = math.ceil(seq_len / BLOCK_SIZE)
    scale = HEAD_DIM**-0.5

    query = mx.random.uniform(shape=(1, N_HEADS, HEAD_DIM)).astype(DTYPE)
    kc = mx.zeros(
        (num_blocks, N_KV_HEADS, HEAD_DIM // _KEY_X, BLOCK_SIZE, _KEY_X), dtype=DTYPE
    )
    vc = mx.zeros((num_blocks, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE), dtype=DTYPE)

    # Fill cache with random data
    k_src = mx.random.uniform(shape=(seq_len, N_KV_HEADS, HEAD_DIM)).astype(DTYPE)
    v_src = mx.random.uniform(shape=(seq_len, N_KV_HEADS, HEAD_DIM)).astype(DTYPE)
    slots = mx.array(np.arange(seq_len, dtype=np.int64))
    kc, vc = reshape_and_cache(k_src, v_src, kc, vc, slots, "auto", 1.0, 1.0)

    block_tables = mx.array(np.arange(num_blocks, dtype=np.int32).reshape(1, -1))
    seq_lens_arr = mx.array([seq_len], dtype=mx.int32)
    mx.eval(query, kc, vc, block_tables, seq_lens_arr)

    max_num_partitions = max(1, math.ceil(seq_len / V2_PARTITION_SIZE))
    return query, kc, vc, block_tables, seq_lens_arr, scale, max_num_partitions


def _bench(fn, n_warmup: int = N_WARMUP, n_iters: int = N_ITERS):
    """Measure fn() → (mean_ms, p50_ms, p95_ms)."""
    for _ in range(n_warmup):
        fn()
    times_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms.sort()
    return (
        sum(times_ms) / n_iters,
        times_ms[n_iters // 2],
        times_ms[int(n_iters * 0.95)],
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    mx.set_default_device(mx.gpu)

    print("paged_attention_v1 vs v2 — latency crossover")
    print(
        f"Config: {N_HEADS} heads, {N_KV_HEADS} KV-heads "
        f"(GQA {N_HEADS // N_KV_HEADS}:1), head_dim={HEAD_DIM}, "
        f"block_size={BLOCK_SIZE}, dtype=float16"
    )
    print(
        f"v2 partition size: {V2_PARTITION_SIZE} tokens  "
        f"(max_partitions = ceil(seq_len / {V2_PARTITION_SIZE}))"
    )
    print()
    print(
        f"{'Seq Len':>9} | {'v1 mean':>9} | {'v1 p95':>8} "
        f"| {'v2 mean':>9} | {'v2 p95':>8} | {'Winner':>8} | {'v1/v2':>7}"
    )
    print(
        f"{'':>9} | {'(ms)':>9} | {'(ms)':>8} "
        f"| {'(ms)':>9} | {'(ms)':>8} | {'':>8} | {'ratio':>7}"
    )
    print("-" * 77)

    for seq_len in SEQ_LENGTHS:
        query, kc, vc, block_tables, seq_lens_arr, scale, max_parts = _build_inputs(
            seq_len
        )

        def run_v1():
            out = paged_attention_v1(
                query,
                kc,
                vc,
                block_tables,
                seq_lens_arr,
                N_KV_HEADS,
                scale,
                BLOCK_SIZE,
                seq_len,
            )
            mx.eval(out)

        def run_v2():
            outputs = paged_attention_v2(
                query,
                kc,
                vc,
                block_tables,
                seq_lens_arr,
                N_KV_HEADS,
                scale,
                BLOCK_SIZE,
                seq_len,
                max_parts,
            )
            mx.eval(outputs[0])

        v1_mean, _, v1_p95 = _bench(run_v1)
        v2_mean, _, v2_p95 = _bench(run_v2)

        winner = "v1" if v1_mean < v2_mean else "v2"
        ratio = v1_mean / v2_mean

        print(
            f"{seq_len:>9} | {v1_mean:>9.4f} | {v1_p95:>8.4f} "
            f"| {v2_mean:>9.4f} | {v2_p95:>8.4f} | {winner:>8} | {ratio:>7.2f}x"
        )

    print()
    print("Interpretation:")
    print("  ratio < 1.0 → v2 is faster  (use v2 for long sequences)")
    print("  ratio > 1.0 → v1 is faster  (use v1 for short sequences)")
    print(
        f"  v2 partitions = ceil(seq_len / {V2_PARTITION_SIZE}); "
        f"v2 adds overhead only when partitions > 1 (seq > {V2_PARTITION_SIZE})."
    )


if __name__ == "__main__":
    main()
