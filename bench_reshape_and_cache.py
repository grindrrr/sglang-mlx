"""
bench_reshape_and_cache.py — Benchmark the KV cache write kernel.

Measures reshape_and_cache (the scatter-write that fills paged blocks with new
token K/V) at varying token counts.  Reports latency, throughput (tokens/s) and
effective write bandwidth (GB/s) so the results can be compared against Apple
Silicon's theoretical memory bandwidth.

Config: 8 KV-heads, head_dim=128, block_size=16, float16  (Llama-style)
"""

from __future__ import annotations

import math
import time

import mlx.core as mx
import numpy as np

try:
    from sglang_mlx.metal_kernels import reshape_and_cache
except ImportError:
    print("ERROR: Metal extension not built. Run: pip install -e .")
    raise


# ── Config ─────────────────────────────────────────────────────────────────────

N_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
_KEY_X = 8  # interleaving factor: 16-bytes/sizeof(fp16)=8
N_WARMUP = 20
N_ITERS = 100
DTYPE = mx.float16

TOKEN_COUNTS = [1, 8, 16, 64, 128, 256, 512, 1024]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_empty_cache(num_blocks: int):
    """Return zeroed key/value cache arrays in paged layout."""
    kc = mx.zeros(
        (num_blocks, N_KV_HEADS, HEAD_DIM // _KEY_X, BLOCK_SIZE, _KEY_X),
        dtype=DTYPE,
    )
    vc = mx.zeros(
        (num_blocks, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE),
        dtype=DTYPE,
    )
    return kc, vc


def _bytes_written(n_tokens: int) -> int:
    """Total bytes written to key+value caches for n_tokens."""
    fp16_bytes = 2
    return n_tokens * N_KV_HEADS * HEAD_DIM * fp16_bytes * 2  # K + V


def _measure(n_tokens: int):
    """Return (mean_ms, p50_ms, p95_ms) for reshape_and_cache of n_tokens."""
    num_blocks = math.ceil(n_tokens / BLOCK_SIZE) + 1
    kc, vc = _make_empty_cache(num_blocks)
    mx.eval(kc, vc)

    # Source tensors: (n_tokens, n_kv_heads, head_dim)
    keys = mx.random.uniform(shape=(n_tokens, N_KV_HEADS, HEAD_DIM)).astype(DTYPE)
    values = mx.random.uniform(shape=(n_tokens, N_KV_HEADS, HEAD_DIM)).astype(DTYPE)
    slots = mx.array(np.arange(n_tokens, dtype=np.int64))
    mx.eval(keys, values, slots)

    # Warmup (triggers Metal JIT)
    for _ in range(N_WARMUP):
        new_kc, new_vc = reshape_and_cache(
            keys, values, kc, vc, slots, "auto", 1.0, 1.0
        )
        mx.eval(new_kc, new_vc)

    # Measure
    times_ms: list[float] = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        new_kc, new_vc = reshape_and_cache(
            keys, values, kc, vc, slots, "auto", 1.0, 1.0
        )
        mx.eval(new_kc, new_vc)
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms.sort()
    return (
        sum(times_ms) / N_ITERS,
        times_ms[N_ITERS // 2],
        times_ms[int(N_ITERS * 0.95)],
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    mx.set_default_device(mx.gpu)

    print("reshape_and_cache throughput")
    print(
        f"Config: n_kv_heads={N_KV_HEADS}, head_dim={HEAD_DIM}, "
        f"block_size={BLOCK_SIZE}, dtype=float16"
    )
    print()
    print(
        f"{'Tokens':>8} | {'mean (ms)':>10} | {'p50 (ms)':>9} | {'p95 (ms)':>9} "
        f"| {'Tokens/s':>12} | {'BW (GB/s)':>10}"
    )
    print("-" * 74)

    for n in TOKEN_COUNTS:
        mean_ms, p50_ms, p95_ms = _measure(n)
        mean_s = mean_ms / 1000
        tok_per_s = n / mean_s
        bw_gbs = _bytes_written(n) / mean_s / 1e9

        print(
            f"{n:>8} | {mean_ms:>10.4f} | {p50_ms:>9.4f} | {p95_ms:>9.4f} "
            f"| {tok_per_s:>12,.0f} | {bw_gbs:>10.2f}"
        )

    print()
    print("Note: 'BW' = bytes written to key+value caches / elapsed time.")
    print("      Apple M-series GPU bandwidth: ~100–400 GB/s (unified memory).")


if __name__ == "__main__":
    main()
