"""Generate paged-attention benchmark tables and SVG graphs.

The reviewer asked for graphs across input shapes and KV-cache sizes. This
script intentionally has no plotting dependency: it writes a CSV, a Markdown
summary, and small SVG bar charts using the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from sglang_mlx.metal_kernels import paged_attention_v1, reshape_and_cache

SHAPES = [
    {
        "seq_len": 512,
        "head_dim": 64,
        "num_heads": 8,
        "num_kv_heads": 4,
        "block_size": 16,
    },
    {
        "seq_len": 1024,
        "head_dim": 64,
        "num_heads": 8,
        "num_kv_heads": 4,
        "block_size": 16,
    },
    {
        "seq_len": 2048,
        "head_dim": 64,
        "num_heads": 8,
        "num_kv_heads": 4,
        "block_size": 16,
    },
    {
        "seq_len": 4096,
        "head_dim": 64,
        "num_heads": 8,
        "num_kv_heads": 4,
        "block_size": 16,
    },
    {
        "seq_len": 8192,
        "head_dim": 64,
        "num_heads": 8,
        "num_kv_heads": 4,
        "block_size": 16,
    },
    {
        "seq_len": 2048,
        "head_dim": 128,
        "num_heads": 16,
        "num_kv_heads": 8,
        "block_size": 16,
    },
    {
        "seq_len": 4096,
        "head_dim": 128,
        "num_heads": 16,
        "num_kv_heads": 8,
        "block_size": 16,
    },
    {
        "seq_len": 8192,
        "head_dim": 128,
        "num_heads": 16,
        "num_kv_heads": 8,
        "block_size": 16,
    },
    {
        "seq_len": 4096,
        "head_dim": 128,
        "num_heads": 16,
        "num_kv_heads": 8,
        "block_size": 32,
    },
    {
        "seq_len": 8192,
        "head_dim": 128,
        "num_heads": 16,
        "num_kv_heads": 8,
        "block_size": 32,
    },
]


def _paged_cache_to_contiguous(key_cache, value_cache, seq_len: int, head_dim: int):
    nb, _, _, block_size, x = key_cache.shape
    key_flat = mx.transpose(key_cache, (0, 3, 1, 2, 4)).reshape(
        nb * block_size, key_cache.shape[1], head_dim
    )
    value_flat = mx.transpose(value_cache, (0, 3, 1, 2)).reshape(
        nb * block_size, value_cache.shape[1], head_dim
    )
    indices = mx.array(np.arange(seq_len, dtype=np.int32))
    return key_flat[indices].transpose(1, 0, 2), value_flat[indices].transpose(1, 0, 2)


def _build_inputs(shape: dict[str, int], dtype=mx.float16):
    seq_len = shape["seq_len"]
    head_dim = shape["head_dim"]
    block_size = shape["block_size"]
    num_heads = shape["num_heads"]
    num_kv_heads = shape["num_kv_heads"]
    x = 8
    num_blocks = math.ceil(seq_len / block_size)

    query = mx.random.uniform(shape=(1, num_heads, head_dim)).astype(dtype)
    keys = mx.random.uniform(shape=(seq_len, num_kv_heads, head_dim)).astype(dtype)
    values = mx.random.uniform(shape=(seq_len, num_kv_heads, head_dim)).astype(dtype)

    key_cache = mx.zeros(
        (num_blocks, num_kv_heads, head_dim // x, block_size, x), dtype=dtype
    )
    value_cache = mx.zeros(
        (num_blocks, num_kv_heads, head_dim, block_size), dtype=dtype
    )
    slots = mx.array(np.arange(seq_len, dtype=np.int32))
    key_cache, value_cache = reshape_and_cache(
        keys, values, key_cache, value_cache, slots
    )

    block_tables = mx.array(np.arange(num_blocks, dtype=np.uint32).reshape(1, -1))
    context_lens = mx.array([seq_len], dtype=mx.uint32)
    mx.eval(query, key_cache, value_cache, block_tables, context_lens)
    return query, key_cache, value_cache, block_tables, context_lens


def _measure(fn, warmup: int, iters: int):
    for _ in range(warmup):
        mx.eval(fn())

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return {
        "mean_ms": statistics.fmean(times),
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
    }


def benchmark_shape(shape: dict[str, int], warmup: int, iters: int):
    query, key_cache, value_cache, block_tables, context_lens = _build_inputs(shape)
    scale = shape["head_dim"] ** -0.5

    def run_paged():
        return paged_attention_v1(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            shape["num_kv_heads"],
            scale,
            shape["block_size"],
            shape["seq_len"],
        )

    key_contig, value_contig = _paged_cache_to_contiguous(
        key_cache, value_cache, shape["seq_len"], shape["head_dim"]
    )
    num_queries_per_kv = shape["num_heads"] // shape["num_kv_heads"]
    key_contig = mx.repeat(key_contig, num_queries_per_kv, axis=0)[None, :, :, :]
    value_contig = mx.repeat(value_contig, num_queries_per_kv, axis=0)[None, :, :, :]
    query_sdpa = query.reshape(1, shape["num_heads"], 1, shape["head_dim"])

    def run_native():
        return mx.fast.scaled_dot_product_attention(
            query_sdpa, key_contig, value_contig, scale=scale
        )

    paged = _measure(run_paged, warmup, iters)
    native = _measure(run_native, warmup, iters)
    row = dict(shape)
    row.update(
        {
            "paged_mean_ms": paged["mean_ms"],
            "paged_p50_ms": paged["p50_ms"],
            "paged_p95_ms": paged["p95_ms"],
            "native_mean_ms": native["mean_ms"],
            "native_p50_ms": native["p50_ms"],
            "native_p95_ms": native["p95_ms"],
            "speedup": native["mean_ms"] / paged["mean_ms"],
        }
    )
    return row


def _write_csv(rows: list[dict[str, float]], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, float]], path: Path):
    columns = [
        "seq_len",
        "head_dim",
        "num_heads",
        "num_kv_heads",
        "block_size",
        "paged_mean_ms",
        "native_mean_ms",
        "speedup",
    ]
    with path.open("w") as f:
        f.write("# Paged Attention Benchmark Results\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = []
            for col in columns:
                value = row[col]
                values.append(
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )
            f.write("| " + " | ".join(values) + " |\n")


def _rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width:.2f}" height="{height}" fill="{fill}"/>'
    )


def _text(x: float, y: float, body: str) -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}">{body}</text>'


def _write_svg(rows: list[dict[str, float]], path: Path):
    width = 1100
    row_h = 34
    left = 250
    top = 40
    max_ms = max(max(row["paged_mean_ms"], row["native_mean_ms"]) for row in rows)
    height = top + len(rows) * row_h + 40
    scale = (width - left - 80) / max_ms

    parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            "<style>text{font-family:-apple-system,BlinkMacSystemFont,"
            "Segoe UI,sans-serif;font-size:12px}</style>"
        ),
        (
            '<text x="20" y="24" font-size="18" font-weight="700">'
            "Paged attention latency by KV-cache size</text>"
        ),
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = (
            f"seq={row['seq_len']}, hd={row['head_dim']}, "
            f"h={row['num_heads']}/{row['num_kv_heads']}, bs={row['block_size']}"
        )
        paged_w = row["paged_mean_ms"] * scale
        native_w = row["native_mean_ms"] * scale
        parts.append(f'<text x="20" y="{y + 16}">{label}</text>')
        parts.append(_rect(left, y, paged_w, 12, "#2563eb"))
        parts.append(_rect(left, y + 15, native_w, 12, "#f97316"))
        parts.append(
            _text(left + paged_w + 6, y + 10, f"{row['paged_mean_ms']:.3f} ms paged")
        )
        parts.append(
            _text(
                left + native_w + 6,
                y + 25,
                f"{row['native_mean_ms']:.3f} ms native",
            )
        )
    parts.append(_rect(left, height - 26, 14, 10, "#2563eb"))
    parts.append(_text(left + 20, height - 17, "Paged"))
    parts.append(_rect(left + 90, height - 26, 14, 10, "#f97316"))
    parts.append(_text(left + 110, height - 17, "Native SDPA on contiguous KV"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("tests/benchmark_results"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [benchmark_shape(shape, args.warmup, args.iters) for shape in SHAPES]
    _write_csv(rows, args.out_dir / "paged_attention_benchmarks.csv")
    _write_markdown(rows, args.out_dir / "paged_attention_benchmarks.md")
    _write_svg(rows, args.out_dir / "paged_attention_benchmarks.svg")


if __name__ == "__main__":
    main()
