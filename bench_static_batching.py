"""
bench_static_batching.py — Static batching: paged vs non-paged

Compares two systems running B sequences simultaneously in each decode step:

  Paged:      ModelRunner.forward_decode_batch
              → BatchedPagedKVCache + paged_attention_v1
                (one kernel call for all B seqs)

  Non-paged:  Batched SimpleKVCache
              → mx.concatenate(K/V) per layer + mx.fast.scaled_dot_product_attention

Uses the synthetic BenchModel (4L / 8H / 4KVH / head_dim=128) to isolate
attention and memory-bandwidth effects from weight-dequantization overhead.

Sections
--------
A. Throughput (tok/s) vs batch size at fixed context length (ctx=1024)
B. Throughput (tok/s) vs context length at fixed batch size (B=4)
"""

from __future__ import annotations

import math
import time
import types

import mlx.core as mx

from sglang_mlx.model_runner import ModelConfig, ModelRunner

try:
    from sglang_mlx import _ext  # noqa: F401
except ImportError:
    print("ERROR: Metal extension not built. Run: pip install -e .")
    raise


# ── Shared model architecture ──────────────────────────────────────────────────

N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 128
VOCAB_SIZE = 2000
BLOCK_SIZE = 16


def _sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    """GQA-aware scaled dot-product attention."""
    if q.shape[1] != k.shape[1]:
        k = mx.repeat(k, q.shape[1] // k.shape[1], axis=1)
        v = mx.repeat(v, q.shape[1] // v.shape[1], axis=1)
    w = mx.softmax(mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale, axis=-1)
    return mx.matmul(w, v)


def _make_weights():
    D = N_HEADS * HEAD_DIM
    mx.random.seed(0)
    embed = (mx.random.normal((VOCAB_SIZE, D)) * 0.02).astype(mx.float16)
    Wout = (mx.random.normal((D, VOCAB_SIZE)) * 0.02).astype(mx.float16)
    mx.eval(embed, Wout)
    return embed, Wout


# ── Non-paged system ───────────────────────────────────────────────────────────


class SimpleKVCache:
    """Per-sequence contiguous KV cache (standard mlx-lm style)."""

    def __init__(self):
        self._k: mx.array | None = None
        self._v: mx.array | None = None

    def update_and_fetch(self, k: mx.array, v: mx.array):
        if self._k is None:
            self._k, self._v = k, v
        else:
            self._k = mx.concatenate([self._k, k], axis=2)
            self._v = mx.concatenate([self._v, v], axis=2)
        return self._k, self._v


class BatchedSimpleKVCache:
    """Groups B SimpleKVCaches; stacks their K/V for batched SDPA.

    update_and_fetch receives (B, n_kv_heads, 1, head_dim) new tokens,
    appends to each per-sequence cache, and returns stacked full context.
    """

    def __init__(self, b: int):
        self._caches = [SimpleKVCache() for _ in range(b)]

    def update_and_fetch(self, k: mx.array, v: mx.array):
        k_full_list, v_full_list = [], []
        for i, cache in enumerate(self._caches):
            kf, vf = cache.update_and_fetch(k[i : i + 1], v[i : i + 1])
            k_full_list.append(kf)
            v_full_list.append(vf)
        return mx.concatenate(k_full_list, axis=0), mx.concatenate(v_full_list, axis=0)


class NonPagedModel:
    def __init__(self, embed: mx.array, Wout: mx.array):
        self.embed = embed
        self.Wout = Wout
        self.scale = HEAD_DIM**-0.5

    def __call__(self, input_ids: mx.array, cache=None):
        B, S = input_ids.shape
        D, KVD = N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM
        x = self.embed[input_ids]
        for li in range(N_LAYERS):
            q = x.reshape(B, S, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            k = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            v = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            if cache is not None:
                k, v = cache[li].update_and_fetch(k, v)
            x = x + _sdpa(q, k, v, self.scale).transpose(0, 2, 1, 3).reshape(B, S, D)
        return x @ self.Wout


# ── Paged system ───────────────────────────────────────────────────────────────


class PagedModel:
    """Same architecture; uses PagedKVCache for decode (store + compute_paged_attn)."""

    def __init__(self, embed: mx.array, Wout: mx.array):
        self.embed = embed
        self.Wout = Wout
        self.scale = HEAD_DIM**-0.5
        self.layers = [None] * N_LAYERS
        self._args = types.SimpleNamespace(
            num_key_value_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            vocab_size=VOCAB_SIZE,
            hidden_size=N_HEADS * HEAD_DIM,
            num_attention_heads=N_HEADS,
        )

    @property
    def args(self):
        return self._args

    def __call__(self, input_ids: mx.array, cache=None):
        B, S = input_ids.shape
        D, KVD = N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM
        x = self.embed[input_ids]
        for li in range(N_LAYERS):
            q = x.reshape(B, S, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            k = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            v = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            if cache is None:
                attn = _sdpa(q, k, v, self.scale)
            elif S == 1:
                from sglang_mlx.srt.mem_cache.paged_pool import BatchedPagedKVCache

                c = cache[li]
                if isinstance(c, BatchedPagedKVCache):
                    c.store_each(k, v, li)
                    attn = c.compute_paged_attn_batch(q, li, self.scale)
                else:
                    c.store(k, v, layer_idx=li)
                    attn = c.compute_paged_attn(q, layer_idx=li, scale=self.scale)
            else:
                c = cache[li]
                k_full, v_full = c.update_and_fetch(k, v)
                attn = _sdpa(q, k_full, v_full, self.scale)
            x = x + attn.transpose(0, 2, 1, 3).reshape(B, S, D)
        return x @ self.Wout


def _make_paged_runner(model: PagedModel, num_blocks: int) -> ModelRunner:
    config = ModelConfig(
        num_layers=N_LAYERS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB_SIZE,
        hidden_size=N_HEADS * HEAD_DIM,
        num_attention_heads=N_HEADS,
    )
    return ModelRunner(
        model,
        tokenizer=None,
        config=config,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        dtype=mx.float16,
    )


# ── Timing helpers ─────────────────────────────────────────────────────────────


def _mean(times: list[float]) -> float:
    return sum(times) / len(times)


def _bench_decode(
    np_model: NonPagedModel,
    paged_runner: ModelRunner,
    batch_size: int,
    ctx_len: int,
    n_warmup: int = 10,
    n_iters: int = 30,
) -> tuple[float, float]:
    """Returns (np_tps, paged_tps) for given batch_size and context length."""
    prompt = list(range(ctx_len))

    # ── Non-paged ──────────────────────────────────────────────────────────────
    np_caches = [BatchedSimpleKVCache(batch_size) for _ in range(N_LAYERS)]
    np_input = mx.array([prompt] * batch_size)  # (B, ctx_len)
    np_logits = np_model(np_input, cache=np_caches)
    mx.eval(np_logits)
    tokens_np = [0] * batch_size

    for _ in range(n_warmup):
        r = np_model(mx.array([[t] for t in tokens_np]), cache=np_caches)
        mx.eval(r)

    np_times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        r = np_model(mx.array([[t] for t in tokens_np]), cache=np_caches)
        mx.eval(r)
        np_times.append((time.perf_counter() - t0) * 1000)
        tokens_np = [int(mx.argmax(r[i, 0]).item()) for i in range(batch_size)]

    # ── Paged ──────────────────────────────────────────────────────────────────
    states, tokens_pg = [], []
    for i in range(batch_size):
        s, res = paged_runner.forward_prefill(f"ctx{ctx_len}-b{i}", prompt)
        mx.eval(res.logits)
        states.append(s)
        tokens_pg.append(int(mx.argmax(res.logits).item()))

    for _ in range(n_warmup):
        results = paged_runner.forward_decode_batch(states, tokens_pg)
        tokens_pg = [int(mx.argmax(r.logits).item()) for r in results]

    paged_times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        results = paged_runner.forward_decode_batch(states, tokens_pg)
        mx.eval(results[0].logits)
        paged_times.append((time.perf_counter() - t0) * 1000)
        tokens_pg = [int(mx.argmax(r.logits).item()) for r in results]

    for s in states:
        paged_runner.finish_request(s)

    np_tps = batch_size * 1000.0 / _mean(np_times)
    paged_tps = batch_size * 1000.0 / _mean(paged_times)
    return np_tps, paged_tps


# ── Section A: throughput vs batch size ───────────────────────────────────────


def bench_batch_size():
    CTX_LEN = 1024
    BATCH_SIZES = [1, 2, 4, 8]
    max_tokens = CTX_LEN + 50
    nb = 4 * math.ceil(max_tokens / BLOCK_SIZE) * max(BATCH_SIZES)

    embed, Wout = _make_weights()
    np_model = NonPagedModel(embed, Wout)
    paged_model = PagedModel(embed, Wout)
    runner = _make_paged_runner(paged_model, num_blocks=nb)

    print("=" * 72)
    print("Section A: Throughput vs Batch Size  (ctx=1024, synthetic BenchModel)")
    print("=" * 72)
    print(
        f"{'Batch':>6} | {'NP tok/s':>10} | {'Paged tok/s':>12} | "
        f"{'Speedup':>8} | {'NP ms/step':>11} | {'Paged ms/step':>13}"
    )
    print("-" * 72)

    rows = []
    for B in BATCH_SIZES:
        np_tps, paged_tps = _bench_decode(np_model, runner, B, CTX_LEN)
        np_ms = B * 1000.0 / np_tps
        paged_ms = B * 1000.0 / paged_tps
        speedup = paged_tps / np_tps
        rows.append((B, np_tps, paged_tps, speedup, np_ms, paged_ms))
        print(
            f"{B:>6} | {np_tps:>10.0f} | {paged_tps:>12.0f} | {speedup:>7.2f}x | "
            f"{np_ms:>11.2f} | {paged_ms:>13.2f}"
        )

    print()
    return rows


# ── Section B: throughput vs context length ───────────────────────────────────


def bench_context_len():
    BATCH_SIZE = 4
    CTX_LENGTHS = [64, 256, 512, 1024, 2048]
    max_tokens = max(CTX_LENGTHS) + 50
    nb = 4 * math.ceil(max_tokens / BLOCK_SIZE) * BATCH_SIZE

    embed, Wout = _make_weights()
    np_model = NonPagedModel(embed, Wout)
    paged_model = PagedModel(embed, Wout)
    runner = _make_paged_runner(paged_model, num_blocks=nb)

    print("=" * 72)
    print("Section B: Throughput vs Context Length  (B=4, synthetic BenchModel)")
    print("=" * 72)
    print(
        f"{'Ctx':>6} | {'NP tok/s':>10} | {'Paged tok/s':>12} | "
        f"{'Speedup':>8} | {'NP ms/step':>11} | {'Paged ms/step':>13}"
    )
    print("-" * 72)

    rows = []
    for ctx in CTX_LENGTHS:
        np_tps, paged_tps = _bench_decode(np_model, runner, BATCH_SIZE, ctx)
        np_ms = BATCH_SIZE * 1000.0 / np_tps
        paged_ms = BATCH_SIZE * 1000.0 / paged_tps
        speedup = paged_tps / np_tps
        rows.append((ctx, np_tps, paged_tps, speedup, np_ms, paged_ms))
        print(
            f"{ctx:>6} | {np_tps:>10.0f} | {paged_tps:>12.0f} | {speedup:>7.2f}x | "
            f"{np_ms:>11.2f} | {paged_ms:>13.2f}"
        )

    print()
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    mx.set_default_device(mx.gpu)
    print(
        f"Model: {N_LAYERS}L / {N_HEADS}H / {N_KV_HEADS}KVH "
        f"(GQA {N_HEADS // N_KV_HEADS}:1) / head_dim={HEAD_DIM} / "
        f"block_size={BLOCK_SIZE} / float16\n"
    )

    bench_batch_size()
    bench_context_len()

    print("=" * 72)
    print("Notes")
    print("=" * 72)
    print(
        "  Non-paged: mx.concatenate(K/V) per step (via BatchedSimpleKVCache) + SDPA."
    )
    print("  Paged:     reshape_and_cache (scatter-write) + paged_attention_v1")
    print("             (single kernel call covers all B sequences).")
    print("  All B sequences decoded in ONE forward pass for both systems.")
    print("  Synthetic model: attention cost is visible; no dequantization overhead.")


if __name__ == "__main__":
    main()
