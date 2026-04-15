"""
bench_paged_vs_nonpaged.py — Full-stack comparison: paged vs non-paged inference.

This is the most important benchmark in the suite.  It answers the question a
user actually cares about:

  "In a real serving scenario where two users share a common prefix (e.g. a
   system prompt), how much faster is the second user's request with paged
   attention + radix cache vs a standard contiguous-KV approach?"

Two systems, same model architecture, same sequence lengths:

  NonPagedRunner  — standard mlx-lm style.  K/V grows as a contiguous tensor
                    per session.  No cross-request prefix reuse.  Every new
                    request recomputes the full prompt from scratch.

  PagedRunner     — ModelRunner with BlockAllocator + RadixCache.  The second
                    request hits the radix cache on the shared prefix and only
                    processes the new tokens.

Sections
--------
A. TTFT (time-to-first-token): shared prefix scenario
   - Non-paged req 2: pays full N-token prefill cost every time
   - Paged req 2:     pays only for the N_new tokens after cache hit

B. TPOT (time-per-output-token): decode latency at equal context lengths
   - Both systems start from the same context length
   - Non-paged: concatenate K/V, then contiguous SDPA
   - Paged:     store to block, then paged_attention_v1

C. Summary table
"""

from __future__ import annotations

import math
import time
import types

import mlx.core as mx

from sglang_mlx.model_runner import ModelConfig, ModelRunner
from sglang_mlx.srt.mem_cache.base_prefix_cache import EvictParams
from sglang_mlx.srt.mem_cache.paged_pool import PagedKVCache

try:
    from sglang_mlx import _ext  # noqa: F401
except ImportError:
    print("ERROR: Metal extension not built. Run: pip install -e .")
    raise


# ── Shared model architecture ──────────────────────────────────────────────────
#
# Both runners use the same weights (same seed, same architecture).
# Dimensions: 4 layers, 8 attn-heads, 4 KV-heads (GQA 2:1), head_dim=128.

N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 128
VOCAB_SIZE = 2000
BLOCK_SIZE = 16


def _sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    """GQA-aware scaled dot-product attention (no causal mask).
    q: (B, n_heads, S_q, D)   k/v: (B, n_kv_heads, S_k, D)
    """
    if q.shape[1] != k.shape[1]:
        k = mx.repeat(k, q.shape[1] // k.shape[1], axis=1)
        v = mx.repeat(v, q.shape[1] // v.shape[1], axis=1)
    w = mx.softmax(mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale, axis=-1)
    return mx.matmul(w, v)


def _make_weights():
    """Shared embedding + output projection weights (seeded for reproducibility)."""
    D = N_HEADS * HEAD_DIM
    mx.random.seed(0)
    embed = (mx.random.normal((VOCAB_SIZE, D)) * 0.02).astype(mx.float16)
    Wout = (mx.random.normal((D, VOCAB_SIZE)) * 0.02).astype(mx.float16)
    mx.eval(embed, Wout)
    return embed, Wout


# ── Non-paged system ───────────────────────────────────────────────────────────


class SimpleKVCache:
    """Contiguous growing KV cache — the standard mlx-lm approach.

    Each decode step concatenates the new token's K/V onto a contiguous tensor.
    No block management, no cross-request reuse.
    """

    def __init__(self):
        self._k: mx.array | None = None  # (1, n_kv_heads, seq_len, head_dim)
        self._v: mx.array | None = None

    def update_and_fetch(self, k: mx.array, v: mx.array):
        """Append new K/V and return full context."""
        if self._k is None:
            self._k, self._v = k, v
        else:
            self._k = mx.concatenate([self._k, k], axis=2)
            self._v = mx.concatenate([self._v, v], axis=2)
        return self._k, self._v


class NonPagedModel:
    """Same architecture as PagedModel but uses SimpleKVCache (contiguous)."""

    def __init__(self, embed: mx.array, Wout: mx.array):
        self.embed = embed
        self.Wout = Wout
        self.scale = HEAD_DIM**-0.5

    def __call__(self, input_ids: mx.array, cache: list[SimpleKVCache] | None = None):
        B, S = input_ids.shape
        D = N_HEADS * HEAD_DIM
        KVD = N_KV_HEADS * HEAD_DIM
        x = self.embed[input_ids]

        for li in range(N_LAYERS):
            q = x.reshape(B, S, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            k = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            v = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

            if cache is not None:
                k, v = cache[li].update_and_fetch(k, v)

            attn = _sdpa(q, k, v, self.scale)
            x = x + attn.transpose(0, 2, 1, 3).reshape(B, S, D)

        return x @ self.Wout


class NonPagedRunner:
    """Standard contiguous-KV runner.  No radix cache, no prefix reuse."""

    def __init__(self, model: NonPagedModel):
        self._model = model

    def forward_prefill(self, prompt_tokens: list[int]):
        """Full forward pass on the entire prompt — always from scratch."""
        cache = [SimpleKVCache() for _ in range(N_LAYERS)]
        input_ids = mx.array([prompt_tokens])
        logits = self._model(input_ids, cache=cache)
        mx.eval(logits)
        return cache, logits[0, -1, :]

    def forward_decode(self, cache: list[SimpleKVCache], token_id: int):
        """One decode step: append K/V and attend over full context."""
        input_ids = mx.array([[token_id]])
        logits = self._model(input_ids, cache=cache)
        mx.eval(logits)
        return logits[0, 0, :]


# ── Paged system ───────────────────────────────────────────────────────────────


class PagedModel:
    """Same architecture, uses PagedKVCache (store + compute_paged_attn for decode,
    update_and_fetch for prefill)."""

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
        D = N_HEADS * HEAD_DIM
        KVD = N_KV_HEADS * HEAD_DIM
        x = self.embed[input_ids]

        for li in range(N_LAYERS):
            q = x.reshape(B, S, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            k = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            v = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

            if cache is None:
                attn = _sdpa(q, k, v, self.scale)
            elif S == 1:
                c: PagedKVCache = cache[li]
                c.store(k, v, layer_idx=li)
                attn = c.compute_paged_attn(q, layer_idx=li, scale=self.scale)
            else:
                c: PagedKVCache = cache[li]
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


def _percentile(times: list[float], p: float) -> float:
    s = sorted(times)
    return s[min(int(len(s) * p), len(s) - 1)]


def _stats(times: list[float]):
    mean = sum(times) / len(times)
    return mean, _percentile(times, 0.50), _percentile(times, 0.95)


# ── Section A: TTFT sweep — vary prompt length, fix unique suffix at 16 tokens ─


def bench_ttft():
    """Sweep TTFT across prompt lengths with three systems:

      1. Non-paged          — contiguous KV, no prefix cache, full recompute
      2. Paged, no cache    — paged kernel only, radix cache evicted each iter
      3. Paged, with cache  — paged kernel + radix cache prefix reuse

    Comparing (1) vs (2) isolates the kernel overhead.
    Comparing (1) vs (3) shows the full-system speedup.
    Comparing (2) vs (3) isolates the prefix cache benefit.
    """
    PROMPT_LENGTHS = [64, 128, 256, 512, 1024, 2048]
    SUFFIX_LEN = 16  # always 1 block of unique tokens
    N_WARMUP = 5
    N_ITERS = 15

    print("=" * 100)
    print("Section A: TTFT Sweep — Three Systems vs Prompt Length")
    print("=" * 100)
    print(
        f"Unique suffix fixed at {SUFFIX_LEN} tokens.  "
        f"Shared prefix = prompt_len - {SUFFIX_LEN}."
    )
    print(
        f"Model: {N_LAYERS}L / {N_HEADS}H / {N_KV_HEADS}KVH "
        f"(GQA {N_HEADS // N_KV_HEADS}:1) / head_dim={HEAD_DIM} / "
        f"block_size={BLOCK_SIZE}"
    )
    print(f"Warmup: {N_WARMUP} iters discarded.  Measured: {N_ITERS} iters.")
    print()
    print(
        "  System 1 (NP):          non-paged, no prefix cache — "
        "full recompute every request"
    )
    print(
        "  System 2 (Paged-cold):  paged kernel, NO prefix cache — "
        "full paged prefill every request"
    )
    print(
        "  System 3 (Paged-warm):  paged kernel + radix cache — "
        "only processes new tokens"
    )
    print()

    hdr = (
        f"{'Prompt':>8} | {'NP (ms)':>9} | {'Paged-cold':>11} | {'vs NP':>7} | "
        f"{'Paged-warm':>11} | {'vs NP':>7} | {'Cache only':>11}"
    )
    sub = (
        f"{'(tokens)':>8} | {'mean':>9} | {'mean (ms)':>11} | {'kernel':>7} | "
        f"{'mean (ms)':>11} | {'full':>7} | {'benefit':>11}"
    )
    print(hdr)
    print(sub)
    print("-" * 100)

    embed, Wout = _make_weights()
    np_model = NonPagedModel(embed, Wout)
    np_runner = NonPagedRunner(np_model)

    rows: list[tuple] = []

    for prompt_len in PROMPT_LENGTHS:
        shared_len = prompt_len - SUFFIX_LEN
        full_prompt = list(range(prompt_len))
        req2_prompt = list(range(shared_len)) + list(
            range(VOCAB_SIZE - SUFFIX_LEN, VOCAB_SIZE)
        )
        nb = 4 * math.ceil(prompt_len / BLOCK_SIZE)

        # ── System 1: Non-paged, full recompute ───────────────────────────────
        for _ in range(N_WARMUP):
            np_runner.forward_prefill(req2_prompt)

        np_times: list[float] = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            np_runner.forward_prefill(req2_prompt)
            np_times.append((time.perf_counter() - t0) * 1000)

        # ── System 2: Paged kernel, no prefix cache ────────────────────────────
        paged_model_cold = PagedModel(embed, Wout)
        paged_runner_cold = _make_paged_runner(paged_model_cold, num_blocks=nb)

        for i in range(N_WARMUP):
            paged_runner_cold._radix_cache.evict(EvictParams(num_tokens=999_999))
            s, r = paged_runner_cold.forward_prefill(f"cw{i}", req2_prompt)
            mx.eval(r.logits)
            paged_runner_cold.finish_request(s)

        cold_times: list[float] = []
        for i in range(N_ITERS):
            paged_runner_cold._radix_cache.evict(EvictParams(num_tokens=999_999))
            t0 = time.perf_counter()
            s, r = paged_runner_cold.forward_prefill(f"cp{i}", req2_prompt)
            mx.eval(r.logits)
            cold_times.append((time.perf_counter() - t0) * 1000)
            paged_runner_cold.finish_request(s)

        # ── System 3: Paged kernel + prefix cache ──────────────────────────────
        paged_model_warm = PagedModel(embed, Wout)
        paged_runner_warm = _make_paged_runner(paged_model_warm, num_blocks=nb)

        s1, _ = paged_runner_warm.forward_prefill("req-1", full_prompt)
        paged_runner_warm.finish_request(s1)

        for i in range(N_WARMUP):
            s, r = paged_runner_warm.forward_prefill(f"ww{i}", req2_prompt)
            mx.eval(r.logits)
            paged_runner_warm.finish_request(s)

        warm_times: list[float] = []
        for i in range(N_ITERS):
            t0 = time.perf_counter()
            s, r = paged_runner_warm.forward_prefill(f"wp{i}", req2_prompt)
            mx.eval(r.logits)
            warm_times.append((time.perf_counter() - t0) * 1000)
            paged_runner_warm.finish_request(s)

        np_mean, _, _ = _stats(np_times)
        cold_mean, _, _ = _stats(cold_times)
        warm_mean, _, _ = _stats(warm_times)

        kernel_speedup = np_mean / cold_mean  # paged kernel vs non-paged
        full_speedup = np_mean / warm_mean  # full system vs non-paged
        cache_speedup = cold_mean / warm_mean  # prefix cache benefit alone

        rows.append(
            (
                prompt_len,
                np_mean,
                cold_mean,
                kernel_speedup,
                warm_mean,
                full_speedup,
                cache_speedup,
            )
        )
        print(
            f"{prompt_len:>8} | {np_mean:>9.2f} | {cold_mean:>11.2f} | "
            f"{kernel_speedup:>6.2f}x | {warm_mean:>11.2f} | "
            f"{full_speedup:>6.2f}x | {cache_speedup:>10.2f}x"
        )

    print()
    print(
        "  'vs NP kernel': speedup of paged-cold over non-paged  (kernel effect only)"
    )
    print("  'vs NP full':   speedup of paged-warm over non-paged  (kernel + cache)")
    print(
        "  'cache benefit': speedup of paged-warm over paged-cold (cache effect only)"
    )
    print()

    return rows


# ── Section B: TPOT — decode latency sweep across context lengths ─────────────


def bench_tpot():
    CTX_LENGTHS = [64, 128, 256, 512, 1024, 2048]
    N_WARMUP = 10  # steps thrown away before measuring
    N_ITERS = 30  # steps measured

    # Single paged runner sized for the largest context + warmup + measurement
    max_tokens = max(CTX_LENGTHS) + N_WARMUP + N_ITERS
    nb = 4 * math.ceil(max_tokens / BLOCK_SIZE)

    embed, Wout = _make_weights()
    np_model = NonPagedModel(embed, Wout)
    np_runner = NonPagedRunner(np_model)
    paged_model = PagedModel(embed, Wout)
    paged_runner = _make_paged_runner(paged_model, num_blocks=nb)

    print("=" * 78)
    print("Section B: TPOT Sweep — Decode Latency vs Context Length")
    print("=" * 78)
    print(
        f"Warmup: {N_WARMUP} steps discarded per context length.  "
        f"Measured: {N_ITERS} steps."
    )
    print()
    hdr = (
        f"{'Ctx Len':>9} | {'NP mean':>8} | {'NP tok/s':>9} | "
        f"{'Paged mean':>11} | {'Paged tok/s':>11} | {'Speedup':>8}"
    )
    print(hdr)
    print(f"{'':>9} | {'(ms)':>8} | {'':>9} | {'(ms)':>11} | {'':>11} | {'':>8}")
    print("-" * 78)

    rows: list[tuple] = []

    for ctx_len in CTX_LENGTHS:
        prompt = list(range(ctx_len))

        # ── Non-paged ──────────────────────────────────────────────────────────
        np_cache, _ = np_runner.forward_prefill(prompt)

        for i in range(N_WARMUP):
            np_runner.forward_decode(np_cache, i)

        np_times: list[float] = []
        for i in range(N_ITERS):
            t0 = time.perf_counter()
            np_runner.forward_decode(np_cache, i)
            np_times.append((time.perf_counter() - t0) * 1000)

        # ── Paged ──────────────────────────────────────────────────────────────
        p_state, _ = paged_runner.forward_prefill(f"ctx-{ctx_len}", prompt)

        for i in range(N_WARMUP):
            r = paged_runner.forward_decode(p_state, i)
            mx.eval(r.logits)

        paged_times: list[float] = []
        for i in range(N_ITERS):
            t0 = time.perf_counter()
            r = paged_runner.forward_decode(p_state, i)
            mx.eval(r.logits)
            paged_times.append((time.perf_counter() - t0) * 1000)

        paged_runner.finish_request(p_state)

        np_mean = sum(np_times) / N_ITERS
        paged_mean = sum(paged_times) / N_ITERS
        np_tps = 1000.0 / np_mean
        paged_tps = 1000.0 / paged_mean
        speedup = np_mean / paged_mean

        rows.append((ctx_len, np_mean, np_tps, paged_mean, paged_tps, speedup))
        print(
            f"{ctx_len:>9} | {np_mean:>8.2f} | {np_tps:>9.0f} | "
            f"{paged_mean:>11.2f} | {paged_tps:>11.0f} | {speedup:>7.2f}x"
        )

    print()
    print("  Non-paged: mx.concatenate(K/V) per step + contiguous SDPA.")
    print("  Paged:     reshape_and_cache (scatter-write) + paged_attention_v1.")
    print()

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    mx.set_default_device(mx.gpu)

    ttft_rows = bench_ttft()
    tpot_rows = bench_tpot()

    print("=" * 62)
    print("Summary")
    print("=" * 62)
    print("  TTFT: kernel vs cache vs full-system speedup")
    print(
        f"  {'Prompt':>8} | {'NP (ms)':>9} | {'Paged-cold':>11} | {'kernel':>8} | "
        f"{'Paged-warm':>11} | {'full':>8} | {'cache only':>11}"
    )
    for prompt_len, np_ms, cold_ms, ks, warm_ms, fs, cs in ttft_rows:
        print(
            f"  {prompt_len:>8} | {np_ms:>9.2f} | {cold_ms:>11.2f} | {ks:>7.2f}x | "
            f"{warm_ms:>11.2f} | {fs:>7.2f}x | {cs:>10.2f}x"
        )
    print()
    print("  TPOT speedup by context length:")
    for ctx_len, np_ms, np_tps, p_ms, p_tps, sp in tpot_rows:
        print(
            f"    ctx={ctx_len:>5}: {np_ms:.2f} ms ({np_tps:.0f} tok/s) → "
            f"{p_ms:.2f} ms ({p_tps:.0f} tok/s)  {sp:.2f}x"
        )
    print()
    print("Notes:")
    print("  - TTFT win: paged processes only suffix tokens after cache hit;")
    print("    non-paged recomputes the full prompt every request.")
    print("  - TPOT: non-paged pays mx.concatenate(O(N) copy) every decode step")
    print("    on top of O(N) SDPA; paged avoids the copy entirely.")
    print("  - Both advantages compound for longer contexts and deeper models.")


if __name__ == "__main__":
    main()
