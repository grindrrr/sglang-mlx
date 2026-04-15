"""
bench_end_to_end.py — End-to-end ModelRunner benchmarks with a real mlx-lm model.

Loads mlx-community/Llama-3.2-1B-Instruct-4bit and exercises the full
sglang-mlx stack:
  - Radix cache prefix reuse (TTFT speedup from skipping recomputation)
  - Decode with paged KV cache using paged_attention_v1 kernel

ModelRunner.load() automatically patches attention layers so single-token
decode steps use store() + compute_paged_attn() (paged_attention_v1 kernel)
instead of update_and_fetch() (scatter-write + gather + mx.fast.SDPA).
Prefill (L > 1) still uses update_and_fetch; eliminating the gather there
requires the paged_attention_prefill kernel (FUTURE_OPTIMIZATIONS.md §7).

Sections
--------
1. TTFT vs prefix cache hit rate (0 / 50 / ~94 %)
2. TPOT decode curve: latency at context positions 32, 64, 128 tokens past prefill
"""

from __future__ import annotations

import math
import time
from typing import Any

import mlx.core as mx

from sglang_mlx.model_runner import ModelRunner
from sglang_mlx.srt.mem_cache.base_prefix_cache import EvictParams

try:
    from sglang_mlx import _ext  # noqa: F401 — presence check only
except ImportError:
    print("ERROR: Metal extension not built. Run: pip install -e .")
    raise


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
BLOCK_SIZE = 16

# System prompt acts as the shared prefix across all requests.
# Chosen to be long enough (~200 tokens) to give meaningful prefix reuse numbers.
_SYSTEM = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. Please ensure that your responses "
    "are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)
_USER = "Explain the concept of quantum entanglement in simple terms."


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_prompt(tokenizer: Any, system: str, user: str) -> list[int]:
    """Encode a system+user exchange using the tokenizer's chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"{system}\n\nUser: {user}\nAssistant:"
    return tokenizer.encode(text)


def _percentile(times: list[float], p: float) -> float:
    s = sorted(times)
    return s[min(int(len(s) * p), len(s) - 1)]


# ── Section 1: TTFT vs cache hit rate ─────────────────────────────────────────


def bench_ttft(runner: ModelRunner, full_prompt: list[int]) -> None:
    """Measure TTFT at 0 / 50 / ~94 % prefix cache hit rates.

    'Prefix' is the tokenised system prompt.  The ~94 % hit case leaves only
    the final block (16 tokens) to recompute — forcing the model to produce
    fresh logits while reusing almost all the KV computation.
    """
    print("=" * 60)
    print("Section 1: TTFT vs Prefix Cache Hit Rate")
    print("=" * 60)

    N = len(full_prompt)
    split_50 = N // 2
    split_94 = N - BLOCK_SIZE  # leave exactly one block as the unique new part

    prefix_50 = full_prompt[:split_50]
    prefix_94 = full_prompt[:split_94]

    print(f"Full prompt:  {N} tokens")
    print(
        f"  0%  hit: no cache              → all "
        f"{math.ceil(N / BLOCK_SIZE)} blocks recomputed"
    )
    print(
        f"  50% hit: {split_50} tokens cached   → "
        f"{math.ceil((N - split_50) / BLOCK_SIZE)} blocks recomputed"
    )
    print(
        f" ~94% hit: {split_94} tokens cached  → 1 block "
        f"({BLOCK_SIZE} tokens) recomputed"
    )
    print()

    N_WARMUP = 3
    N_ITERS = 8

    def run_once(tokens: list[int], req_id: str) -> None:
        state, result = runner.forward_prefill(req_id, tokens)
        mx.eval(result.logits)
        runner.finish_request(state)

    # ── 0 % hit ───────────────────────────────────────────────────────────────
    cold_times: list[float] = []
    for i in range(N_WARMUP + N_ITERS):
        runner._radix_cache.evict(EvictParams(num_tokens=999_999))
        t0 = time.perf_counter()
        run_once(full_prompt, f"cold-{i}")
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= N_WARMUP:
            cold_times.append(elapsed)

    # ── 50 % hit ─────────────────────────────────────────────────────────────
    partial_times: list[float] = []
    for i in range(N_WARMUP + N_ITERS):
        runner._radix_cache.evict(EvictParams(num_tokens=999_999))
        run_once(prefix_50, "setup-50")
        t0 = time.perf_counter()
        run_once(full_prompt, f"partial-{i}")
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= N_WARMUP:
            partial_times.append(elapsed)

    # ── ~94 % hit ─────────────────────────────────────────────────────────────
    run_once(prefix_94, "populate-94")  # cold-populate shared portion once
    warm_times: list[float] = []
    for i in range(N_WARMUP + N_ITERS):
        t0 = time.perf_counter()
        run_once(full_prompt, f"warm-{i}")
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= N_WARMUP:
            warm_times.append(elapsed)

    def stats(times: list[float]) -> tuple[float, float]:
        mean = sum(times) / len(times)
        p95 = _percentile(times, 0.95)
        return mean, p95

    cold_mean, cold_p95 = stats(cold_times)
    partial_mean, partial_p95 = stats(partial_times)
    warm_mean, warm_p95 = stats(warm_times)

    print(f"{'Hit rate':>12} | {'mean (ms)':>10} | {'p95 (ms)':>9} | {'Speedup':>8}")
    print("-" * 50)
    print(f"{'0%  (cold)':>12} | {cold_mean:>10.2f} | {cold_p95:>9.2f} | {'1.00x':>8}")
    print(
        f"{'50%':>12} | {partial_mean:>10.2f} | {partial_p95:>9.2f} | "
        f"{cold_mean / partial_mean:>7.2f}x"
    )
    print(
        f"{'~94% (warm)':>12} | {warm_mean:>10.2f} | {warm_p95:>9.2f} | "
        f"{cold_mean / warm_mean:>7.2f}x"
    )
    print()


# ── Section 2: TPOT decode curve ──────────────────────────────────────────────


def bench_tpot(runner: ModelRunner, prompt: list[int]) -> None:
    """Measure time-per-output-token as context grows past the prefill point."""
    print("=" * 60)
    print("Section 2: TPOT Decode Curve")
    print("=" * 60)
    print(f"Prefill: {len(prompt)} tokens.  Measuring decode latency as context grows.")
    print()

    MAX_DECODE = 128
    REPORT_AT = [32, 64, 128]
    WINDOW = 4

    # Warmup — JIT-compile kernels before timing
    warmup_state, _ = runner.forward_prefill("warmup", prompt[:8])
    for i in range(8):
        r = runner.forward_decode(warmup_state, i % 1000)
        mx.eval(r.logits)
    runner.finish_request(warmup_state)

    # Main run: prefill, then MAX_DECODE greedy decode steps
    state, r0 = runner.forward_prefill("tpot-run", prompt)
    mx.eval(r0.logits)

    step_times: dict[int, float] = {}
    token = int(mx.argmax(r0.logits).item())
    for step in range(MAX_DECODE):
        seq_pos = len(prompt) + step + 1  # context length after this step
        t0 = time.perf_counter()
        result = runner.forward_decode(state, token)
        mx.eval(result.logits)
        step_times[seq_pos] = (time.perf_counter() - t0) * 1000
        token = int(mx.argmax(result.logits).item())

    runner.finish_request(state)

    all_pos = sorted(step_times.keys())
    print(f"{'Steps past prefill':>18} | {'TPOT mean (ms)':>15} | {'~tok/s':>8}")
    print("-" * 50)
    for checkpoint in REPORT_AT:
        target = len(prompt) + checkpoint
        window_keys = [p for p in all_pos if abs(p - target) <= WINDOW]
        if not window_keys:
            continue
        avg = sum(step_times[p] for p in window_keys) / len(window_keys)
        tps = 1000.0 / avg
        print(f"{f'+{checkpoint} tokens':>18} | {avg:>15.2f} | {tps:>8.0f}")
    print()


# ── Section 3: Multi-request throughput ───────────────────────────────────────

# Distinct user questions — each becomes a separate request that shares _SYSTEM.
_USER_QUESTIONS = [
    "Explain the concept of quantum entanglement in simple terms.",
    "What is the difference between machine learning and deep learning?",
    "How does photosynthesis work at the molecular level?",
    "What are the main causes of climate change?",
    "Explain how the internet works from a high level.",
    "What is the significance of the Turing test in AI?",
    "How do vaccines train the immune system?",
    "What is the difference between fusion and fission in nuclear physics?",
    "Explain the concept of supply and demand in economics.",
    "How does GPS determine your location?",
]

_DECODE_STEPS = 32  # output tokens per request


def bench_multi_request(runner: ModelRunner) -> None:
    """Simulate N sequential requests sharing a common system-prompt prefix.

    Request 1 is cold (populates the radix cache). Requests 2-N hit the cache
    on the shared system prompt and skip its recomputation. Measures per-request
    TTFT and decode wall time, plus total throughput.
    """
    print("=" * 60)
    print("Section 3: Multi-Request Throughput (Shared System Prompt)")
    print("=" * 60)
    tokenizer = runner._tokenizer
    N = len(_USER_QUESTIONS)
    print(f"  {N} requests, each with the shared system prompt + unique user question.")
    print(f"  {_DECODE_STEPS} decode tokens per request.")
    print()

    prompts = [_build_prompt(tokenizer, _SYSTEM, q) for q in _USER_QUESTIONS]
    len(_build_prompt(tokenizer, _SYSTEM, ""))

    # Warmup pass to JIT-compile all kernels
    warmup_state, _ = runner.forward_prefill("warmup-mr", prompts[0][:8])
    for i in range(4):
        runner.forward_decode(warmup_state, i % 1000)
    runner.finish_request(warmup_state)
    runner._radix_cache.evict(
        __import__(
            "sglang_mlx.srt.mem_cache.base_prefix_cache", fromlist=["EvictParams"]
        ).EvictParams(num_tokens=999_999)
    )

    ttft_times: list[float] = []
    decode_times: list[float] = []  # total decode wall time per request
    wall_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        # TTFT
        t0 = time.perf_counter()
        state, result = runner.forward_prefill(f"mr-{i}", prompt)
        mx.eval(result.logits)
        ttft_times.append((time.perf_counter() - t0) * 1000)

        # Decode
        token = int(mx.argmax(result.logits).item())
        t_dec = time.perf_counter()
        for _ in range(_DECODE_STEPS):
            result = runner.forward_decode(state, token)
            mx.eval(result.logits)
            token = int(mx.argmax(result.logits).item())
        decode_times.append((time.perf_counter() - t_dec) * 1000)

        runner.finish_request(state)

    total_wall = (time.perf_counter() - wall_start) * 1000

    print(
        f"  {'Req':>4} | {'TTFT (ms)':>10} | {'Decode (ms)':>12} | {'tok/s':>7} | Note"
    )
    print("  " + "-" * 52)
    for i, (ttft, dec) in enumerate(zip(ttft_times, decode_times)):
        tps = _DECODE_STEPS / (dec / 1000)
        note = "cold (populates cache)" if i == 0 else "cache hit"
        print(f"  {i + 1:>4} | {ttft:>10.1f} | {dec:>12.1f} | {tps:>7.0f} | {note}")

    print()
    ttft_cold = ttft_times[0]
    ttft_warm_mean = sum(ttft_times[1:]) / (N - 1)
    dec_mean = sum(decode_times) / N
    total_output_tokens = N * _DECODE_STEPS
    throughput = total_output_tokens / (total_wall / 1000)

    print(f"  TTFT  — req 1 (cold):      {ttft_cold:>8.1f} ms")
    print(
        f"  TTFT  — reqs 2-{N} (warm):  {ttft_warm_mean:>8.1f} ms  "
        f"({ttft_cold / ttft_warm_mean:.1f}x faster than cold)"
    )
    print(
        f"  Decode mean per request:   {dec_mean:>8.1f} ms  "
        f"({_DECODE_STEPS / (dec_mean / 1000):.0f} tok/s)"
    )
    print(f"  Total wall time ({N} reqs):  {total_wall:>8.1f} ms")
    print(
        f"  Throughput:                {throughput:>8.1f} tok/s  "
        f"({N / (total_wall / 1000):.2f} req/s)"
    )
    print()


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    mx.set_default_device(mx.gpu)

    print(f"Loading {MODEL_ID} ...")
    runner = ModelRunner.load(MODEL_ID, block_size=BLOCK_SIZE)
    tokenizer = runner._tokenizer
    print("Model loaded.\n")

    full_prompt = _build_prompt(tokenizer, _SYSTEM, _USER)
    N = len(full_prompt)
    print(
        f"Prompt: {N} tokens  (pool: {runner._pool.num_blocks} blocks × "
        f"{BLOCK_SIZE} = {runner._pool.num_blocks * BLOCK_SIZE} token capacity)\n"
    )

    bench_ttft(runner, full_prompt)
    bench_tpot(runner, full_prompt)
    bench_multi_request(runner)

    print("=" * 60)
    print("Notes")
    print("=" * 60)
    print(f"  Model:      {MODEL_ID}")
    print(f"  Block size: {BLOCK_SIZE} tokens/block")
    print()
    print("  TTFT speedup: radix cache skips recomputing the cached prefix.")
    print("  The ~94% hit case recomputes only the last 16-token block.")
    print()
    print("  Decode (TPOT): attention layers are patched by ModelRunner.load().")
    print("  Single-token steps use store() + paged_attention_v1 kernel directly —")
    print("  no gather copy. Prefill (L > 1) still uses update_and_fetch.")
    print("  (see FUTURE_OPTIMIZATIONS.md §7 for the prefill fix.)")


if __name__ == "__main__":
    main()
