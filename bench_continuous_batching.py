"""
bench_continuous_batching.py — Non-paged CB vs Paged CB.

The canonical baseline for continuous batching is not sequential processing
but non-paged continuous batching (Orca-style): requests are still batched
together each step, but KV is stored as contiguous tensors with mx.concatenate
rather than in paged blocks.

Two systems:
  Non-paged CB:  each request has a per-sequence SimpleKVCache.
                 On each decode step, projections run over the full batch (B, 1),
                 but attention is computed per-sequence (different ctx_lens prevent
                 batched SDPA without padding).  Grows O(B × ctx_len) per step.

  Paged CB:      BatchedPagedKVCache + paged_attention_v1.
                 Single kernel call handles all B sequences at their actual lengths.
                 No gather copy.  Attention cost stays flat as B grows.

Uses the synthetic BenchModel (4L / 8H / 4KVH / head_dim=128) so weight-loading
does not mask the attention cost difference.  Variable-length prompts (64–256
tokens) simulate realistic continuous batching workloads where different requests
join and leave at different times.

Metrics
-------
  Wall time:    total time to complete all N requests
  Throughput:   output tokens / wall time  (tok/s)
  Decode steps: number of forward passes (lower = more efficient batching)
  TTFT:         time from request submission to first generated token
"""

from __future__ import annotations

import math
import time
import types
import uuid
from collections import deque

import mlx.core as mx

from sglang_mlx.continuous_batching import ContinuousBatchingScheduler
from sglang_mlx.model_runner import ModelConfig, ModelRunner
from sglang_mlx.srt.mem_cache.base_prefix_cache import EvictParams

try:
    from sglang_mlx import _ext  # noqa: F401
except ImportError:
    print("ERROR: Metal extension not built. Run: pip install -e .")
    raise


# ── Config ─────────────────────────────────────────────────────────────────────

N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 128
VOCAB_SIZE = 2000
BLOCK_SIZE = 16

N_REQUESTS = 16
MAX_TOKENS = 64  # output tokens per request
MAX_BATCH_SIZE = 8

# Variable prompt lengths — realistic CB workload where requests have different
# context sizes.  Pattern repeats across N_REQUESTS.
PROMPT_LENGTHS = [
    64,
    128,
    96,
    256,
    128,
    64,
    192,
    256,
    96,
    128,
    64,
    192,
    256,
    128,
    96,
    64,
]


# ── Shared weights ─────────────────────────────────────────────────────────────


def _make_weights():
    D = N_HEADS * HEAD_DIM
    mx.random.seed(42)
    embed = (mx.random.normal((VOCAB_SIZE, D)) * 0.02).astype(mx.float16)
    Wout = (mx.random.normal((D, VOCAB_SIZE)) * 0.02).astype(mx.float16)
    mx.eval(embed, Wout)
    return embed, Wout


def _sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    if q.shape[1] != k.shape[1]:
        k = mx.repeat(k, q.shape[1] // k.shape[1], axis=1)
        v = mx.repeat(v, q.shape[1] // v.shape[1], axis=1)
    w = mx.softmax(mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale, axis=-1)
    return mx.matmul(w, v)


# ── Non-paged system ───────────────────────────────────────────────────────────


class SimpleKVCache:
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


class NonPagedModel:
    """Projections run over the full batch; attention is computed per-sequence.

    This is the honest non-paged CB approach: weight loading is amortised across
    the batch but SDPA cannot be batched because each sequence has a different
    context length.
    """

    def __init__(self, embed: mx.array, Wout: mx.array):
        self.embed = embed
        self.Wout = Wout
        self.scale = HEAD_DIM**-0.5

    def __call__(
        self,
        input_ids: mx.array,  # (B, S)
        caches: list[list[SimpleKVCache]] | None,  # [layer][seq] or None
    ) -> mx.array:
        B, S = input_ids.shape
        D, KVD = N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM
        x = self.embed[input_ids]  # (B, S, D)

        for li in range(N_LAYERS):
            # Projections — batched across all B sequences
            q = x.reshape(B, S, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            k = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            v = x[..., :KVD].reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

            if caches is None:
                attn = _sdpa(q, k, v, self.scale)
            else:
                # Attention — per-sequence (different ctx_lens prevent batched SDPA)
                attn_list = []
                for i, cache in enumerate(caches[li]):
                    k_full, v_full = cache.update_and_fetch(k[i : i + 1], v[i : i + 1])
                    attn_list.append(_sdpa(q[i : i + 1], k_full, v_full, self.scale))
                attn = mx.concatenate(attn_list, axis=0)  # (B, H, S, D)

            x = x + attn.transpose(0, 2, 1, 3).reshape(B, S, D)

        return x @ self.Wout  # (B, S, V)


class NonPagedCBScheduler:
    """Continuous batching scheduler using contiguous per-sequence KV caches."""

    def __init__(self, model: NonPagedModel, max_batch_size: int = 8):
        self._model = model
        self._max_batch_size = max_batch_size
        self._waiting: deque = deque()
        self._active: list[
            dict
        ] = []  # {tokens, kv_caches, next_token, max_tokens, req_id}

    def submit(
        self, prompt: list[int], max_tokens: int, req_id: str | None = None
    ) -> str:
        rid = req_id or uuid.uuid4().hex
        self._waiting.append(
            {"prompt": prompt, "max_tokens": max_tokens, "req_id": rid}
        )
        return rid

    def step(self) -> list[dict]:
        # Phase 1 — Admit: prefill each new request individually
        while self._waiting and len(self._active) < self._max_batch_size:
            req = self._waiting.popleft()
            [
                [SimpleKVCache() for _ in range(len(self._waiting) + 1)]
                for _ in range(N_LAYERS)
            ]
            # Prefill with a fresh set of caches for this single sequence
            [[SimpleKVCache() for _ in range(N_LAYERS)]]
            # One sequence: shape (1, prompt_len)
            inp = mx.array([req["prompt"]])
            # Build per-layer cache list: [[cache_seq0], [cache_seq0], ...]
            layer_caches = [[SimpleKVCache()] for _ in range(N_LAYERS)]
            logits = self._model(inp, layer_caches)
            mx.eval(logits)
            next_tok = int(mx.argmax(logits[0, -1]).item())
            self._active.append(
                {
                    "req_id": req["req_id"],
                    "kv_caches": layer_caches,  # list[layer] of list[seq=1]
                    "next_token": next_tok,
                    "output_len": 0,
                    "max_tokens": req["max_tokens"],
                }
            )

        if not self._active:
            return []

        len(self._active)

        # Phase 2 — Batched decode
        # Build per-layer KV cache list: caches[layer] = [cache_seq0, cache_seq1, ...]
        batched_caches = [
            [req["kv_caches"][li][0] for req in self._active] for li in range(N_LAYERS)
        ]
        token_ids = [req["next_token"] for req in self._active]
        inp = mx.array([[t] for t in token_ids])  # (B, 1)
        logits = self._model(inp, batched_caches)  # (B, 1, V)
        mx.eval(logits)

        # Phase 3 — Sample, check stopping
        completed = []
        next_active = []
        for i, req in enumerate(self._active):
            next_tok = int(mx.argmax(logits[i, 0]).item())
            req["output_len"] += 1
            req["next_token"] = next_tok
            if req["output_len"] >= req["max_tokens"]:
                completed.append(
                    {"req_id": req["req_id"], "output_len": req["output_len"]}
                )
            else:
                next_active.append(req)

        self._active = next_active
        return completed

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    @property
    def active(self) -> int:
        return len(self._active)


# ── Paged system ───────────────────────────────────────────────────────────────


class PagedModel:
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


# ── Benchmark helpers ──────────────────────────────────────────────────────────


def _percentile(times: list[float], p: float) -> float:
    s = sorted(times)
    return s[min(int(len(s) * p), len(s) - 1)]


def bench_nonpaged_cb(
    model: NonPagedModel,
    prompts: list[list[int]],
    max_batch_size: int,
) -> dict:
    scheduler = NonPagedCBScheduler(model, max_batch_size=max_batch_size)

    submit_times: dict[str, float] = {}
    ttft_times: list[float] = []
    total_output = 0
    seen_active: set[str] = set()

    t_wall = time.perf_counter()
    for i, prompt in enumerate(prompts):
        rid = f"np-{i}"
        submit_times[rid] = time.perf_counter()
        scheduler.submit(prompt, max_tokens=MAX_TOKENS, req_id=rid)

    step_count = 0
    while scheduler.waiting or scheduler.active:
        completed = scheduler.step()
        step_count += 1

        for req in scheduler._active:
            if req["req_id"] not in seen_active:
                seen_active.add(req["req_id"])
                ttft_times.append(
                    (time.perf_counter() - submit_times[req["req_id"]]) * 1000
                )
        for c in completed:
            total_output += c["output_len"]
            if c["req_id"] not in seen_active:
                seen_active.add(c["req_id"])
                ttft_times.append(
                    (time.perf_counter() - submit_times[c["req_id"]]) * 1000
                )

    wall_ms = (time.perf_counter() - t_wall) * 1000
    return {
        "wall_ms": wall_ms,
        "tps": total_output / (wall_ms / 1000),
        "rps": len(prompts) / (wall_ms / 1000),
        "steps": step_count,
        "ttft_mean": sum(ttft_times) / len(ttft_times) if ttft_times else 0,
        "ttft_p95": _percentile(ttft_times, 0.95) if ttft_times else 0,
    }


def bench_paged_cb(
    runner: ModelRunner,
    prompts: list[list[int]],
    max_batch_size: int,
) -> dict:
    scheduler = ContinuousBatchingScheduler(runner, max_batch_size=max_batch_size)

    submit_times: dict[str, float] = {}
    ttft_times: list[float] = []
    total_output = 0
    seen_active: set[str] = set()

    t_wall = time.perf_counter()
    for i, prompt in enumerate(prompts):
        rid = f"pg-{i}"
        submit_times[rid] = time.perf_counter()
        scheduler.submit(prompt, max_tokens=MAX_TOKENS, request_id=rid)

    step_count = 0
    while scheduler.waiting or scheduler.active:
        completed = scheduler.step()
        step_count += 1

        for req in scheduler._active:
            if req.request_id not in seen_active:
                seen_active.add(req.request_id)
                ttft_times.append(
                    (time.perf_counter() - submit_times[req.request_id]) * 1000
                )
        for c in completed:
            total_output += len(c.output_tokens)
            if c.request_id not in seen_active:
                seen_active.add(c.request_id)
                ttft_times.append(
                    (time.perf_counter() - submit_times[c.request_id]) * 1000
                )

    wall_ms = (time.perf_counter() - t_wall) * 1000
    return {
        "wall_ms": wall_ms,
        "tps": total_output / (wall_ms / 1000),
        "rps": len(prompts) / (wall_ms / 1000),
        "steps": step_count,
        "ttft_mean": sum(ttft_times) / len(ttft_times) if ttft_times else 0,
        "ttft_p95": _percentile(ttft_times, 0.95) if ttft_times else 0,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    mx.set_default_device(mx.gpu)

    prompts = [list(range(p_len)) for p_len in PROMPT_LENGTHS]
    avg_len = sum(PROMPT_LENGTHS) / len(PROMPT_LENGTHS)
    total_out = N_REQUESTS * MAX_TOKENS

    print(
        f"Model:           {N_LAYERS}L / {N_HEADS}H / {N_KV_HEADS}KVH / "
        f"head_dim={HEAD_DIM} / float16"
    )
    print(
        f"Requests:        {N_REQUESTS}  (prompt lengths: "
        f"{sorted(set(PROMPT_LENGTHS))} tokens)"
    )
    print(f"Avg prompt len:  {avg_len:.0f} tokens")
    print(f"Output tokens:   {MAX_TOKENS} per request  ({total_out} total)")
    print(f"Max batch size:  {MAX_BATCH_SIZE}")
    print()

    embed, Wout = _make_weights()

    # Paged runner — sized for worst case: all requests fully decoded
    max_tokens_total = max(PROMPT_LENGTHS) + MAX_TOKENS
    nb = 4 * math.ceil(max_tokens_total / BLOCK_SIZE) * MAX_BATCH_SIZE
    np_model = NonPagedModel(embed, Wout)
    paged_model = PagedModel(embed, Wout)
    runner = _make_paged_runner(paged_model, num_blocks=nb)

    # Warmup — JIT-compile all kernels
    print("Warming up...")
    _s, _r = runner.forward_prefill("w", prompts[0][:8])
    mx.eval(_r.logits)
    _t = int(mx.argmax(_r.logits).item())
    for _ in range(4):
        _r = runner.forward_decode(_s, _t)
        mx.eval(_r.logits)
    runner.finish_request(_s)
    runner._radix_cache.evict(EvictParams(num_tokens=999_999))

    _inp = mx.array([prompts[0][:8]])
    _lc = [[SimpleKVCache()] for _ in range(N_LAYERS)]
    _out = np_model(_inp, _lc)
    mx.eval(_out)
    print("Done.\n")

    # Run benchmarks
    print("Running non-paged CB...")
    np_result = bench_nonpaged_cb(np_model, prompts, MAX_BATCH_SIZE)

    print("Running paged CB...")
    runner._radix_cache.evict(EvictParams(num_tokens=999_999))
    pg_result = bench_paged_cb(runner, prompts, MAX_BATCH_SIZE)

    # Print results
    print()
    print("=" * 68)
    print(
        f"Results — {N_REQUESTS} requests × {MAX_TOKENS} output tokens "
        "(synthetic BenchModel)"
    )
    print("=" * 68)
    print(f"{'Metric':<28} {'Non-paged CB':>14} {'Paged CB':>10} {'Speedup':>12}")
    print("-" * 68)

    def row(label, np_val, pg_val, fmt=".1f", lower_is_better=False):
        if lower_is_better:
            ratio = np_val / pg_val if pg_val > 0 else float("nan")
            tag = f"{ratio:.2f}x faster" if ratio > 1 else f"{1 / ratio:.2f}x slower"
        else:
            ratio = pg_val / np_val if np_val > 0 else float("nan")
            tag = f"{ratio:.2f}x"
        print(f"{label:<28} {np_val:>14{fmt}} {pg_val:>10{fmt}} {tag:>12}")

    row(
        "TTFT mean (ms)",
        np_result["ttft_mean"],
        pg_result["ttft_mean"],
        lower_is_better=True,
    )
    row(
        "TTFT p95 (ms)",
        np_result["ttft_p95"],
        pg_result["ttft_p95"],
        lower_is_better=True,
    )
    row(
        "Wall time (ms)",
        np_result["wall_ms"],
        pg_result["wall_ms"],
        lower_is_better=True,
    )
    row("Throughput (tok/s)", np_result["tps"], pg_result["tps"])
    row("Req/s", np_result["rps"], pg_result["rps"])
    print(f"  {'Decode steps':<26} {np_result['steps']:>14}  {pg_result['steps']:>10}")

    print()
    print("Notes:")
    print("  Non-paged CB: projections batched over B sequences; attention computed")
    print("  per-sequence (different ctx_lens prevent batched SDPA without padding).")
    print("  Paged CB: projections + paged_attention_v1 both run over full batch.")
    print(
        "  Synthetic model: attention cost is fully visible; "
        "no weight-loading bottleneck."
    )


if __name__ == "__main__":
    main()
