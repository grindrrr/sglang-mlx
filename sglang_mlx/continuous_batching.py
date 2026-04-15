"""
continuous_batching.py — Continuous batching scheduler for paged attention.

Orchestrates a request queue on top of ModelRunner, enabling multiple
requests to share each decode step:

  1. Admit waiting requests (prefill) whenever batch space and memory allow.
  2. Run one forward_decode_batch step for all active requests.
  3. Remove completed requests (EOS or max_tokens reached), free their blocks.
  4. Repeat until the queue and active set are both empty.

The key throughput benefit: model weights are loaded from RAM once per step
regardless of batch size. On memory-bandwidth-bound Apple Silicon hardware,
doubling the batch size roughly doubles output token throughput.

Usage
-----
    runner    = ModelRunner.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    scheduler = ContinuousBatchingScheduler(runner, max_batch_size=4)

    for prompt in prompts:
        scheduler.submit(tokenizer.encode(prompt), max_tokens=128)

    results = scheduler.run_until_done()   # dict[request_id → list[int]]
"""

from __future__ import annotations

import dataclasses
import math
import uuid
from collections import deque
from collections.abc import Callable

import mlx.core as mx

from sglang_mlx.model_runner import ForwardResult, ModelRunner, RequestState

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclasses.dataclass
class InferenceRequest:
    """A request in the waiting queue or active in the decode batch."""

    request_id: str
    prompt_tokens: list[int]
    max_tokens: int
    sampler: Callable[[mx.array], int]
    # Populated after prefill:
    state: RequestState | None = None
    next_token: int | None = None  # token sampled from prefill logits


@dataclasses.dataclass
class CompletedRequest:
    """A finished request returned by step()."""

    request_id: str
    output_tokens: list[int]  # does NOT include the EOS token itself


# ── Scheduler ─────────────────────────────────────────────────────────────────


class ContinuousBatchingScheduler:
    """Continuous batching scheduler for PagedMHAPool-backed models.

    On each ``step()``:
      * Admits waiting requests (prefills them) as long as batch slots and
        memory are available.
      * Runs ``forward_decode_batch`` for all active requests in one pass.
      * Collects requests that hit EOS or their ``max_tokens`` limit.

    Args:
        runner:        A ``ModelRunner`` instance (paged pool already allocated).
        max_batch_size: Maximum number of sequences decoded simultaneously.
        stop_tokens:   Set of token IDs that terminate generation (e.g. EOS).
    """

    def __init__(
        self,
        runner: ModelRunner,
        max_batch_size: int = 8,
        stop_tokens: set[int] | None = None,
    ):
        self._runner = runner
        self._max_batch_size = max_batch_size
        self._stop_tokens: set[int] = stop_tokens or set()
        self._waiting: deque[InferenceRequest] = deque()
        self._active: list[InferenceRequest] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 256,
        sampler: Callable[[mx.array], int] | None = None,
        request_id: str | None = None,
    ) -> str:
        """Enqueue a request for processing.

        Args:
            prompt_tokens: Tokenised prompt.
            max_tokens:    Maximum number of output tokens to generate.
            sampler:       Token-sampling function ``(logits) → token_id``.
                           Defaults to greedy argmax.
            request_id:    Optional caller-supplied ID; auto-generated if None.

        Returns:
            The request_id string.
        """
        if sampler is None:
            sampler = _argmax
        rid = request_id or uuid.uuid4().hex
        self._waiting.append(
            InferenceRequest(
                request_id=rid,
                prompt_tokens=list(prompt_tokens),
                max_tokens=max_tokens,
                sampler=sampler,
            )
        )
        return rid

    def step(self) -> list[CompletedRequest]:
        """Run one scheduler iteration.

        Returns:
            List of requests that completed (EOS or max_tokens) this step.
        """
        # Phase 1 — Admit waiting requests
        self._admit()

        # Nothing to do
        if not self._active:
            return []

        # Phase 2 — Batched decode (one forward pass for all active requests)
        states = [req.state for req in self._active]
        token_ids = [req.next_token for req in self._active]  # type: ignore[misc]

        results: list[ForwardResult] = self._runner.forward_decode_batch(
            states, token_ids
        )

        # Phase 3 — Sample, check stopping criteria
        completed: list[CompletedRequest] = []
        next_active: list[InferenceRequest] = []

        for req, result in zip(self._active, results):
            next_tok = req.sampler(result.logits)

            is_eos = next_tok in self._stop_tokens
            is_maxed = len(req.state.output_tokens) >= req.max_tokens  # type: ignore[union-attr]

            if is_eos or is_maxed:
                self._runner.finish_request(req.state)  # type: ignore[arg-type]
                completed.append(
                    CompletedRequest(
                        request_id=req.request_id,
                        output_tokens=list(req.state.output_tokens),  # type: ignore[union-attr]
                    )
                )
            else:
                req.next_token = next_tok
                next_active.append(req)

        self._active = next_active
        return completed

    def run_until_done(self) -> dict[str, list[int]]:
        """Process all submitted requests to completion.

        Returns:
            ``{request_id: output_tokens}`` for every submitted request.
        """
        results: dict[str, list[int]] = {}
        while self._waiting or self._active:
            for completed in self.step():
                results[completed.request_id] = completed.output_tokens
        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def waiting(self) -> int:
        """Number of requests in the waiting queue."""
        return len(self._waiting)

    @property
    def active(self) -> int:
        """Number of requests currently being decoded."""
        return len(self._active)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _admit(self) -> None:
        """Prefill waiting requests until the batch is full or memory is tight."""
        runner = self._runner

        while self._waiting and len(self._active) < self._max_batch_size:
            req = self._waiting[0]  # peek

            # Estimate blocks needed for this prompt.
            # The check is pessimistic (ignores potential radix cache hits) so
            # actual consumption will be ≤ estimate — safe for admission control.
            blocks_needed = math.ceil(len(req.prompt_tokens) / runner._block_size)
            # Reserve one extra block per active sequence for their next decode
            reserved = len(self._active) + 1
            if runner._allocator.available_blocks < blocks_needed + reserved:
                break  # not enough memory; retry next step

            self._waiting.popleft()

            state, result = runner.forward_prefill(req.request_id, req.prompt_tokens)
            req.state = state
            req.next_token = req.sampler(result.logits)
            self._active.append(req)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _argmax(logits: mx.array) -> int:
    return int(mx.argmax(logits).item())
