"""Model runner for MLX-based inference.

Bridges mlx-lm models with the radix cache and paged KV pool, enabling
prefill (process prompt with prefix reuse) and decode (generate one
token at a time).

Components:
- ModelConfig: extracted model dimensions
- RequestState: mutable per-request state (blocks, tokens, caches)
- ForwardResult: logits from a single forward pass
- ModelRunner: orchestrates loading, cache management, and forward passes
"""

from __future__ import annotations

import dataclasses
import enum
import math
import uuid
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from sglang_mlx.srt.mem_cache.base_prefix_cache import InsertParams
from sglang_mlx.srt.mem_cache.paged_pool import (
    BatchedPagedKVCache,
    BlockAllocator,
    PagedKVCache,
    PagedMHAPool,
)
from sglang_mlx.srt.mem_cache.radix_cache import RadixCache


class ForwardMode(enum.Enum):
    """Whether we're processing a prompt or generating tokens."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclasses.dataclass
class ModelConfig:
    """Extracted model configuration needed by the runner."""

    num_layers: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    sliding_window: int | None = None


@dataclasses.dataclass
class RequestState:
    """Mutable state tracking an in-flight request."""

    request_id: str
    prompt_tokens: list[int]
    output_tokens: list[int]
    all_block_ids: list[int]  # physical block IDs in sequence order
    radix_node: Any  # for lock/unlock
    prefix_len: int  # tokens reused from radix cache
    kv_caches: list[PagedKVCache]

    @property
    def seq_len(self) -> int:
        """Total sequence length (prompt + generated)."""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def total_cached_len(self) -> int:
        """Number of allocated blocks (= tokens when block_size=1)."""
        return len(self.all_block_ids)


@dataclasses.dataclass
class ForwardResult:
    """Result of a single forward pass."""

    logits: mx.array  # (vocab_size,) for last position
    forward_mode: ForwardMode


def _extract_model_config(model: Any, config_dict: dict | None = None) -> ModelConfig:
    """Extract model configuration from an mlx-lm model.

    Reads from ``model.args`` (mlx-lm convention) first, falls back to
    *config_dict* for missing fields.
    """
    if config_dict is None:
        config_dict = {}

    args = getattr(model, "args", None)

    def _get(name: str, default: Any = None) -> Any:
        if args is not None and hasattr(args, name):
            return getattr(args, name)
        return config_dict.get(name, default)

    num_layers: int = (
        len(model.layers) if hasattr(model, "layers") else _get("num_hidden_layers")
    )
    num_attention_heads: int = _get("num_attention_heads")
    hidden_size: int = _get("hidden_size")
    head_dim: int | None = _get("head_dim")
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads
    n_kv_heads: int = _get("num_key_value_heads") or num_attention_heads
    vocab_size: int = _get("vocab_size")
    sliding_window: int | None = _get("sliding_window")

    return ModelConfig(
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sliding_window=sliding_window,
    )


class ModelRunner:
    """Orchestrates model loading, KV cache management, and forward passes.

    Owns the paged KV pool, block allocator, and radix cache. Provides methods
    for prefill (prompt processing with prefix reuse), decode (one token at a
    time), and a convenience ``generate`` loop.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ModelConfig,
        num_blocks: int = 512,
        block_size: int = 1,
        dtype: mx.Dtype = mx.float16,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._block_size = block_size
        self._dtype = dtype

        self._allocator = BlockAllocator(num_blocks, block_size)
        self._pool = PagedMHAPool(
            num_blocks=num_blocks,
            num_layers=config.num_layers,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            dtype=dtype,
        )
        self._radix_cache = RadixCache()
        self._radix_cache.set_allocator(self._allocator)
        self._active_requests: dict[str, RequestState] = {}

    @classmethod
    def load(
        cls,
        model_path: str,
        num_blocks: int = 512,
        block_size: int = 16,
        dtype: mx.Dtype = mx.float16,
    ) -> ModelRunner:
        """Load a model from a local path or HuggingFace Hub ID.

        Automatically patches attention layers for paged decode: single-token
        steps use store() + compute_paged_attn() instead of update_and_fetch(),
        eliminating the O(num_blocks) gather on every decode step.
        """
        from mlx_lm import load as mlx_load

        model, tokenizer = mlx_load(model_path)
        config = _extract_model_config(model)
        runner = cls(model, tokenizer, config, num_blocks, block_size, dtype)
        n_patched = _patch_for_paged_decode(model)
        if n_patched:
            import logging

            logging.getLogger(__name__).debug(
                "Patched %d attention layer(s) for paged decode.", n_patched
            )
        return runner

    # ── Core forward methods ──────────────────────────────────────────────────

    def forward_prefill(
        self, request_id: str, prompt_tokens: list[int]
    ) -> tuple[RequestState, ForwardResult]:
        """Process prompt tokens, reusing any cached prefix.

        Returns:
            ``(RequestState, ForwardResult)`` — the state to pass to
            ``forward_decode``, and logits for sampling the first output token.
        """
        # 1. Match prefix in radix cache
        #    Values are stored as block IDs repeated per token; deduplicate to
        #    get ordered unique block IDs.
        match = self._radix_cache.match_prefix(prompt_tokens)
        prefix_blocks = list(dict.fromkeys(match.device_indices))
        prefix_tokens = len(prefix_blocks) * self._block_size

        # 2. Handle full cache hit: reprocess last block for fresh logits
        if prefix_tokens >= len(prompt_tokens) and prefix_tokens > 0:
            reused_block = [prefix_blocks[-1]]
            prefix_blocks = prefix_blocks[:-1]
            prefix_tokens = len(prefix_blocks) * self._block_size
            new_tokens = prompt_tokens[prefix_tokens:]
            new_blocks = reused_block
        else:
            new_tokens = prompt_tokens[prefix_tokens:]
            new_blocks = self._allocator.alloc_for_tokens(len(new_tokens))

        # 3. Create per-layer PagedKVCache
        kv_caches: list[PagedKVCache] = []
        for i in range(self._config.num_layers):
            cache = PagedKVCache(self._pool, layer_idx=i)
            cache.set_blocks(prefix_blocks, new_blocks, prefix_tokens)
            kv_caches.append(cache)

        # 4. Run model forward — single mx.eval sync point
        input_ids = mx.array([new_tokens])  # (1, N)
        logits = self._model(input_ids, cache=kv_caches)  # (1, N, V)
        mx.eval(logits)

        # 5. Insert into radix cache — store block IDs at token granularity
        #    (one repeated block_id per token) to preserve token-level matching
        all_blocks = prefix_blocks + new_blocks
        block_ids_expanded = [
            all_blocks[i // self._block_size] for i in range(len(prompt_tokens))
        ]
        self._radix_cache.insert(
            InsertParams(key=list(prompt_tokens), value=block_ids_expanded)
        )

        # 6. Lock the node to prevent eviction during decoding
        match_after = self._radix_cache.match_prefix(prompt_tokens)
        radix_node = match_after.last_node
        self._radix_cache.inc_lock_ref(radix_node)

        # 7. Build state
        state = RequestState(
            request_id=request_id,
            prompt_tokens=list(prompt_tokens),
            output_tokens=[],
            all_block_ids=list(all_blocks),
            radix_node=radix_node,
            prefix_len=prefix_tokens,
            kv_caches=kv_caches,
        )
        self._active_requests[request_id] = state

        last_logits = logits[0, -1, :]  # (V,)
        return state, ForwardResult(
            logits=last_logits, forward_mode=ForwardMode.PREFILL
        )

    def forward_decode(self, state: RequestState, token_id: int) -> ForwardResult:
        """Process one generated token and extend the KV cache.

        Args:
            state:    Request state from prefill or previous decode.
            token_id: The token to process.

        Returns:
            ForwardResult with logits for sampling the next token.
        """
        total_tokens = state.seq_len  # tokens processed so far

        # 1. Allocate a new block if the current position needs one
        needed_blocks = math.ceil((total_tokens + 1) / self._block_size)
        if needed_blocks > len(state.all_block_ids):
            new_block = self._allocator.alloc(1)
            state.all_block_ids.extend(new_block)

        # 2. Update per-layer caches: all current blocks are prefix
        for cache in state.kv_caches:
            cache.set_blocks(
                prefix_blocks=state.all_block_ids,
                new_blocks=[],
                prefix_tokens=total_tokens,
            )

        # 3. Run model forward — single mx.eval sync point
        input_ids = mx.array([[token_id]])  # (1, 1)
        logits = self._model(input_ids, cache=state.kv_caches)  # (1, 1, V)
        mx.eval(logits)

        # 4. Update state
        state.output_tokens.append(token_id)

        # 5. Insert into radix cache
        all_tokens = state.prompt_tokens + state.output_tokens
        block_ids_expanded = [
            state.all_block_ids[i // self._block_size] for i in range(len(all_tokens))
        ]
        self._radix_cache.insert(InsertParams(key=all_tokens, value=block_ids_expanded))

        last_logits = logits[0, -1, :]  # (V,)
        return ForwardResult(logits=last_logits, forward_mode=ForwardMode.DECODE)

    def forward_decode_batch(
        self, states: list[RequestState], token_ids: list[int]
    ) -> list[ForwardResult]:
        """Process one decode step for N sequences in a single forward pass.

        All sequences are decoded simultaneously — weights are loaded once from
        RAM for the whole batch, yielding near-linear throughput scaling with
        batch size on memory-bandwidth-bound hardware.

        Args:
            states:    List of N active RequestState objects (all in decode phase).
            token_ids: List of N token IDs to process (one per sequence).

        Returns:
            List of N ForwardResult objects, one per sequence.
        """
        assert len(states) == len(token_ids), "states and token_ids must be same length"

        # 1. Allocate new blocks for any sequence that needs one
        for state in states:
            needed = math.ceil((state.seq_len + 1) / self._block_size)
            if needed > len(state.all_block_ids):
                state.all_block_ids.extend(self._allocator.alloc(1))

        # 2. Set all per-layer caches to their current block state
        for state in states:
            total = state.seq_len
            for cache in state.kv_caches:
                cache.set_blocks(
                    prefix_blocks=state.all_block_ids,
                    new_blocks=[],
                    prefix_tokens=total,
                )

        # 3. Build one BatchedPagedKVCache per layer (wraps all N sequences)
        batched_caches = [
            BatchedPagedKVCache([state.kv_caches[li] for state in states])
            for li in range(self._config.num_layers)
        ]

        # 4. Single forward pass for the whole batch — (N, 1) input
        input_ids = mx.array([[t] for t in token_ids])  # (N, 1)
        logits = self._model(input_ids, cache=batched_caches)  # (N, 1, V)
        mx.eval(logits)

        # 5. Update each state and insert into radix cache
        results: list[ForwardResult] = []
        for i, (state, token_id) in enumerate(zip(states, token_ids)):
            state.output_tokens.append(token_id)
            all_tokens = state.prompt_tokens + state.output_tokens
            block_ids_expanded = [
                state.all_block_ids[j // self._block_size]
                for j in range(len(all_tokens))
            ]
            self._radix_cache.insert(
                InsertParams(key=all_tokens, value=block_ids_expanded)
            )
            results.append(
                ForwardResult(logits=logits[i, -1, :], forward_mode=ForwardMode.DECODE)
            )

        return results

    def finish_request(self, state: RequestState) -> None:
        """Release radix cache lock and clean up request state."""
        self._radix_cache.dec_lock_ref(state.radix_node)
        self._active_requests.pop(state.request_id, None)

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 100,
        sampler: Callable[[mx.array], int] | None = None,
        stop_tokens: set[int] | None = None,
    ) -> list[int]:
        """Convenience: prefill then decode loop until stop or *max_tokens*."""
        if sampler is None:
            sampler = _argmax_sampler
        if stop_tokens is None:
            stop_tokens = set()

        request_id = uuid.uuid4().hex
        state, result = self.forward_prefill(request_id, prompt_tokens)

        generated: list[int] = []
        for _ in range(max_tokens):
            token = sampler(result.logits)
            generated.append(token)

            if token in stop_tokens:
                break

            result = self.forward_decode(state, token)

        self.finish_request(state)
        return generated


def _patch_for_paged_decode(model: Any) -> int:
    """Patch attention layers for paged decode.

    Uses store()+compute_paged_attn() for single-token decode steps.

    For each attention layer found in the model, swaps its class to a thin
    subclass that intercepts ``__call__`` when ``L == 1`` and the cache is a
    ``PagedKVCache``:

    * **Decode** (``L == 1``): scatter-writes new K/V via ``cache.store()``,
      then runs ``paged_attention_v1`` via ``cache.compute_paged_attn()``.
      No gather copy. O(seq_len) attention, O(1) write.

    * **Prefill** (``L > 1``) or non-``PagedKVCache`` cache: falls through to
      the original ``__call__``, which uses ``update_and_fetch`` as before.

    Returns the number of layers successfully patched.

    Notes
    -----
    Uses class-swapping (``attn.__class__ = PatchedSubclass``) because Python
    resolves special methods (``obj()``) through ``type(obj)``, not instance
    attributes, so per-instance ``__call__`` assignment has no effect.

    One patched subclass is created per unique original attention class, then
    reused for all layers of that type.
    """
    from sglang_mlx.srt.mem_cache.paged_pool import PagedKVCache

    # Locate the layer list — handles Llama-style (model.model.layers) and
    # flat-style (model.layers).
    layers: list | None = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if not layers:
        return 0

    patched_classes: dict[type, type] = {}  # original_cls → patched_cls
    n_patched = 0

    for layer in layers:
        # Resolve the attention sub-module (Llama: self_attn; others: attn / attention)
        attn: Any = None
        for attr in ("self_attn", "attn", "attention"):
            if hasattr(layer, attr):
                attn = getattr(layer, attr)
                break
        if attn is None:
            continue

        # Verify the module has the attributes the patched path needs
        _REQUIRED = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "rope",
            "n_heads",
            "n_kv_heads",
            "scale",
        )
        if not all(hasattr(attn, a) for a in _REQUIRED):
            continue

        orig_cls = type(attn)

        if orig_cls not in patched_classes:
            # Build a subclass of orig_cls that overrides __call__ for L==1 decode.
            # The class is created once and reused for all layers of the same type.
            class _PagedAttention(orig_cls):  # type: ignore[valid-type]
                def __call__(
                    self,
                    x: mx.array,
                    mask: Any = None,
                    cache: Any = None,
                ) -> mx.array:
                    # Intercept single-token decode for both single-sequence
                    # (PagedKVCache) and batched (BatchedPagedKVCache) paths.
                    # All other cases fall through to the original __call__.
                    is_single = isinstance(cache, PagedKVCache)
                    is_batched = isinstance(cache, BatchedPagedKVCache)
                    if not (is_single or is_batched) or x.shape[1] != 1:
                        return super().__call__(x, mask=mask, cache=cache)

                    B, L, _ = x.shape

                    # Project Q / K / V
                    queries = (
                        self.q_proj(x)
                        .reshape(B, L, self.n_heads, -1)
                        .transpose(0, 2, 1, 3)
                    )
                    keys = (
                        self.k_proj(x)
                        .reshape(B, L, self.n_kv_heads, -1)
                        .transpose(0, 2, 1, 3)
                    )
                    values = (
                        self.v_proj(x)
                        .reshape(B, L, self.n_kv_heads, -1)
                        .transpose(0, 2, 1, 3)
                    )

                    if is_single:
                        # Single-sequence path: one RoPE offset, one store, one attn
                        queries = self.rope(queries, offset=cache.offset)
                        keys = self.rope(keys, offset=cache.offset)
                        cache.store(keys, values, layer_idx=cache._layer_idx)
                        output = cache.compute_paged_attn(
                            queries, layer_idx=cache._layer_idx, scale=self.scale
                        )
                    else:
                        # Batched path: each sequence is at a different position,
                        # so RoPE must be applied per-sequence with its own offset.
                        offsets = cache.offsets
                        q_seqs, k_seqs = [], []
                        for i, off in enumerate(offsets):
                            q_seqs.append(self.rope(queries[i : i + 1], offset=off))
                            k_seqs.append(self.rope(keys[i : i + 1], offset=off))
                        queries = mx.concatenate(q_seqs, axis=0)
                        keys = mx.concatenate(k_seqs, axis=0)

                        layer_idx = cache._layer_idx
                        cache.store_each(keys, values, layer_idx)
                        output = cache.compute_paged_attn_batch(
                            queries, layer_idx=layer_idx, scale=self.scale
                        )

                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(output)

            _PagedAttention.__name__ = f"Paged{orig_cls.__name__}"
            _PagedAttention.__qualname__ = f"Paged{orig_cls.__qualname__}"
            patched_classes[orig_cls] = _PagedAttention

        attn.__class__ = patched_classes[orig_cls]
        n_patched += 1

    return n_patched


def _argmax_sampler(logits: mx.array) -> int:
    """Default greedy sampler."""
    return mx.argmax(logits).item()
