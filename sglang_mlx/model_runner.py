"""Model runner for MLX-based inference.

Bridges mlx-lm models with the radix cache and KV pool, enabling
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
import uuid
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from sglang_mlx.srt.mem_cache.base_prefix_cache import InsertParams
from sglang_mlx.srt.mem_cache.paged_pool import (
    BlockAllocator,
    PagedKVCache,
    PagedMHAPool,
    block_table_from_token_blocks,
    expand_blocks_to_tokens,
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
    all_block_indices: list[int]  # token-granular physical block IDs
    radix_node: Any  # for lock/unlock
    prefix_len: int  # tokens reused from radix cache
    kv_caches: list[PagedKVCache]
    committed_len: int = 0

    @property
    def seq_len(self) -> int:
        """Total sequence length (prompt + generated)."""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def total_cached_len(self) -> int:
        """Number of token positions currently backed by KV blocks."""
        return len(self.all_block_indices)


@dataclasses.dataclass
class ForwardResult:
    """Result of a single forward pass."""

    logits: mx.array  # (vocab_size,) for last position
    forward_mode: ForwardMode


def _extract_model_config(model: Any, config_dict: dict | None = None) -> ModelConfig:
    """Extract model configuration from an mlx-lm model.

    Reads from ``model.args`` (mlx-lm convention) first, falls back to
    *config_dict* for missing fields.

    Args:
        model: An mlx-lm model with ``.args`` and ``.layers``.
        config_dict: Optional config.json contents as fallback.

    Returns:
        Populated ModelConfig.
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

    Owns the KV pool, block allocator, and radix cache.  Provides methods for
    prefill (prompt processing with prefix reuse), decode (one token at a
    time), and a convenience ``generate`` loop.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ModelConfig,
        pool_size: int = 4096,
        block_size: int = 16,
        dtype: mx.Dtype = mx.float16,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._pool_size = pool_size
        self._block_size = block_size
        self._dtype = dtype

        self._allocator = BlockAllocator(pool_size, block_size)
        self._pool = PagedMHAPool(
            num_blocks=pool_size,
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
        pool_size: int = 4096,
        block_size: int = 16,
        dtype: mx.Dtype = mx.float16,
    ) -> ModelRunner:
        """Load a model from a local path or HuggingFace Hub ID.

        Args:
            model_path: Local directory or HF model ID
                (e.g. ``mlx-community/...``).
            pool_size: Number of physical blocks in the KV pool.
            dtype: Data type for KV cache buffers.

        Returns:
            Configured ModelRunner ready for inference.
        """
        from mlx_lm import load as mlx_load

        model, tokenizer = mlx_load(model_path)
        config = _extract_model_config(model)
        return cls(model, tokenizer, config, pool_size, block_size, dtype)

    # ---- Core forward methods ----

    def forward_prefill(
        self, request_id: str, prompt_tokens: list[int]
    ) -> tuple[RequestState, ForwardResult]:
        """Process prompt tokens, reusing any cached prefix.

        Args:
            request_id: Unique identifier for this request.
            prompt_tokens: Full prompt token IDs.

        Returns:
            ``(RequestState, ForwardResult)`` — the state to pass to
            ``forward_decode``, and logits for sampling the first output
            token.
        """
        # 1. Match prefix in radix cache
        match = self._radix_cache.match_prefix(prompt_tokens)
        matched_token_blocks = list(match.device_indices)
        prefix_len = self._usable_prefix_len(len(matched_token_blocks))

        # Full block-aligned cache hits still need logits. Recompute the final
        # block and keep that transient block out of the radix cache.
        recompute_full_hit = prefix_len >= len(prompt_tokens) and prefix_len > 0
        if recompute_full_hit:
            prefix_len = max(0, prefix_len - self._block_size)

        prefix_token_blocks = matched_token_blocks[:prefix_len]
        prefix_blocks = block_table_from_token_blocks(
            prefix_token_blocks, self._block_size
        )
        new_tokens = prompt_tokens[prefix_len:]
        new_blocks = self._allocator.alloc_for_tokens(len(new_tokens))

        # 3. Create per-layer PagedKVCache
        kv_caches: list[PagedKVCache] = []
        for i in range(self._config.num_layers):
            cache = PagedKVCache(self._pool, layer_idx=i)
            cache.set_blocks(prefix_blocks, new_blocks, prefix_len)
            kv_caches.append(cache)

        # 4. Run model forward — single mx.eval sync point
        input_ids = mx.array([new_tokens])  # (1, N)
        logits = self._model(input_ids, cache=kv_caches)  # (1, N, V)
        mx.eval(logits)

        # 5. Insert into radix cache
        all_block_indices = prefix_token_blocks + expand_blocks_to_tokens(
            new_blocks, len(new_tokens), self._block_size
        )
        if recompute_full_hit:
            committed_len = prefix_len
        else:
            committed_len = self._commit_full_blocks(prompt_tokens, all_block_indices)

        # 6. Lock the node to prevent eviction during decoding
        match_after = self._radix_cache.match_prefix(prompt_tokens[:committed_len])
        radix_node = match_after.last_node
        if committed_len > 0:
            self._radix_cache.inc_lock_ref(radix_node)

        # 7. Build state
        state = RequestState(
            request_id=request_id,
            prompt_tokens=list(prompt_tokens),
            output_tokens=[],
            all_block_indices=list(all_block_indices),
            radix_node=radix_node,
            prefix_len=prefix_len,
            kv_caches=kv_caches,
            committed_len=committed_len,
        )
        self._active_requests[request_id] = state

        last_logits = logits[0, -1, :]  # (V,)
        return state, ForwardResult(
            logits=last_logits, forward_mode=ForwardMode.PREFILL
        )

    def forward_decode(self, state: RequestState, token_id: int) -> ForwardResult:
        """Process one generated token and extend the KV cache.

        Args:
            state: Request state from prefill or previous decode.
            token_id: The token to process.

        Returns:
            ForwardResult with logits for sampling the next token.
        """
        # 1. Allocate a new block only when the next token starts a new page.
        old_len = state.seq_len
        new_blocks = self._allocator.alloc(1) if old_len % self._block_size == 0 else []
        prefix_blocks = block_table_from_token_blocks(
            state.all_block_indices, self._block_size
        )

        # 2. Update per-layer caches: all existing tokens become prefix
        for cache in state.kv_caches:
            cache.set_blocks(prefix_blocks, new_blocks, old_len)

        # 3. Run model forward — single mx.eval sync point
        input_ids = mx.array([[token_id]])  # (1, 1)
        logits = self._model(input_ids, cache=state.kv_caches)  # (1, 1, V)
        mx.eval(logits)

        # 4. Update state
        state.output_tokens.append(token_id)
        all_blocks = prefix_blocks + new_blocks
        state.all_block_indices.append(all_blocks[old_len // self._block_size])

        # 5. Commit newly completed full blocks to the radix cache.
        all_tokens = state.prompt_tokens + state.output_tokens
        old_committed_len = state.committed_len
        state.committed_len = self._commit_full_blocks(
            all_tokens, state.all_block_indices
        )
        if state.committed_len > old_committed_len:
            if old_committed_len > 0:
                self._radix_cache.dec_lock_ref(state.radix_node)
            match_after = self._radix_cache.match_prefix(
                all_tokens[: state.committed_len]
            )
            state.radix_node = match_after.last_node
            self._radix_cache.inc_lock_ref(state.radix_node)

        last_logits = logits[0, -1, :]  # (V,)
        return ForwardResult(logits=last_logits, forward_mode=ForwardMode.DECODE)

    def finish_request(self, state: RequestState) -> None:
        """Release radix cache lock and clean up request state.

        Args:
            state: The completed request's state.
        """
        if state.committed_len > 0:
            self._radix_cache.dec_lock_ref(state.radix_node)
        tail_blocks = state.all_block_indices[state.committed_len :]
        if tail_blocks:
            self._allocator.free(tail_blocks)
        self._active_requests.pop(state.request_id, None)

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 100,
        sampler: Callable[[mx.array], int] | None = None,
        stop_tokens: set[int] | None = None,
    ) -> list[int]:
        """Convenience: prefill then decode loop until stop or *max_tokens*.

        Args:
            prompt_tokens: Prompt token IDs.
            max_tokens: Maximum number of tokens to generate.
            sampler: Maps logits to a token ID.  Defaults to argmax.
            stop_tokens: Token IDs that terminate generation.

        Returns:
            List of generated token IDs (excluding the prompt).
        """
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

    def _usable_prefix_len(self, matched_len: int) -> int:
        """Only full blocks are shareable through the radix cache."""
        return (matched_len // self._block_size) * self._block_size

    def _commit_full_blocks(
        self,
        tokens: list[int],
        token_blocks: list[int],
    ) -> int:
        """Insert the full-block prefix into radix and return committed length."""
        committed_len = self._usable_prefix_len(len(token_blocks))
        if committed_len == 0:
            return 0
        self._radix_cache.insert(
            InsertParams(
                key=list(tokens[:committed_len]),
                value=list(token_blocks[:committed_len]),
            )
        )
        return committed_len


def _argmax_sampler(logits: mx.array) -> int:
    """Default greedy sampler."""
    return mx.argmax(logits).item()
