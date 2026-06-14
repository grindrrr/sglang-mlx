# sglang-mlx

SGLang-style LLM serving for Apple Silicon, built on **pure MLX — no PyTorch**.

## Why

Bring SGLang's high-throughput serving to Apple Silicon. Unified memory changes
the paging tradeoffs, while block-addressable KV caches and radix-tree prefix
sharing still provide important serving wins.

## Paged Attention Kernel Scope

The Metal extension currently provides a single-pass, non-partitioned decode
kernel exposed as `paged_attention_v1`. Each sequence/head pair is handled by
one threadgroup. This is intended for initial correctness and batched decode
validation; it is not a direct port of vLLM's CUDA-specific V2 path.

Long-context, batch-1 behavior should be benchmarked before introducing a
Metal-native partitioned implementation. The current path validates device
threadgroup-memory requirements before dispatch.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       OpenAI-Compatible API                          │
│                                                                      │
│    /v1/chat/completions    /v1/completions    /v1/models             │
│    Async HTTP (FastAPI) · SSE Streaming · Request Validation         │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                            Engine                                    │
│                                                                      │
│  ┌──────────────────┐  ┌───────────────┐  ┌──────────────────────┐   │
│  │    Scheduler     │  │   Tokenizer   │  │   Detokenizer        │   │
│  │                  │  │   (mlx-lm)    │  │   (incremental)      │   │
│  │  Request Queue   │  │               │  │                      │   │
│  │  Continuous      │  │  Encode text  │  │  Token IDs → text    │   │
│  │  Batching        │  │  → token IDs  │  │  SSE streaming       │   │
│  │  (prefill/decode)│  │               │  │                      │   │
│  └────────┬─────────┘  └───────────────┘  └──────────────────────┘   │
│           │                                                          │
├───────────▼──────────────────────────────────────────────────────────┤
│                        Model Execution                               │
│                                                                      │
│  ┌───────────────────────┐    ┌───────────────────────────────────┐  │
│  │    ModelRunner        │    │    mlx-lm Model Zoo               │  │
│  │                       │    │                                   │  │
│  │  Manages forward      │    │  Llama · Qwen · Mistral · Gemma   │  │
│  │  passes with batched  │◄── │  Phi · DeepSeek · Cohere · ...    │  │
│  │  PooledKVCache        │    │                                   │  │
│  │                       │    │  Weight loading · Sampling        │  │
│  └──────────┬────────────┘    └───────────────────────────────────┘  │
│             │                                                        │
├─────────────▼────────────────────────────────────────────────────────┤
│                       Memory Management                              │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────┐  ┌───────────────────┐   │
│  │  Radix Prefix      │  │  KV Pool       │  │  Eviction         │   │
│  │  Cache             │  │                │  │  Policies         │   │
│  │                    │  │ Slot Allocator │  │                   │   │
│  │  Tree-based prefix │  │  (free list)   │  │  LRU · LFU        │   │
│  │  sharing across    │  │                │  │  FIFO · Priority  │   │
│  │  requests          │  │  MHATokenPool  │  │                   │   │
│  │                    │  │  (MLX arrays)  │  │                   │   │
│  └────────────────────┘  └────────────────┘  └───────────────────┘   │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                         MLX Runtime                                  │
│                                                                      │
│    mx.fast.scaled_dot_product_attention · mx.compile                 │
│    Lazy Evaluation · Unified Memory Arrays · mx.stream               │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                    Apple Silicon Hardware                            │
│                                                                      │
│           GPU Cores · Neural Engine · Unified Memory                 │
│                  No paged attention needed                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| No paged attention | Simple slot allocator | Unified memory = OS handles paging. No fragmentation problem. |
| No PyTorch | Pure MLX + Python | Zero interop overhead. MLX lazy eval handles everything. |
| Single process | Async event loop | Unified memory = no CPU↔GPU transfers = no need for separate GPU process. |
| mlx-lm models | Hybrid integration | 100+ architectures for free. Only replace KV cache interface. |
| PooledKVCache | Adapter pattern | Thin wrapper over MHATokenPool. Models unchanged. |
