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
│  │  passes with paged    │◄── │  Phi · DeepSeek · Cohere · ...    │  │
│  │  KV cache blocks      │    │                                   │  │
│  │                       │    │  Weight loading · Sampling        │  │
│  └──────────┬────────────┘    └───────────────────────────────────┘  │
│             │                                                        │
├─────────────▼────────────────────────────────────────────────────────┤
│                       Memory Management                              │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────┐  ┌───────────────────┐   │
│  │  Radix Prefix      │  │  KV Pool       │  │  Eviction         │   │
│  │  Cache             │  │                │  │  Policies         │   │
│  │                    │  │ Block Allocator│  │                   │   │
│  │  Tree-based prefix │  │  (free list)   │  │  LRU · LFU        │   │
│  │  sharing across    │  │                │  │  FIFO · Priority  │   │
│  │  requests          │  │  PagedMHAPool  │  │                   │   │
│  │                    │  │  (MLX arrays)  │  │                   │   │
│  └────────────────────┘  └────────────────┘  └───────────────────┘   │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                         MLX Runtime                                  │
│                                                                      │
│    Paged attention Metal kernels · mx.compile                        │
│    Lazy Evaluation · Unified Memory Arrays · mx.stream               │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                    Apple Silicon Hardware                            │
│                                                                      │
│           GPU Cores · Neural Engine · Unified Memory                 │
│                  Unified memory + block KV reuse                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Paged KV cache | Block allocator + Metal paged attention | Prefix reuse stays block-addressable and avoids gather copies. |
| No PyTorch | Pure MLX + Python | Zero interop overhead. MLX lazy eval handles everything. |
| Single process | Async event loop | Unified memory = no CPU↔GPU transfers = no need for separate GPU process. |
| mlx-lm models | Hybrid integration | 100+ architectures for free. Only replace KV cache interface. |
| PagedKVCache | Adapter pattern | Per-request view over shared PagedMHAPool blocks. |
