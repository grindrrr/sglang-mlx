// This file is ported from EricB/kernels-paged-attention-metal (https://huggingface.co/EricB/kernels-paged-attention-metal)
// Modified for MLX integration

#pragma once
#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <vector>
#include <utility>

namespace mx = mlx::core;

// ── Paged Attention ───────────────────────────────────────────────────────────

mx::array paged_attention_v1(
    const mx::array& query,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& block_tables,
    const mx::array& context_lens,
    int num_kv_heads,
    float scale,
    int block_size,
    int max_seq_len,
    const std::optional<mx::array>& alibi_slopes = std::nullopt
);

// ── Cache Operations ───────────────────────────────────────────────────────────

std::vector<mx::array> reshape_and_cache(
    const mx::array& key,
    const mx::array& value,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& slot_mapping
);

std::vector<mx::array> reshape_and_cache_flash(
    const mx::array& key,
    const mx::array& value,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& slot_mapping
);

std::pair<std::vector<mx::array>, std::vector<mx::array>> copy_blocks(
    const std::vector<mx::array>& key_caches,
    const std::vector<mx::array>& value_caches,
    const mx::array& block_mapping
);

mx::array swap_blocks(
    const mx::array& src,
    const mx::array& dst,
    const mx::array& block_mapping
);

// ── Device Attributes ─────────────────────────────────────────────────────────

int64_t get_device_attribute(int64_t attribute, int64_t device_id);
int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);
