// This file is ported from EricB/kernels-paged-attention-metal (https://huggingface.co/EricB/kernels-paged-attention-metal)
// Modified for MLX integration

#pragma once
#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <vector>

namespace mx = mlx::core;

// ── Paged Attention ───────────────────────────────────────────────────────────

// v1: single-pass. Returns out [num_seqs, num_heads, head_size].
mx::array paged_attention_v1(
    const mx::array& query,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    int num_kv_heads,
    float scale,
    int block_size,
    int max_seq_len,
    const std::optional<mx::array>& alibi_slopes,
    const std::string& kv_cache_dtype,
    float k_scale,
    float v_scale
);

// v2: two-pass with partitioning. Returns [out, exp_sums, max_logits, tmp_out].
std::vector<mx::array> paged_attention_v2(
    const mx::array& query,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    int num_kv_heads,
    float scale,
    int block_size,
    int max_seq_len,
    int max_num_partitions,
    const std::optional<mx::array>& alibi_slopes,
    const std::string& kv_cache_dtype,
    float k_scale,
    float v_scale
);

// ── Cache Operations ──────────────────────────────────────────────────────────

// Returns [new_key_cache, new_value_cache] (in-place via shared buffer).
std::vector<mx::array> reshape_and_cache(
    const mx::array& key,
    const mx::array& value,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& slot_mapping,
    const std::string& kv_cache_dtype,
    float k_scale,
    float v_scale
);

// Flash-layout cache. Returns [new_key_cache, new_value_cache].
std::vector<mx::array> reshape_and_cache_flash(
    const mx::array& key,
    const mx::array& value,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& slot_mapping,
    const std::string& kv_cache_dtype,
    float k_scale,
    float v_scale
);

// Returns [new_key_caches, new_value_caches] (in-place via shared buffer).
std::pair<std::vector<mx::array>, std::vector<mx::array>> copy_blocks(
    const std::vector<mx::array>& key_caches,
    const std::vector<mx::array>& value_caches,
    const mx::array& block_mapping
);

// Returns new_dst (src blocks copied into dst via blit).
mx::array swap_blocks(
    const mx::array& src,
    const mx::array& dst,
    const mx::array& block_mapping
);

// ── FP8 Conversion ────────────────────────────────────────────────────────────

mx::array convert_fp8(
    const mx::array& src_cache,
    float scale,
    const std::string& kv_cache_dtype,
    const std::string& dst_dtype_str
);

// ── Device Attributes ─────────────────────────────────────────────────────────

int64_t get_device_attribute(int64_t attribute, int64_t device_id);
int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);
