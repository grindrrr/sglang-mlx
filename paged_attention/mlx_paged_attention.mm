// This file is ported from EricB/kernels-paged-attention-metal (https://huggingface.co/EricB/kernels-paged-attention-metal)
// Modified for MLX integration

// mlx_paged_attention.mm
// MLX primitives for paged_attention_v1 and paged_attention_v2.

#include "mlx_ops.h"
#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>

#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace mx = mlx::core;
using metal_device = mlx::core::metal::Device;

// ── Utilities ─────────────────────────────────────────────────────────────────

static std::string getModuleDirectory() {
    Dl_info info;
    if (dladdr((void *)getModuleDirectory, &info)) {
        std::string p(info.dli_fname);
        auto pos = p.rfind('/');
        if (pos != std::string::npos) return p.substr(0, pos);
    }
    return ".";
}

static std::string dtypeStr(mx::Dtype d) {
    switch (d) {
        case mx::float32:  return "float";
        case mx::float16:  return "half";
        case mx::bfloat16: return "bfloat16_t";
        case mx::uint8:    return "uchar";
        default: throw std::invalid_argument("Unsupported dtype");
    }
}

static std::string attnKernelName(
    mx::Dtype dtype, mx::Dtype cache_dtype,
    int head_size, int block_size,
    int num_threads, int num_simd_lanes, int partition_size)
{
    return "paged_attention_"
        + dtypeStr(dtype) + "_cache_" + dtypeStr(cache_dtype)
        + "_hs" + std::to_string(head_size)
        + "_bs" + std::to_string(block_size)
        + "_nt" + std::to_string(num_threads)
        + "_nsl" + std::to_string(num_simd_lanes)
        + "_ps" + std::to_string(partition_size);
}

static bool isValidConfig(int head_size, int block_size) {
    static const int hs[] = {32, 64, 80, 96, 112, 120, 128, 192, 256};
    static const int bs[] = {8, 16, 32};
    bool ok_hs = false, ok_bs = false;
    for (int v : hs) ok_hs |= (v == head_size);
    for (int v : bs) ok_bs |= (v == block_size);
    return ok_hs && ok_bs;
}

// ── PagedAttentionV1 ──────────────────────────────────────────────────────────

struct PagedAttentionV1 : mx::Primitive {
    int   num_kv_heads_;
    float scale_;
    int   block_size_;
    int   max_seq_len_;
    bool  use_fp8_;
    float k_scale_, v_scale_;
    bool  use_alibi_;
    std::string kernel_name_;
    size_t smem_;

    PagedAttentionV1(mx::Stream s,
                     int num_kv_heads, float scale, int block_size, int max_seq_len,
                     bool use_fp8, float k_scale, float v_scale, bool use_alibi,
                     std::string kname, size_t smem)
        : mx::Primitive(s),
          num_kv_heads_(num_kv_heads), scale_(scale),
          block_size_(block_size), max_seq_len_(max_seq_len),
          use_fp8_(use_fp8), k_scale_(k_scale), v_scale_(v_scale),
          use_alibi_(use_alibi), kernel_name_(std::move(kname)), smem_(smem) {}

    const char* name() const override { return "PagedAttentionV1"; }

    bool is_equivalent(const mx::Primitive& o) const override {
        const auto& p = static_cast<const PagedAttentionV1&>(o);
        return num_kv_heads_ == p.num_kv_heads_ && scale_ == p.scale_
            && block_size_ == p.block_size_ && max_seq_len_ == p.max_seq_len_
            && use_fp8_ == p.use_fp8_ && use_alibi_ == p.use_alibi_
            && kernel_name_ == p.kernel_name_;
    }

    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("PagedAttentionV1 is GPU-only");
    }

    // inputs: [query(0), key_cache(1), value_cache(2), block_tables(3),
    //          seq_lens(4), alibi_slopes(5, optional)]
    // outputs: [out(0)]
    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& s = stream();
        auto& d = mlx::core::metal::device(s.device);

        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));

        int64_t num_seqs  = inputs[0].shape(0);
        int64_t num_heads = inputs[0].shape(1);
        int32_t max_blocks = static_cast<int32_t>(inputs[3].shape(1));

        std::string lib_path = getModuleDirectory() + "/" METALLIB_PATH;
        auto* lib = d.get_library("paged_attention_mlx", lib_path);

        bool use_part = false;
        mlx::core::metal::MTLFCList fc = {
            {&use_part,   MTL::DataTypeBool, NS::UInteger(10)},
            {&use_alibi_, MTL::DataTypeBool, NS::UInteger(20)},
            {&use_fp8_,   MTL::DataTypeBool, NS::UInteger(30)},
        };
        std::string hash = kernel_name_
            + "_p0_a" + (use_alibi_ ? "1" : "0")
            + "_f"    + (use_fp8_   ? "1" : "0");
        auto* kernel = d.get_kernel(kernel_name_, lib, hash, fc);

        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(kernel);
        enc.set_threadgroup_memory_length(smem_, 0);

        // Buffer indices match the Metal kernel signature.
        // Buffers 0 (exp_sums) and 1 (max_logits) are skipped for v1.
        enc.set_output_array(out,        2);
        enc.set_input_array(inputs[0],   3);   // query
        enc.set_input_array(inputs[1],   4);   // key_cache
        enc.set_input_array(inputs[2],   5);   // value_cache
        if (use_fp8_) {
            enc.set_bytes(k_scale_, 6);
            enc.set_bytes(v_scale_, 7);
        }
        // 6,7 unused when use_fp8_scales == false (kernel won't access them)
        enc.set_bytes(static_cast<int32_t>(num_kv_heads_), 8);
        enc.set_bytes(scale_, 9);
        enc.set_bytes(1.0f,   10);             // softcapping (no-op)
        enc.set_input_array(inputs[3],  11);   // block_tables
        enc.set_input_array(inputs[4],  12);   // seq_lens
        enc.set_bytes(max_blocks,       13);
        if (use_alibi_) {
            enc.set_input_array(inputs[5], 14);
        }
        enc.set_bytes(static_cast<int32_t>(inputs[0].strides(0)), 15); // q_stride
        enc.set_bytes(static_cast<int32_t>(inputs[1].strides(0)), 16); // kv_block_stride
        enc.set_bytes(static_cast<int32_t>(inputs[1].strides(1)), 17); // kv_head_stride

        enc.dispatch_threadgroups(
            MTL::Size::Make(num_heads, num_seqs, 1),
            MTL::Size::Make(256, 1, 1));
    }
};

// ── PagedAttentionV2 ──────────────────────────────────────────────────────────

struct PagedAttentionV2 : mx::Primitive {
    int   num_kv_heads_;
    float scale_;
    int   block_size_;
    int   max_num_partitions_;
    bool  use_fp8_;
    float k_scale_, v_scale_;
    bool  use_alibi_;
    std::string kernel_name_, reduce_kernel_name_;
    size_t smem_;

    PagedAttentionV2(mx::Stream s,
                     int num_kv_heads, float scale, int block_size,
                     int max_num_partitions,
                     bool use_fp8, float k_scale, float v_scale, bool use_alibi,
                     std::string kname, std::string reduce_kname, size_t smem)
        : mx::Primitive(s),
          num_kv_heads_(num_kv_heads), scale_(scale),
          block_size_(block_size), max_num_partitions_(max_num_partitions),
          use_fp8_(use_fp8), k_scale_(k_scale), v_scale_(v_scale),
          use_alibi_(use_alibi), kernel_name_(std::move(kname)),
          reduce_kernel_name_(std::move(reduce_kname)), smem_(smem) {}

    const char* name() const override { return "PagedAttentionV2"; }

    bool is_equivalent(const mx::Primitive& o) const override {
        const auto& p = static_cast<const PagedAttentionV2&>(o);
        return num_kv_heads_ == p.num_kv_heads_ && scale_ == p.scale_
            && block_size_ == p.block_size_
            && max_num_partitions_ == p.max_num_partitions_
            && use_fp8_ == p.use_fp8_ && use_alibi_ == p.use_alibi_
            && kernel_name_ == p.kernel_name_;
    }

    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("PagedAttentionV2 is GPU-only");
    }

    // inputs: [query(0), key_cache(1), value_cache(2), block_tables(3),
    //          seq_lens(4), alibi_slopes(5, optional)]
    // outputs: [out(0), exp_sums(1), max_logits(2), tmp_out(3)]
    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& s = stream();
        auto& d = mlx::core::metal::device(s.device);

        for (auto& o : outputs)
            o.set_data(mx::allocator::malloc(o.nbytes()));

        int64_t num_seqs  = inputs[0].shape(0);
        int64_t num_heads = inputs[0].shape(1);
        int32_t max_blocks = static_cast<int32_t>(inputs[3].shape(1));

        std::string lib_path = getModuleDirectory() + "/" METALLIB_PATH;
        auto* lib = d.get_library("paged_attention_mlx", lib_path);

        // ── Phase 1: main attention with partitioning ─────────────────────
        bool use_part = true;
        mlx::core::metal::MTLFCList fc = {
            {&use_part,   MTL::DataTypeBool, NS::UInteger(10)},
            {&use_alibi_, MTL::DataTypeBool, NS::UInteger(20)},
            {&use_fp8_,   MTL::DataTypeBool, NS::UInteger(30)},
        };
        std::string hash = kernel_name_
            + "_p1_a" + (use_alibi_ ? "1" : "0")
            + "_f"    + (use_fp8_   ? "1" : "0");
        auto* kernel = d.get_kernel(kernel_name_, lib, hash, fc);

        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(kernel);
        enc.set_threadgroup_memory_length(smem_, 0);

        enc.set_output_array(outputs[1],  0);  // exp_sums
        enc.set_output_array(outputs[2],  1);  // max_logits
        enc.set_output_array(outputs[3],  2);  // tmp_out
        enc.set_input_array(inputs[0],    3);  // query
        enc.set_input_array(inputs[1],    4);  // key_cache
        enc.set_input_array(inputs[2],    5);  // value_cache
        if (use_fp8_) {
            enc.set_bytes(k_scale_, 6);
            enc.set_bytes(v_scale_, 7);
        }
        enc.set_bytes(static_cast<int32_t>(num_kv_heads_), 8);
        enc.set_bytes(scale_, 9);
        enc.set_bytes(1.0f,   10);
        enc.set_input_array(inputs[3], 11);    // block_tables
        enc.set_input_array(inputs[4], 12);    // seq_lens
        enc.set_bytes(max_blocks,      13);
        if (use_alibi_) {
            enc.set_input_array(inputs[5], 14);
        }
        enc.set_bytes(static_cast<int32_t>(inputs[0].strides(0)), 15);
        enc.set_bytes(static_cast<int32_t>(inputs[1].strides(0)), 16);
        enc.set_bytes(static_cast<int32_t>(inputs[1].strides(1)), 17);

        enc.dispatch_threadgroups(
            MTL::Size::Make(num_heads, num_seqs, max_num_partitions_),
            MTL::Size::Make(256, 1, 1));

        // ── Phase 2: reduction kernel ────────────────────────────────────
        // set_input_array on buffers written in phase 1 triggers auto-barrier.
        auto* reduce_kernel = d.get_kernel(reduce_kernel_name_, lib);

        size_t reduce_smem = static_cast<size_t>(max_num_partitions_) * sizeof(float) * 2;
        enc.set_compute_pipeline_state(reduce_kernel);
        enc.set_threadgroup_memory_length(reduce_smem, 0);

        enc.set_output_array(outputs[0], 0);   // out (final)
        enc.set_input_array(outputs[1],  1);   // exp_sums
        enc.set_input_array(outputs[2],  2);   // max_logits
        enc.set_input_array(outputs[3],  3);   // tmp_out
        enc.set_input_array(inputs[4],   4);   // seq_lens
        enc.set_bytes(static_cast<int32_t>(max_num_partitions_), 5);

        enc.dispatch_threadgroups(
            MTL::Size::Make(num_heads, num_seqs, 1),
            MTL::Size::Make(256, 1, 1));
    }
};

// ── Public API ────────────────────────────────────────────────────────────────

mx::array paged_attention_v1(
    const mx::array& query,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    int num_kv_heads, float scale, int block_size, int max_seq_len,
    const std::optional<mx::array>& alibi_slopes,
    const std::string& kv_cache_dtype,
    float k_scale, float v_scale)
{
    auto s = mx::default_stream(mx::Device::gpu);
    int head_size = query.shape(2);
    if (!isValidConfig(head_size, block_size))
        throw std::invalid_argument("Unsupported head_size/block_size: "
            + std::to_string(head_size) + "/" + std::to_string(block_size));

    bool use_fp8  = (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3");
    bool use_alibi= alibi_slopes.has_value();
    const int nt = 256, nsl = 32, ps = 0;
    int padded = ((max_seq_len + block_size - 1) / block_size) * block_size;
    size_t smem = std::max((size_t)(padded * sizeof(float)),
                           (size_t)(((nt/nsl)/2) * head_size * sizeof(float)));

    std::string kname = attnKernelName(query.dtype(), key_cache.dtype(),
                                       head_size, block_size, nt, nsl, ps);

    std::vector<mx::array> ins = {query, key_cache, value_cache, block_tables, seq_lens};
    if (use_alibi) ins.push_back(*alibi_slopes);

    return mx::array(
        {query.shape(0), query.shape(1), (int)head_size},
        query.dtype(),
        std::make_shared<PagedAttentionV1>(
            s, num_kv_heads, scale, block_size, max_seq_len,
            use_fp8, k_scale, v_scale, use_alibi, kname, smem),
        ins);
}

std::vector<mx::array> paged_attention_v2(
    const mx::array& query,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    int num_kv_heads, float scale, int block_size, int max_seq_len,
    int max_num_partitions,
    const std::optional<mx::array>& alibi_slopes,
    const std::string& kv_cache_dtype,
    float k_scale, float v_scale)
{
    auto s = mx::default_stream(mx::Device::gpu);
    int head_size = query.shape(2);
    if (!isValidConfig(head_size, block_size))
        throw std::invalid_argument("Unsupported head_size/block_size");

    bool use_fp8  = (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3");
    bool use_alibi= alibi_slopes.has_value();
    const int nt = 256, nsl = 32, ps = 512;
    size_t smem = std::max((size_t)(ps * sizeof(float)),
                           (size_t)(((nt/nsl)/2) * head_size * sizeof(float)));

    std::string kname = attnKernelName(query.dtype(), key_cache.dtype(),
                                       head_size, block_size, nt, nsl, ps);
    std::string reduce_kname = "paged_attention_v2_reduce_"
        + dtypeStr(query.dtype())
        + "_hs"  + std::to_string(head_size)
        + "_nt"  + std::to_string(nt)
        + "_nsl" + std::to_string(nsl)
        + "_ps"  + std::to_string(ps);

    std::vector<mx::array> ins = {query, key_cache, value_cache, block_tables, seq_lens};
    if (use_alibi) ins.push_back(*alibi_slopes);

    int64_t ns = query.shape(0), nh = query.shape(1);

    return mx::array::make_arrays(
        {
            {(int)ns, (int)nh, head_size},
            {(int)ns, (int)nh, max_num_partitions},
            {(int)ns, (int)nh, max_num_partitions},
            {(int)ns, (int)nh, max_num_partitions, head_size},
        },
        {query.dtype(), mx::float32, mx::float32, query.dtype()},
        std::make_shared<PagedAttentionV2>(
            s, num_kv_heads, scale, block_size, max_num_partitions,
            use_fp8, k_scale, v_scale, use_alibi, kname, reduce_kname, smem),
        ins);
}
