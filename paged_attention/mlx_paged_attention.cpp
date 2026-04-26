// This file is ported from EricB/kernels-paged-attention-metal (https://huggingface.co/EricB/kernels-paged-attention-metal)
// Modified for MLX integration

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
        default: throw std::invalid_argument("Unsupported dtype");
    }
}

static std::string attnKernelName(
    mx::Dtype dtype, mx::Dtype cache_dtype,
    int head_size, int block_size,
    int num_threads, int num_simd_lanes)
{
    return "paged_attention_"
        + dtypeStr(dtype) + "_cache_" + dtypeStr(cache_dtype)
        + "_hs" + std::to_string(head_size)
        + "_bs" + std::to_string(block_size)
        + "_nt" + std::to_string(num_threads)
        + "_nsl" + std::to_string(num_simd_lanes);
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
    bool  use_alibi_;
    std::string kernel_name_;
    size_t smem_;

    PagedAttentionV1(mx::Stream s,
                     int num_kv_heads, float scale, int block_size, int max_seq_len,
                     bool use_alibi, std::string kname, size_t smem)
        : mx::Primitive(s),
          num_kv_heads_(num_kv_heads), scale_(scale),
          block_size_(block_size), max_seq_len_(max_seq_len),
          use_alibi_(use_alibi), kernel_name_(std::move(kname)), smem_(smem) {}

    const char* name() const override { return "PagedAttentionV1"; }

    bool is_equivalent(const mx::Primitive& o) const override {
        const auto& p = static_cast<const PagedAttentionV1&>(o);
        return num_kv_heads_ == p.num_kv_heads_ && scale_ == p.scale_
            && block_size_ == p.block_size_ && max_seq_len_ == p.max_seq_len_
            && use_alibi_ == p.use_alibi_ && kernel_name_ == p.kernel_name_;
    }

    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("PagedAttentionV1 is GPU-only");
    }

    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& s = stream();
        auto& d = mx::metal::device(s.device);

        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));

        int num_seqs = inputs[0].shape(0);
        int num_heads = inputs[0].shape(1);
        int head_size = inputs[0].shape(2);
        int q_stride  = static_cast<int>(inputs[0].strides(0));
        int kv_block_stride = static_cast<int>(inputs[1].strides(0));
        int kv_head_stride  = static_cast<int>(inputs[1].strides(1));

        std::string lib_path = getModuleDirectory() + "/" METALLIB_PATH;
        auto* lib = d.get_library("paged_attention_mlx", lib_path);

        bool use_partitioning = false;
        mx::metal::MTLFCList fc = {
            {&use_partitioning, MTL::DataTypeBool, NS::UInteger(10)},
            {&use_alibi_,       MTL::DataTypeBool, NS::UInteger(20)},
        };
        auto* kernel = d.get_kernel(kernel_name_, lib, kernel_name_, fc);

        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(kernel);

        // Dummy buffers for V1 compatibility with kernel signature if needed,
        // but our simplified kernel doesn't use buffers 0 and 1 for V1.
        mx::array exp_sums_dummy({1}, mx::float32);
        mx::array max_logits_dummy({1}, mx::float32);
        exp_sums_dummy.set_data(mx::allocator::malloc(exp_sums_dummy.nbytes()));
        max_logits_dummy.set_data(mx::allocator::malloc(max_logits_dummy.nbytes()));

        enc.set_output_array(exp_sums_dummy, 0);
        enc.set_output_array(max_logits_dummy, 1);
        enc.set_output_array(outputs[0], 2);
        enc.set_input_array(inputs[0],  3);
        enc.set_input_array(inputs[1],  4);
        enc.set_input_array(inputs[2],  5);

        enc.set_bytes(static_cast<int32_t>(num_kv_heads_), 6);
        enc.set_bytes(scale_, 7);
        enc.set_bytes(1.0f,   8);
        enc.set_input_array(inputs[3], 9);
        enc.set_input_array(inputs[4], 10);
        enc.set_bytes(static_cast<int32_t>(inputs[3].shape(1)), 11);

        if (use_alibi_) {
            enc.set_input_array(inputs[5], 12);
        }
        enc.set_bytes(q_stride, 13);
        enc.set_bytes(kv_block_stride, 14);
        enc.set_bytes(kv_head_stride,  15);
        enc.set_bytes(static_cast<int32_t>(max_seq_len_), 16);

        enc.set_threadgroup_memory_length(smem_, 0);

        enc.dispatch_threadgroups(
            MTL::Size::Make(num_heads, num_seqs, 1),
            MTL::Size::Make(256, 1, 1));
        }

};

// ── Public API ────────────────────────────────────────────────────────────────

mx::array paged_attention_v1(
    const mx::array& query, const mx::array& key_cache, const mx::array& value_cache,
    const mx::array& block_tables, const mx::array& context_lens,
    int num_kv_heads, float scale, int block_size, int max_seq_len,
    const std::optional<mx::array>& alibi_slopes)
{
    int head_size = query.shape(2);
    if (!isValidConfig(head_size, block_size))
        throw std::invalid_argument("Unsupported head_size/block_size for paged_attention");

    bool use_alibi = alibi_slopes.has_value();
    const int nt = 256, nsl = 32;
    const size_t smem =
        static_cast<size_t>(max_seq_len + 128 + (nt / nsl) * head_size)
        * sizeof(float);

    std::string kname = attnKernelName(query.dtype(), key_cache.dtype(),
                                       head_size, block_size, nt, nsl);

    auto prim = std::make_shared<PagedAttentionV1>(
        mx::default_stream(mx::Device::gpu),
        num_kv_heads, scale, block_size, max_seq_len,
        use_alibi, kname, smem);

    std::vector<mx::array> ins = {query, key_cache, value_cache, block_tables, context_lens};
    if (use_alibi) ins.push_back(*alibi_slopes);

    return mx::array(query.shape(), query.dtype(), prim, ins);
}
