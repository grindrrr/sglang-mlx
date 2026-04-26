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
        default: throw std::invalid_argument("Unsupported cache dtype");
    }
}

// ── ReshapeAndCache ───────────────────────────────────────────────────────────

struct ReshapeAndCache : mx::Primitive {
    std::string kernel_name_;
    int32_t num_heads_, head_size_, block_size_, x_;

    ReshapeAndCache(mx::Stream s, std::string kname,
                    int32_t num_heads, int32_t head_size,
                    int32_t block_size, int32_t x)
        : mx::Primitive(s), kernel_name_(std::move(kname)),
          num_heads_(num_heads), head_size_(head_size),
          block_size_(block_size), x_(x) {}

    const char* name() const override { return "ReshapeAndCache"; }
    bool is_equivalent(const mx::Primitive& o) const override {
        const auto& p = static_cast<const ReshapeAndCache&>(o);
        return kernel_name_ == p.kernel_name_;
    }
    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("ReshapeAndCache is GPU-only");
    }

    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& s = stream();
        auto& d = mx::metal::device(s.device);

        outputs[0].copy_shared_buffer(inputs[2]);
        outputs[1].copy_shared_buffer(inputs[3]);

        int64_t num_tokens = inputs[0].shape(0);
        int32_t key_stride   = static_cast<int32_t>(inputs[0].strides(0));
        int32_t value_stride = static_cast<int32_t>(inputs[1].strides(0));

        std::string lib_path = getModuleDirectory() + "/" METALLIB_PATH;
        auto* lib = d.get_library("paged_attention_mlx", lib_path);
        auto* kernel = d.get_kernel(kernel_name_, lib);

        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(kernel);

        enc.set_input_array(inputs[0],  0);    // key
        enc.set_input_array(inputs[1],  1);    // value
        enc.set_output_array(outputs[0], 2);   // key_cache
        enc.set_output_array(outputs[1], 3);   // value_cache
        enc.set_input_array(inputs[4],  4);    // slot_mapping (MUST be int32)
        
        enc.set_bytes(key_stride,   7);
        enc.set_bytes(value_stride, 8);
        enc.set_bytes(num_heads_,   9);
        enc.set_bytes(head_size_,  10);
        enc.set_bytes(block_size_, 11);
        enc.set_bytes(x_,         12);

        uint64_t tg_size = std::min<uint64_t>(512, num_heads_ * head_size_);
        enc.dispatch_threadgroups(
            MTL::Size::Make(num_tokens, 1, 1),
            MTL::Size::Make(tg_size, 1, 1));
    }
};

// ── ReshapeAndCacheFlash ──────────────────────────────────────────────────────

struct ReshapeAndCacheFlash : mx::Primitive {
    std::string kernel_name_;
    int32_t num_heads_, head_size_, block_size_;

    ReshapeAndCacheFlash(mx::Stream s, std::string kname,
                         int32_t num_heads, int32_t head_size, int32_t block_size)
        : mx::Primitive(s), kernel_name_(std::move(kname)),
          num_heads_(num_heads), head_size_(head_size), block_size_(block_size) {}

    const char* name() const override { return "ReshapeAndCacheFlash"; }
    bool is_equivalent(const mx::Primitive& o) const override {
        return kernel_name_ == static_cast<const ReshapeAndCacheFlash&>(o).kernel_name_;
    }
    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("ReshapeAndCacheFlash is GPU-only");
    }

    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& s = stream();
        auto& d = mx::metal::device(s.device);

        outputs[0].copy_shared_buffer(inputs[2]);
        outputs[1].copy_shared_buffer(inputs[3]);

        int64_t num_tokens = inputs[0].shape(0);
        int32_t key_stride   = static_cast<int32_t>(inputs[0].strides(0));
        int32_t value_stride = static_cast<int32_t>(inputs[1].strides(0));

        std::string lib_path = getModuleDirectory() + "/" METALLIB_PATH;
        auto* lib = d.get_library("paged_attention_mlx", lib_path);
        auto* kernel = d.get_kernel(kernel_name_, lib);

        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(kernel);

        enc.set_input_array(inputs[0],   0);
        enc.set_input_array(inputs[1],   1);
        enc.set_output_array(outputs[0], 2);
        enc.set_output_array(outputs[1], 3);
        enc.set_input_array(inputs[4],   4);
        enc.set_bytes(key_stride,   5);
        enc.set_bytes(value_stride, 6);
        enc.set_bytes(num_heads_,   7);
        enc.set_bytes(head_size_,   8);
        enc.set_bytes(block_size_,  9);

        uint64_t tg_size = std::min<uint64_t>(512, num_heads_ * head_size_);
        enc.dispatch_threadgroups(
            MTL::Size::Make(num_tokens, 1, 1),
            MTL::Size::Make(tg_size, 1, 1));
    }
};

// ── CopyBlocksLayer ───────────────────────────────────────────────────────────

struct CopyBlocksLayer : mx::Primitive {
    std::string kernel_name_;
    int32_t numel_per_block_;
    int64_t num_pairs_;

    CopyBlocksLayer(mx::Stream s, std::string kname,
                    int32_t numel_per_block, int64_t num_pairs)
        : mx::Primitive(s), kernel_name_(std::move(kname)),
          numel_per_block_(numel_per_block), num_pairs_(num_pairs) {}

    const char* name() const override { return "CopyBlocksLayer"; }
    bool is_equivalent(const mx::Primitive& o) const override {
        const auto& p = static_cast<const CopyBlocksLayer&>(o);
        return kernel_name_ == p.kernel_name_ && numel_per_block_ == p.numel_per_block_;
    }
    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("CopyBlocksLayer is GPU-only");
    }

    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& s = stream();
        auto& d = mx::metal::device(s.device);

        outputs[0].copy_shared_buffer(inputs[0]);
        outputs[1].copy_shared_buffer(inputs[1]);

        std::string lib_path = getModuleDirectory() + "/" METALLIB_PATH;
        auto* lib = d.get_library("paged_attention_mlx", lib_path);
        auto* kernel = d.get_kernel(kernel_name_, lib);

        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(kernel);

        enc.set_output_array(outputs[0], 0);
        enc.set_output_array(outputs[1], 1);
        enc.set_input_array(inputs[2],   2);
        enc.set_bytes(numel_per_block_,  3);

        uint32_t tg_size = std::min<uint32_t>(256, numel_per_block_);
        enc.dispatch_threads(
            MTL::Size::Make((uint64_t)tg_size * num_pairs_, 1, 1),
            MTL::Size::Make(tg_size, 1, 1));
    }
};

// ── SwapBlocks ────────────────────────────────────────────────────────────────

struct SwapBlocks : mx::Primitive {
    int64_t block_size_bytes_;
    int64_t num_pairs_;

    SwapBlocks(mx::Stream s, int64_t block_size_bytes, int64_t num_pairs)
        : mx::Primitive(s),
          block_size_bytes_(block_size_bytes), num_pairs_(num_pairs) {}

    const char* name() const override { return "SwapBlocks"; }
    bool is_equivalent(const mx::Primitive& o) const override {
        const auto& p = static_cast<const SwapBlocks&>(o);
        return block_size_bytes_ == p.block_size_bytes_;
    }
    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("SwapBlocks is GPU-only");
    }

    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& s = stream();
        auto& d = mx::metal::device(s.device);

        outputs[0].copy_shared_buffer(inputs[1]);

        auto* cmd_buf = d.get_command_buffer(s.index);
        auto* blit    = cmd_buf->blitCommandEncoder();

        auto* src_buf = static_cast<MTL::Buffer*>(const_cast<void*>(inputs[0].buffer().ptr()));
        auto* dst_buf = static_cast<MTL::Buffer*>(const_cast<void*>(outputs[0].buffer().ptr()));

        const int64_t* bm = inputs[2].data<int64_t>();
        for (int64_t i = 0; i < num_pairs_; ++i) {
            NS::UInteger src_off = static_cast<NS::UInteger>(bm[i * 2    ] * block_size_bytes_);
            NS::UInteger dst_off = static_cast<NS::UInteger>(bm[i * 2 + 1] * block_size_bytes_);
            blit->copyFromBuffer(src_buf, src_off, dst_buf, dst_off,
                                 static_cast<NS::UInteger>(block_size_bytes_));
        }
        blit->endEncoding();
    }
};

// ── Public API ────────────────────────────────────────────────────────────────

std::vector<mx::array> reshape_and_cache(
    const mx::array& key, const mx::array& value,
    const mx::array& key_cache, const mx::array& value_cache,
    const mx::array& slot_mapping)
{
    auto s = mx::default_stream(mx::Device::gpu);
    std::string kname = "reshape_and_cache_kv_"
        + dtypeStr(key.dtype()) + "_cache_" + dtypeStr(key_cache.dtype());

    int32_t num_heads  = static_cast<int32_t>(key.shape(1));
    int32_t head_size  = static_cast<int32_t>(key.shape(2));
    int32_t block_size = static_cast<int32_t>(key_cache.shape(3));
    int32_t x          = static_cast<int32_t>(key_cache.shape(4));

    auto prim = std::make_shared<ReshapeAndCache>(
        s, kname, num_heads, head_size, block_size, x);

    return mx::array::make_arrays(
        {key_cache.shape(), value_cache.shape()},
        {key_cache.dtype(), value_cache.dtype()},
        prim,
        {key, value, key_cache, value_cache, mx::astype(slot_mapping, mx::int32)});
}

std::vector<mx::array> reshape_and_cache_flash(
    const mx::array& key, const mx::array& value,
    const mx::array& key_cache, const mx::array& value_cache,
    const mx::array& slot_mapping)
{
    auto s = mx::default_stream(mx::Device::gpu);
    std::string kname = "reshape_and_cache_flash_" + dtypeStr(key.dtype());

    int32_t num_heads  = static_cast<int32_t>(key.shape(1));
    int32_t head_size  = static_cast<int32_t>(key.shape(2));
    int32_t block_size = static_cast<int32_t>(key_cache.shape(1));

    auto prim = std::make_shared<ReshapeAndCacheFlash>(
        s, kname, num_heads, head_size, block_size);

    return mx::array::make_arrays(
        {key_cache.shape(), value_cache.shape()},
        {key_cache.dtype(), value_cache.dtype()},
        prim,
        {key, value, key_cache, value_cache, mx::astype(slot_mapping, mx::int32)});
}

std::pair<std::vector<mx::array>, std::vector<mx::array>> copy_blocks(
    const std::vector<mx::array>& key_caches,
    const std::vector<mx::array>& value_caches,
    const mx::array& block_mapping)
{
    auto s = mx::default_stream(mx::Device::gpu);
    if (key_caches.size() != value_caches.size())
        throw std::invalid_argument("key_caches and value_caches size mismatch");

    int64_t num_pairs     = block_mapping.shape(0);
    int64_t num_layers    = static_cast<int64_t>(key_caches.size());

    std::vector<mx::array> new_keys, new_vals;
    new_keys.reserve(num_layers);
    new_vals.reserve(num_layers);

    for (int64_t i = 0; i < num_layers; ++i) {
        const auto& kc = key_caches[i];
        const auto& vc = value_caches[i];
        int32_t numel = static_cast<int32_t>(kc.size() / kc.shape(0));

        std::string kname = "copy_blocks_" + dtypeStr(kc.dtype());

        auto prim = std::make_shared<CopyBlocksLayer>(s, kname, numel, num_pairs);
        auto outs = mx::array::make_arrays(
            {kc.shape(), vc.shape()},
            {kc.dtype(), vc.dtype()},
            prim,
            {kc, vc, block_mapping});

        new_keys.push_back(outs[0]);
        new_vals.push_back(outs[1]);
    }
    return {new_keys, new_vals};
}

mx::array swap_blocks(
    const mx::array& src, const mx::array& dst,
    const mx::array& block_mapping)
{
    auto s = mx::default_stream(mx::Device::gpu);
    int64_t block_numel = src.size() / src.shape(0);
    int64_t block_bytes = static_cast<int64_t>(block_numel * src.itemsize());
    int64_t num_pairs   = block_mapping.shape(0);

    auto prim   = std::make_shared<SwapBlocks>(s, block_bytes, num_pairs);

    return mx::array(dst.shape(), dst.dtype(), prim, {src, dst, block_mapping});
}
