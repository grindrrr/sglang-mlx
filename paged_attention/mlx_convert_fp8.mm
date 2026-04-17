// This file is ported from EricB/kernels-paged-attention-metal (https://huggingface.co/EricB/kernels-paged-attention-metal)
// Modified for MLX integration

// mlx_convert_fp8.mm
// MLX primitive for FP8 format conversion.

#include "mlx_ops.h"
#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>

#include <dlfcn.h>
#include <stdexcept>
#include <string>

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

static std::string fp8DtypeStr(mx::Dtype d) {
    switch (d) {
        case mx::float32:  return "float";
        case mx::float16:  return "half";
        case mx::bfloat16: return "bfloat16_t";
        case mx::uint8:    return "uchar";
        default: throw std::invalid_argument("Unsupported dtype for convert_fp8");
    }
}

// ── ConvertFP8 ────────────────────────────────────────────────────────────────

struct ConvertFP8 : mx::Primitive {
    std::string kernel_name_;
    float scale_;

    ConvertFP8(mx::Stream s, std::string kname, float scale)
        : mx::Primitive(s), kernel_name_(std::move(kname)), scale_(scale) {}

    const char* name() const override { return "ConvertFP8"; }
    bool is_equivalent(const mx::Primitive& o) const override {
        const auto& p = static_cast<const ConvertFP8&>(o);
        return kernel_name_ == p.kernel_name_ && scale_ == p.scale_;
    }
    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("ConvertFP8 is GPU-only");
    }

    // inputs: [src_cache(0)]
    // outputs: [dst_cache(0)]
    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& s = stream();
        auto& d = mlx::core::metal::device(s.device);

        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));

        uint32_t num_elements = static_cast<uint32_t>(inputs[0].size());
        if (num_elements == 0) return;

        std::string lib_path = getModuleDirectory() + "/" METALLIB_PATH;
        auto* lib    = d.get_library("paged_attention_mlx", lib_path);
        auto* kernel = d.get_kernel(kernel_name_, lib);

        auto& enc = d.get_command_encoder(s.index);
        enc.set_compute_pipeline_state(kernel);

        enc.set_input_array(inputs[0],  0);    // src
        enc.set_output_array(out,       1);    // dst
        enc.set_bytes(scale_,           2);
        enc.set_bytes(num_elements,     3);

        uint32_t tg_size    = std::min<uint32_t>(1024, num_elements);
        uint32_t num_groups = (num_elements + tg_size - 1) / tg_size;
        enc.dispatch_threadgroups(
            MTL::Size::Make(num_groups, 1, 1),
            MTL::Size::Make(tg_size,   1, 1));
    }
};

// ── Public API ────────────────────────────────────────────────────────────────

mx::array convert_fp8(
    const mx::array& src_cache,
    float scale,
    const std::string& kv_cache_dtype,
    const std::string& dst_dtype_str)
{
    mx::Dtype dst_dtype = (dst_dtype_str == "float16" || dst_dtype_str == "fp16")
        ? mx::float16
        : (dst_dtype_str == "bfloat16" || dst_dtype_str == "bf16")
            ? mx::bfloat16
            : mx::float32;

    auto s = mx::default_stream(mx::Device::gpu);
    std::string kname = "convert_fp8_"
        + fp8DtypeStr(src_cache.dtype()) + "_to_" + fp8DtypeStr(dst_dtype);

    return mx::array(
        src_cache.shape(), dst_dtype,
        std::make_shared<ConvertFP8>(s, kname, scale),
        {src_cache});
}
