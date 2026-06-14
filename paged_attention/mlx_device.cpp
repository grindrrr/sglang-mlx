// This file is ported from EricB/kernels-paged-attention-metal (https://huggingface.co/EricB/kernels-paged-attention-metal)
// Modified for MLX integration

// mlx_device.mm
// Device attribute queries for the MLX extension.

#include "mlx_ops.h"
#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>
#include <stdexcept>

int64_t get_device_attribute(int64_t /*attribute*/, int64_t /*device_id*/) {
    throw std::runtime_error("get_device_attribute is not supported on Metal");
}

int64_t get_max_shared_memory_per_block_device_attribute(int64_t /*device_id*/) {
    // MLX exposes the MTLDevice through its metal backend.
    auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
    return static_cast<int64_t>(d.mtl_device()->maxThreadgroupMemoryLength());
}
