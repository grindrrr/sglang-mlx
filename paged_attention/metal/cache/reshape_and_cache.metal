// This file is ported from EricB/kernels-paged-attention-metal (https://huggingface.co/EricB/kernels-paged-attention-metal)
// Modified for MLX integration

#include "../utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename KV_T, typename CACHE_T>
inline CACHE_T to_cache(KV_T v) {
    return static_cast<CACHE_T>(v);
}

template <typename KV_T, typename CACHE_T>
[[kernel]] void reshape_and_cache(
    const device KV_T *__restrict__ key
    [[buffer(0)]], // [num_tokens, num_heads, head_size]
    const device KV_T *__restrict__ value
    [[buffer(1)]], // [num_tokens, num_heads, head_size]
    device CACHE_T *__restrict__ key_cache
    [[buffer(2)]], // [num_blocks, num_heads, head_size/x, block_size, x]
    device CACHE_T *__restrict__ value_cache
    [[buffer(3)]], // [num_blocks, num_heads, head_size, block_size]
    const device int *__restrict__ slot_mapping
    [[buffer(4)]], // [num_tokens] (Changed from int64 to int)
    device const int &key_stride [[buffer(7)]],
    device const int &value_stride [[buffer(8)]],
    device const int &num_heads [[buffer(9)]],
    device const int &head_size [[buffer(10)]],
    device const int &block_size [[buffer(11)]],
    device const int &x [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int token_idx = gid;
  const int slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        (int64_t)block_idx * (num_heads * (head_size / x) * block_size * x) +
        (int64_t)head_idx  * ((head_size / x) * block_size * x) +
        (int64_t)x_idx     * (block_size * x) +
        (int64_t)block_offset * x +
        (int64_t)x_offset;

    const int64_t tgt_value_idx =
        (int64_t)block_idx * (num_heads * head_size * block_size) +
        (int64_t)head_idx  * (head_size * block_size) +
        (int64_t)head_offset * block_size +
        (int64_t)block_offset;

    key_cache[tgt_key_idx] = to_cache<KV_T, CACHE_T>(key[src_key_idx]);
    value_cache[tgt_value_idx] = to_cache<KV_T, CACHE_T>(value[src_value_idx]);
  }
}

#define instantiate_reshape_and_cache(kv_type, cache_type)                     \
  template [[host_name("reshape_and_cache_kv_" #kv_type                        \
                       "_cache_" #cache_type)]] [[kernel]] void                \
  reshape_and_cache<kv_type, cache_type>(                                      \
      const device kv_type *__restrict__ key [[buffer(0)]],                    \
      const device kv_type *__restrict__ value [[buffer(1)]],                  \
      device cache_type *__restrict__ key_cache [[buffer(2)]],                 \
      device cache_type *__restrict__ value_cache [[buffer(3)]],               \
      const device int *__restrict__ slot_mapping [[buffer(4)]],               \
      device const int &key_stride [[buffer(7)]],                              \
      device const int &value_stride [[buffer(8)]],                            \
      device const int &num_heads [[buffer(9)]],                               \
      device const int &head_size [[buffer(10)]],                              \
      device const int &block_size [[buffer(11)]],                             \
      device const int &x [[buffer(12)]],                                      \
      uint gid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_reshape_and_cache(float, float);
instantiate_reshape_and_cache(bfloat16_t, bfloat16_t);
instantiate_reshape_and_cache(half, half);
instantiate_reshape_and_cache(float, half);
instantiate_reshape_and_cache(float, bfloat16_t);

// Flash version
template <typename T>
[[kernel]] void reshape_and_cache_flash(
    const device T *__restrict__ key
    [[buffer(0)]], // [num_tokens, num_heads, head_size]
    const device T *__restrict__ value
    [[buffer(1)]], // [num_tokens, num_heads, head_size]
    device T *__restrict__ key_cache
    [[buffer(2)]], // [num_blocks, block_size, num_heads, head_size]
    device T *__restrict__ value_cache
    [[buffer(3)]], // [num_blocks, block_size, num_heads, head_size]
    const device int *__restrict__ slot_mapping
    [[buffer(4)]], // [num_tokens]
    device const int &key_stride, device const int &value_stride,
    device const int &num_heads, device const int &head_size,
    device const int &block_size, uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int token_idx = gid;
  const int slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int64_t src_key_idx = (int64_t)token_idx * key_stride + i;
    const int64_t src_value_idx = (int64_t)token_idx * value_stride + i;

    const int64_t tgt_key_idx = (int64_t)block_idx * block_size * num_heads * head_size +
                                (int64_t)block_offset * num_heads * head_size + i;
    const int64_t tgt_value_idx = (int64_t)block_idx * block_size * num_heads * head_size +
                                  (int64_t)block_offset * num_heads * head_size + i;
    
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

#define instantiate_reshape_and_cache_flash(type)                              \
  template [[host_name("reshape_and_cache_flash_" #type)]] [[kernel]] void     \
  reshape_and_cache_flash<type>(                                               \
      const device type *__restrict__ key [[buffer(0)]],                       \
      const device type *__restrict__ value [[buffer(1)]],                     \
      device type *__restrict__ key_cache [[buffer(2)]],                       \
      device type *__restrict__ value_cache [[buffer(3)]],                     \
      const device int *__restrict__ slot_mapping [[buffer(4)]],               \
      device const int &key_stride, device const int &value_stride,            \
      device const int &num_heads, device const int &head_size,                \
      device const int &block_size, uint gid [[threadgroup_position_in_grid]], \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_reshape_and_cache_flash(float);
instantiate_reshape_and_cache_flash(bfloat16_t);
instantiate_reshape_and_cache_flash(half);
