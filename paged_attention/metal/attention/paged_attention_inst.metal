#include "paged_attention_kernels.metal"

using namespace metal;

#define instantiate_paged_attention_inner(type, cache_type, head_size,         \
                                          block_size, num_threads,             \
                                          num_simd_lanes)                      \
  template [[host_name("paged_attention_" #type "_cache_" #cache_type          \
                       "_hs" #head_size "_bs" #block_size "_nt" #num_threads   \
                       "_nsl" #num_simd_lanes)]] [[kernel]] void               \
  paged_attention<type, cache_type, head_size, block_size, num_threads,        \
                  num_simd_lanes>(                                             \
      device float *exp_sums [[buffer(0)]],                                    \
      device float *max_logits [[buffer(1)]],                                  \
      device type *out [[buffer(2)]], device const type *q [[buffer(3)]],      \
      device const cache_type *k_cache [[buffer(4)]],                          \
      device const cache_type *v_cache [[buffer(5)]],                          \
      const constant int &num_kv_heads [[buffer(6)]],                          \
      const constant float &scale [[buffer(7)]],                               \
      const constant float &softcapping [[buffer(8)]],                         \
      device const uint32_t *block_tables [[buffer(9)]],                       \
      device const uint32_t *context_lens [[buffer(10)]],                      \
      const constant int &max_num_blocks_per_seq [[buffer(11)]],               \
      device const float *alibi_slopes [[buffer(12)]],                         \
      const constant int &q_stride [[buffer(13)]],                             \
      const constant int &kv_block_stride [[buffer(14)]],                      \
      const constant int &kv_head_stride [[buffer(15)]],                       \
      const constant int &max_seq_len [[buffer(16)]],                          \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_heads(                                     \
    type, cache_type, block_size, num_threads, num_simd_lanes)                 \
  instantiate_paged_attention_inner(type, cache_type, 32, block_size,          \
                                    num_threads, num_simd_lanes);              \
  instantiate_paged_attention_inner(type, cache_type, 64, block_size,          \
                                    num_threads, num_simd_lanes);              \
  instantiate_paged_attention_inner(type, cache_type, 80, block_size,          \
                                    num_threads, num_simd_lanes);              \
  instantiate_paged_attention_inner(type, cache_type, 96, block_size,          \
                                    num_threads, num_simd_lanes);              \
  instantiate_paged_attention_inner(type, cache_type, 112, block_size,         \
                                    num_threads, num_simd_lanes);              \
  instantiate_paged_attention_inner(type, cache_type, 120, block_size,         \
                                    num_threads, num_simd_lanes);              \
  instantiate_paged_attention_inner(type, cache_type, 128, block_size,         \
                                    num_threads, num_simd_lanes);              \
  instantiate_paged_attention_inner(type, cache_type, 192, block_size,         \
                                    num_threads, num_simd_lanes);              \
  instantiate_paged_attention_inner(type, cache_type, 256, block_size,         \
                                    num_threads, num_simd_lanes);

#define instantiate_paged_attention_block_size(type, cache_type, num_threads,  \
                                               num_simd_lanes)                 \
  instantiate_paged_attention_heads(type, cache_type, 8, num_threads,          \
                                    num_simd_lanes);                           \
  instantiate_paged_attention_heads(type, cache_type, 16, num_threads,         \
                                    num_simd_lanes);                           \
  instantiate_paged_attention_heads(type, cache_type, 32, num_threads,         \
                                    num_simd_lanes);

// V1
#define instantiate_paged_attention_v1(type, cache_type, num_simd_lanes)       \
  instantiate_paged_attention_block_size(type, cache_type, 256,                \
                                         num_simd_lanes);

instantiate_paged_attention_v1(float, float, 32);
instantiate_paged_attention_v1(bfloat16_t, bfloat16_t, 32);
instantiate_paged_attention_v1(half, half, 32);
instantiate_paged_attention_v1(float, half, 32);
instantiate_paged_attention_v1(float, bfloat16_t, 32);
