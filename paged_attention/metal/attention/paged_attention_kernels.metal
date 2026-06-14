#pragma once
#include "paged_attention_utils.metal"

using namespace metal;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

constant bool use_alibi [[function_constant(20)]];

template <typename T, typename CACHE_T, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int NUM_SIMD_LANES>
[[kernel]] void paged_attention(
    device T *out [[buffer(2)]],
    device const T *q [[buffer(3)]],
    device const CACHE_T *k_cache [[buffer(4)]],
    device const CACHE_T *v_cache [[buffer(5)]],
    const constant int &num_kv_heads [[buffer(6)]],
    const constant float &scale [[buffer(7)]],
    const constant float &softcapping [[buffer(8)]],
    device const uint32_t *block_tables [[buffer(9)]],
    device const uint32_t *context_lens [[buffer(10)]],
    const constant int &max_num_blocks_per_seq [[buffer(11)]],
    device const float *alibi_slopes [[buffer(12)]],
    const constant int &q_stride [[buffer(13)]],
    const constant int &kv_block_stride [[buffer(14)]],
    const constant int &kv_head_stride [[buffer(15)]],
    const constant int &max_seq_len [[buffer(16)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  
  const int seq_idx = (int)threadgroup_position_in_grid.y;
  const int head_idx = (int)threadgroup_position_in_grid.x;
  const int num_heads = (int)threadgroups_per_grid.x;
  const int thread_idx = (int)thread_position_in_threadgroup.x;

  device T *out_ptr = out + (int64_t)seq_idx * num_heads * HEAD_SIZE +
                      (int64_t)head_idx * HEAD_SIZE;
  const uint32_t context_len = context_lens[seq_idx];
  if (context_len == 0 || context_len > (uint32_t)max_seq_len) {
    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
      out_ptr[i] = (T)0;
    }
    return;
  }
  
  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = (int)simd_tid;
  const int lane = (int)simd_lid;

  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = !use_alibi ? 0.f : alibi_slopes[head_idx];

  // Shared Memory Offsets
  threadgroup float *logits = (threadgroup float *)(shared_mem);
  threadgroup float *red_smem = logits + max_seq_len;
  threadgroup float *out_smem = red_smem + 128;
  
  // Initialize
  for (int i = thread_idx; i < max_seq_len; i += NUM_THREADS) logits[i] = -FLT_MAX;
  for (int i = thread_idx; i < NUM_WARPS * HEAD_SIZE; i += NUM_THREADS) out_smem[i] = 0.f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q into REGISTERS
  float q_local[HEAD_SIZE];
  const device T *q_ptr = q + (int64_t)seq_idx * q_stride + (int64_t)head_idx * HEAD_SIZE;
  for (int i = 0; i < HEAD_SIZE; ++i) q_local[i] = (float)q_ptr[i];

  // Step 1: Q*K
  float qk_max = -FLT_MAX;
  const device uint32_t *block_table = block_tables + (int64_t)seq_idx * max_num_blocks_per_seq;
  
  constexpr int x = 16 / sizeof(CACHE_T);
  for (int block_idx = warp_idx; block_idx < num_context_blocks; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const device CACHE_T *k_ptr = k_cache + physical_block_number * kv_block_stride + (int64_t)kv_head_idx * kv_head_stride;

    for (int t = lane; t < BLOCK_SIZE; t += NUM_SIMD_LANES) {
        const int token_idx = block_idx * BLOCK_SIZE + t;
        if (token_idx < (int)context_len) {
            float qk = 0.f;
            for (int d = 0; d < HEAD_SIZE; ++d) {
                const int offset1 = d / x;
                const int offset2 = d % x;
                float k_val = (float)k_ptr[(int64_t)offset1 * BLOCK_SIZE * x + (int64_t)t * x + offset2];
                qk += q_local[d] * k_val;
            }
            qk *= scale;
            if (softcapping != 0.0f && softcapping != 1.0f) qk = precise::tanh(qk / softcapping) * softcapping;
            if (use_alibi && alibi_slope != 0) qk += alibi_slope * float(token_idx - int(context_len) + 1);
            logits[token_idx] = qk;
            qk_max = metal::max(qk_max, qk);
        }
    }
  }

  // Softmax Max Reduction
#pragma clang loop unroll(full)
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    qk_max = metal::max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  if (lane == 0) red_smem[warp_idx] = qk_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_idx < NUM_SIMD_LANES) {
    float m = (thread_idx < NUM_WARPS) ? red_smem[thread_idx] : -FLT_MAX;
#pragma clang loop unroll(full)
    for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
      m = metal::max(m, simd_shuffle_xor(m, mask));
    }
    if (thread_idx == 0) red_smem[127] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  qk_max = red_smem[127];

  // Softmax Exp Sum Reduction
  float exp_sum = 0.f;
  for (int i = thread_idx; i < (int)context_len; i += NUM_THREADS) {
    float val = (logits[i] == -FLT_MAX) ? 0.f : metal::exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
#pragma clang loop unroll(full)
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    exp_sum += simd_shuffle_xor(exp_sum, mask);
  }
  if (lane == 0) red_smem[warp_idx] = exp_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_idx < NUM_SIMD_LANES) {
    float s = (thread_idx < NUM_WARPS) ? red_smem[thread_idx] : 0.f;
#pragma clang loop unroll(full)
    for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
      s += simd_shuffle_xor(s, mask);
    }
    if (thread_idx == 0) red_smem[126] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_sum = divide(1.f, red_smem[126] + 1e-6f);

  // Step 2: Logits * V
  for (int block_idx = warp_idx; block_idx < num_context_blocks; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const device CACHE_T *v_ptr = v_cache + physical_block_number * kv_block_stride + (int64_t)kv_head_idx * kv_head_stride;
    for (int d = lane; d < HEAD_SIZE; d += NUM_SIMD_LANES) {
        float acc = 0.f;
        for (int t = 0; t < BLOCK_SIZE; t++) {
            const int token_idx = block_idx * BLOCK_SIZE + t;
            if (token_idx < (int)context_len) {
                acc += (logits[token_idx] * inv_sum) * (float)v_ptr[(int64_t)d * BLOCK_SIZE + t];
            }
        }
        out_smem[warp_idx * HEAD_SIZE + d] += acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 3: Write Output
  for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
    float final_acc = 0.f;
    for (int w = 0; w < NUM_WARPS; w++) final_acc += out_smem[w * HEAD_SIZE + i];
    out_ptr[i] = (T)final_acc;
  }
}
