/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "common.h"
#include "copy_sm80.cuh"
#include "dmem_layout.cuh"
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "reduction.cuh"
#include "smem_layout.cuh"
#include "utils.cuh"

namespace kernel {

using bfloat16 = type::bfloat16_t;

// Multi-token linear kernel for EAGLE tree generation
// Input: concatenated features [feat0 || feat1 || ... || featN] with shape [1, num_tokens * REDUCTION_SIZE]
// Weight: standard weight matrix [REDUCTION_SIZE, OUTPUT_SIZE]
// Residual: concatenated residuals [res0 || res1 || ... || resN] with shape [1, num_tokens * OUTPUT_SIZE]
// Output: concatenated outputs [out0 || out1 || ... || outN] with shape [1, num_tokens * OUTPUT_SIZE]
// 
// Each block processes one token's linear transformation using multiple blocks per token
// Block assignment: token_id = blockIdx.x / BLOCKS_PER_TOKEN
//                  block_in_token = blockIdx.x % BLOCKS_PER_TOKEN
template <typename T,
          int MAX_TOKENS,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int BLOCKS_PER_TOKEN = 1,
          int K_PIPE_MAX = 3>
__device__ __forceinline__ void multi_token_linear_kernel(
    void const *input_ptr,      // [1, num_tokens * REDUCTION_SIZE]
    void const *weight_ptr,     // [REDUCTION_SIZE, OUTPUT_SIZE]
    void const *residual_ptr,   // [1, num_tokens * OUTPUT_SIZE] or nullptr
    void *output_ptr,           // [1, num_tokens * OUTPUT_SIZE]
    int num_tokens,
    bool residual = true) {
  
  // Derive token ID and block within token from block index
  int token_id = blockIdx.x / BLOCKS_PER_TOKEN;
  int block_in_token = blockIdx.x % BLOCKS_PER_TOKEN;
  
  // Early exit if this block is beyond the number of tokens
  if (token_id >= num_tokens) return;
  
  // Constants - maintain batch size of 1 as per MPK architecture
  constexpr int BATCH_SIZE = 1;
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  
  // Divide output among blocks for this token
  constexpr int OUTPUT_PER_BLOCK = OUTPUT_SIZE / BLOCKS_PER_TOKEN;
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_PER_BLOCK <= 128 ? OUTPUT_PER_BLOCK : 128;
  constexpr int NUM_OUTPUT_ATOMS = OUTPUT_PER_BLOCK / OUTPUT_ATOM_SIZE;
  
  // Calculate this block's output range within the token
  int block_output_start = block_in_token * OUTPUT_PER_BLOCK;
  int block_output_end = min(block_output_start + OUTPUT_PER_BLOCK, OUTPUT_SIZE);
  
  constexpr int TILE_SIZE = 128;
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;
  
  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_C = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  
  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  
  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);
  
  // Warp configuration
  constexpr int NUM_WARPS_N = OUTPUT_ATOM_SIZE / 16 <= 4 ? OUTPUT_ATOM_SIZE / 16 : 4;
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;
  
  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N = OUTPUT_ATOM_SIZE / NUM_WARPS_N / 16;
  constexpr int NUM_ITERS_K = TILE_SIZE / NUM_WARPS_K / 16;
  
  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);
  
  int warp_idx = warp_id();
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = lane_id();
  
  // Adjust pointers for this token using flattened layout
  // Input: offset by token_id * REDUCTION_SIZE
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr) + 
                                  token_id * REDUCTION_SIZE;
  // Weight: offset by block's output range
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr) + 
                                   block_output_start * REDUCTION_SIZE;
  // Residual: offset by token_id * OUTPUT_SIZE + block's output range
  T const *__restrict__ d_residual = residual ? 
      static_cast<T const *>(residual_ptr) + token_id * OUTPUT_SIZE + block_output_start : 
      nullptr;
  // Output: offset by token_id * OUTPUT_SIZE + block's output range
  T *__restrict__ d_output = static_cast<T *>(output_ptr) + 
                             token_id * OUTPUT_SIZE + block_output_start;
  
  // Define memory layouts - use stride to handle flattened layout
  // The key is that we're treating the flattened data as having batch size 1
  // with the actual stride being the full concatenated size
  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, MAX_TOKENS * REDUCTION_SIZE>;
  using WeightDmem = dmem_col_const<T, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;
  using ResidualDmem = dmem_row_const<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, MAX_TOKENS * OUTPUT_SIZE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, MAX_TOKENS * OUTPUT_SIZE>;
  
  InputDmem input_dmem(d_input);
  WeightDmem weight_dmem(d_weight);
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);
  
  extern __shared__ char smem[];
  
  // Shared memory layout (same as original)
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  constexpr size_t SHARED_INPUT_BUFFER_OFFSET = ZERO_BUFFER_OFFSET + sizeof(T) * 8;
  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET = 
      SHARED_INPUT_BUFFER_OFFSET + sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE;
  constexpr size_t SHARED_RESIDUAL_OFFSET = 
      SHARED_WEIGHT_BUFFER_OFFSET + sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE;
  constexpr size_t MM_INTERMEDIATE_OFFSET = 
      SHARED_RESIDUAL_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  constexpr size_t SHARED_OUTPUT_OFFSET = 
      MM_INTERMEDIATE_OFFSET + sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  
  // Initialize shared memory pointers
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  *((__uint128_t *)zero_buf) = 0ul;
  
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);
  T *shared_residual = residual ? (T *)(smem + SHARED_RESIDUAL_OFFSET) : nullptr;
  T *mm_intermediate = (T *)(smem + MM_INTERMEDIATE_OFFSET);
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);
  
  // Define swizzle patterns
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using InputBufferSmem = smem_row<T, 0, 0, 0, K_PIPE_MAX * BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem = smem_col<T, 3, 3, 3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using WeightBufferSmem = smem_col<T, 3, 3, 3, TILE_SIZE, K_PIPE_MAX * OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using OutputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;
  using MatMulIntermediateSmem = smem_row<T, 0, 0, 0, NUM_WARPS_K * BATCH_SIZE, 
                                          OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;
  
  ZeroBufferSmem zero_buffer(zero_buf);
  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);
  OutputSmem residual_smem(shared_residual);
  MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);
  OutputSmem output_smem(shared_output);
  
  // Process output atoms
  for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++,
       d_weight += OUTPUT_ATOM_SIZE * REDUCTION_SIZE,
       d_residual = residual ? d_residual + OUTPUT_ATOM_SIZE : nullptr,
       d_output += OUTPUT_ATOM_SIZE) {
    
    weight_dmem.set_ptr(d_weight);
    residual_dmem.set_ptr(d_residual);
    output_dmem.set_ptr(d_output);
    
    InputBufferSmem input_buffer_smem(shared_input_buffer);
    WeightBufferSmem weight_buffer_smem(shared_weight_buffer);
    
    // Load residual if needed
    if (residual) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_C; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_C;
        int col = (i & (CHUNKS_PER_ROW_C - 1)) << log2_CHUNK_SIZE;
        load_smem(residual_smem(row, col), residual_dmem(row, col));
      }
    }
    
    // Pipeline initial loads
#pragma unroll
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        int src_row = i >> log2_CHUNKS_PER_ROW_A;
        int dst_row = src_row + ((k_pipe + 1) << log2_constexpr(BATCH_SIZE));
        int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (k_pipe << log2_constexpr(TILE_SIZE));
        load_smem(input_buffer_smem(dst_row, dst_col), input_dmem(src_row, src_col));
      }
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
        int src_row = dst_row + (k_pipe << log2_constexpr(TILE_SIZE));
        int src_col = i >> log2_CHUNKS_PER_COL_B;
        int dst_col = src_col + ((k_pipe + 1) << log2_constexpr(OUTPUT_ATOM_SIZE));
        load_smem(weight_buffer_smem(dst_row, dst_col), weight_dmem(src_row, src_col));
      }
      cp_async_fence();
    }
    
    // Initialize accumulator
    float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[m][n]);
      }
    }
    
    // Main computation loop
    for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
      // Continue pipelining
      if (for_idx + K_PIPE_MAX - 1 < FORLOOP_RANGE) {
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
          int row = i >> log2_CHUNKS_PER_ROW_A;
          int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int src_col = dst_col + ((for_idx + K_PIPE_MAX - 1) << log2_constexpr(TILE_SIZE));
          load_smem(input_buffer_smem(row, dst_col), input_dmem(row, src_col));
        }
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
          int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int src_row = dst_row + ((for_idx + K_PIPE_MAX - 1) << log2_constexpr(TILE_SIZE));
          int col = i >> log2_CHUNKS_PER_COL_B;
          load_smem(weight_buffer_smem(dst_row, col), weight_dmem(src_row, col));
        }
        cp_async_fence();
        cp_async_wait<K_PIPE_MAX - 1>();
      } else if (for_idx + K_PIPE_MAX - 1 == FORLOOP_RANGE) {
        cp_async_wait<0>();
      }
      
      // Rotate buffers
      input_buffer_smem.set_ptr(shared_input_buffer + 
                                BATCH_SIZE * TILE_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      input_smem.set_ptr(shared_input_buffer + 
                         BATCH_SIZE * TILE_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      weight_buffer_smem.set_ptr(shared_weight_buffer + 
                                 TILE_SIZE * OUTPUT_ATOM_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      weight_smem.set_ptr(shared_weight_buffer + 
                          TILE_SIZE * OUTPUT_ATOM_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      __syncthreads();
      
      // Tensor core operations
      uint32_t a_frag[4], b_frag[4];
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF);
        bool is_valid = (m_row < BATCH_SIZE);
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          int n_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
          for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        ((lane_idx >> 4) << 3);
            int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        (((lane_idx & 0xF) >> 3) << 3);
            T *src_ptr = is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
            ldsm(src_ptr, a_frag);
            ldsm(weight_smem(n_row, n_col), b_frag);
            mma_m16n16k16_bf16bf16bf32(s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
          }
        }
      }
      __syncthreads();
    }
    
    // Write results back to shared memory
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
          if (row_in_warp < BATCH_SIZE) {
            int col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
            mm_intermediate_smem.at(warp_row + row_in_warp, col) =
                bfloat16(s_frag[m][n][(i << 1)]);
            mm_intermediate_smem.at(warp_row + row_in_warp, col + 1) =
                bfloat16(s_frag[m][n][(i << 1) | 0x1]);
          }
        }
      }
    }
    __syncthreads();
    
    // Reduce across warps if needed
    if (NUM_WARPS_K > 1) {
      reduction_sum_row<decltype(output_smem), decltype(mm_intermediate_smem)>(
          output_smem, mm_intermediate_smem);
      __syncthreads();
    }
    
    // Write final output with residual
#pragma unroll
    for (int i = threadIdx.x; i < OUTPUT_ATOM_SIZE; i += NUM_THREADS) {
      int row = 0;
      T val = NUM_WARPS_K > 1 ? output_smem.at(row, i) : mm_intermediate_smem.at(row, i);
      output_dmem.at(row, i) = residual ? val + residual_smem.at(row, i) : val;
    }
    
    if (output_atom_idx + 1 < NUM_OUTPUT_ATOMS) {
      __syncthreads();
    }
  }
}

} // namespace kernel