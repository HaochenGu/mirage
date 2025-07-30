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
#include "utils.cuh"
#include "reduction.cuh"
#include "softmax.cuh"  // For block_reduce_max and block_reduce_sum

namespace kernel {

using bfloat16 = type::bfloat16_t;

template <typename T,
          int VOCAB_SIZE,
          int MAX_TOKENS>
__device__ __forceinline__ void multi_token_softmax_kernel(
    void const *input_ptr,      // Points to this token's input slice
    void *output_ptr,           // Points to this token's output slice
    int num_tokens,             // Runtime number of tokens
    float temperature = 1.0f) 
{
    // Simple softmax implementation
    // MPK handles token partitioning - each block gets one token's data
    
    // Cast pointers
    T const *input = static_cast<T const *>(input_ptr);
    T *output = static_cast<T *>(output_ptr);
    
    // Step 1: Find max value for numerical stability
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < VOCAB_SIZE; i += blockDim.x) {
        float val = float(input[i]) / temperature;
        thread_max = fmaxf(thread_max, val);
    }
    
    // Reduce max across block
    block_reduce_max(thread_max);
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = thread_max;
    }
    __syncthreads();
    float max_val = shared_max;
    
    // Step 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < VOCAB_SIZE; i += blockDim.x) {
        float val = float(input[i]) / temperature;
        float exp_val = expf(val - max_val);
        output[i] = T(exp_val); // Store exp values temporarily
        thread_sum += exp_val;
    }
    
    // Reduce sum across block
    block_reduce_sum(thread_sum);
    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = thread_sum;
    }
    __syncthreads();
    float sum_val = shared_sum;
    
    // Step 3: Normalize by sum
    float inv_sum = 1.0f / sum_val;
    for (int i = threadIdx.x; i < VOCAB_SIZE; i += blockDim.x) {
        float normalized = float(output[i]) * inv_sum;
        output[i] = T(normalized);
    }
}

} // namespace kernel