#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void scaled_dot_product_attention_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    float* __restrict__ output,
    int sequence_length,
    int head_dimension,
    float scale,
    bool causal) {
    extern __shared__ float shared_buffer[];
    float* score_buffer = shared_buffer;
    float* reduction_buffer = shared_buffer + blockDim.x;
    const int row_index = blockIdx.x;
    const int batch_head_index = blockIdx.y;
    const int query_position = row_index;
    const int thread_index = threadIdx.x;
    const int query_offset = (batch_head_index * sequence_length + query_position) * head_dimension;
    const int key_value_offset = batch_head_index * sequence_length * head_dimension;

    for (int key_position = thread_index; key_position < sequence_length; key_position += blockDim.x) {
        if (!causal || key_position <= query_position) {
            float similarity = 0.0f;
            const int key_offset = key_value_offset + key_position * head_dimension;
            for (int feature = 0; feature < head_dimension; ++feature) {
                similarity += queries[query_offset + feature] * keys[key_offset + feature];
            }
            score_buffer[key_position] = similarity * scale;
        } else {
            score_buffer[key_position] = -INFINITY;
        }
    }
    if (thread_index >= sequence_length) {
        score_buffer[thread_index] = -INFINITY;
    }
    reduction_buffer[thread_index] = score_buffer[thread_index];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            reduction_buffer[thread_index] = fmaxf(reduction_buffer[thread_index], reduction_buffer[thread_index + stride]);
        }
        __syncthreads();
    }
    const float row_maximum = reduction_buffer[0];

    if (thread_index < sequence_length) {
        reduction_buffer[thread_index] = expf(score_buffer[thread_index] - row_maximum);
    } else {
        reduction_buffer[thread_index] = 0.0f;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            reduction_buffer[thread_index] += reduction_buffer[thread_index + stride];
        }
        __syncthreads();
    }
    const float reciprocal_partition = 1.0f / reduction_buffer[0];
    if (thread_index < sequence_length) {
        score_buffer[thread_index] = expf(score_buffer[thread_index] - row_maximum) * reciprocal_partition;
    }
    __syncthreads();

    const int output_offset = query_offset;
    for (int feature = thread_index; feature < head_dimension; feature += blockDim.x) {
        float weighted_value = 0.0f;
        for (int key_position = 0; key_position < sequence_length; ++key_position) {
            const float normalized_score = score_buffer[key_position];
            weighted_value += normalized_score * values[key_value_offset + key_position * head_dimension + feature];
        }
        output[output_offset + feature] = weighted_value;
    }
}

} 

torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    double scale,
    bool causal) {
    const auto batch_size = queries.size(0);
    const auto head_count = queries.size(1);
    const auto sequence_length = queries.size(2);
    const auto head_dimension = queries.size(3);
    auto output = torch::empty_like(queries);
    int threads = 1;
    while (threads < sequence_length || threads < head_dimension) {
        threads <<= 1;
    }
    threads = threads > 1024 ? 1024 : threads;
    const dim3 grid(sequence_length, batch_size * head_count);
    const auto shared_bytes = 2 * threads * sizeof(float);
    scaled_dot_product_attention_kernel<<<grid, threads, shared_bytes, c10::cuda::getCurrentCUDAStream(queries.device().index()).stream()>>>(
        queries.data_ptr<float>(), keys.data_ptr<float>(), values.data_ptr<float>(), output.data_ptr<float>(),
        sequence_length, head_dimension, static_cast<float>(scale), causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
