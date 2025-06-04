#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"

template<typename input_float_t>
__global__
void
cudabf16MatMulKernel(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, input_float_t *vec, __nv_bfloat16 *out) {
    extern __shared__ __align__(sizeof(float)) char shared[];

    input_float_t *vec_shared = reinterpret_cast<input_float_t *>(shared);
    float *reduction_shared = reinterpret_cast<float *>(vec_shared + k);

    const int32_t thread = threadIdx.x;

    // vector into shared memory 
    int32_t index = thread;
    while (index < k) {
        vec_shared[index] = vec[index];
        
        index += blockDim.x;
    }
    __syncthreads();

    // every block takes a row
    int32_t row = blockIdx.x;
    if (row >= m) return;

    __nv_bfloat16 *row_pointer = mat + row * k;
    float sum = 0.0f;

    index = thread;
    while (index < k) {
        float mat_value = __bfloat162float(row_pointer[index]);
        float vec_value = static_cast<float>(vec_shared[index]);

        sum += mat_value * vec_value;

        index += blockDim.x;
    }
    reduction_shared[thread] = sum;
    __syncthreads();

    for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
        if(thread < s) reduction_shared[thread] += reduction_shared[thread + s];

        __syncthreads();
    }

    if (thread == 0) {
        float final = reduction_shared[0];

        if (bias) {
            final += __bfloat162float(bias[row]);
        }

        out[row] = __float2bfloat16_rn(final);
    }
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    int32_t BLOCK_SIZE = 256;

    cudabf16MatMulKernel<<<m, BLOCK_SIZE, k * sizeof(input_float_t) + BLOCK_SIZE * sizeof(float), stream>>>(
        m, k, mat, bias, vec, out
    );
}

// explicit instantiations
template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
