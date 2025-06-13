#pragma once
// DiffusionKernels.cuh
// --------------------------------------------------------------
// Common utilities, global CUDA handles, fused element‑wise kernels, and basic
// helper macros shared by all other diffusion headers.
// --------------------------------------------------------------
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <cudnn.h>
#include <cuda_bf16.h>

#include <memory>
#include <vector>
#include <cmath>

#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"

namespace kernels {

__global__ void silu_kernel(__nv_bfloat16* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float v = __bfloat162float(x[i]); v = v / (1.f + expf(-v)); x[i] = __float2bfloat16(v);
}

__global__ void silu_fp32_kernel(float *x, size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.f + expf(-v));      // SiLU
    }
}

__global__ void relu_bf16_kernel(__nv_bfloat16 *x, size_t n)
{
    // plain 1-D indexing
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                 // <- **stop out-of-range threads**

    float v = __bfloat162float(x[i]);
    if (v < 0.f) v = 0.f;
    x[i] = __float2bfloat16(v);
}

__global__ void add_bias_fp32_kernel(float *y, const __nv_bfloat16 *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += __bfloat162float(b[i]);
}

__global__ void add_time_bias_kernel(__nv_bfloat16 *y,
                                   const float   *bias32,   //  FP-32!
                                   int B, int C, int H, int W)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t N   = size_t(B) * C * H * W;
    if (idx >= N) return;

    // Calculate batch and channel indices correctly
    size_t hw = size_t(H) * W;
    size_t chw = size_t(C) * hw;
    int b = int(idx / chw);  // Batch index
    int c = int((idx % chw) / hw);  // Channel index

    float v = __bfloat162float(y[idx]) + bias32[b * C + c];
    y[idx]  = __float2bfloat16(v);       // single rounding
}

__global__ void upsample_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst, int C, int H, int W) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t HW  = size_t(H) * W;
    size_t N   = size_t(C) * HW * 4ULL;
    if (idx >= N) return;
    int c  = idx / (HW * 4);
    size_t pos = idx % (HW * 4);
    int oh = pos / (W * 2);
    int ow = (pos / 2) % W;
    size_t src_idx = c * HW + oh * W + ow;
    dst[idx] = src[src_idx];
}

__global__ void fp32_to_bf16_kernel(const float *src,
                                    __nv_bfloat16 *dst,
                                    size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// linearlayer: add bias kernel
static __global__ void lin_add_bias(__nv_bfloat16 *y, const __nv_bfloat16 *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n)
        y[i] = __float2bfloat16(__bfloat162float(y[i]) + __bfloat162float(b[i]));
}

// convlayer: add bias kernel
static __global__ void conv_add_bias(__nv_bfloat16 *y, const __nv_bfloat16 *b, int B, int C, int HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size_t(B) * C * HW) return;
    int c = (idx / HW) % C; y[idx] = __float2bfloat16(__bfloat162float(y[idx]) + __bfloat162float(b[c]));
}

// __global__ void scale_kernel(const __nv_bfloat16* x, const float* s, __nv_bfloat16* y, size_t perImg, size_t n) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n) return;
//     size_t b = idx / perImg; y[idx] = __float2bfloat16(__bfloat162float(x[idx]) * s[b]);
// }

// __global__ void blend_kernel(const __nv_bfloat16* x, const __nv_bfloat16* raw,
//                                     const float* cskip, const float* cout,
//                                     __nv_bfloat16* y, size_t perImg, size_t n) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n) return;
//     size_t b = idx / perImg;
//     float v  = cskip[b] * __bfloat162float(x[idx]) + cout[b] * __bfloat162float(raw[idx]);
//     y[idx]   = __float2bfloat16(v);
// }

// // ------------- fused LayerNorm kernel ---------------------------------------
// __global__ __launch_bounds__(256) void ln_bf16_kernel(const __nv_bfloat16 *x,
//                                                      __nv_bfloat16 *y,
//                                                      const float *gamma,
//                                                      const float *beta,
//                                                      int HW, float eps)
// {
//     extern __shared__ float shared[];
//     float *mean_buf = shared;            // [blockDim.x]
//     float *var_buf  = shared + blockDim.x; // [blockDim.x]

//     const __nv_bfloat16 *x_vec = x + blockIdx.x * HW;

//     // mean ------------------------------------------------------------------
//     float m = 0.f;
//     for (int i = threadIdx.x; i < HW; i += blockDim.x)
//         m += __bfloat162float(x_vec[i]);
//     mean_buf[threadIdx.x] = m;
//     __syncthreads();
//     for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
//         if (threadIdx.x < stride)
//             mean_buf[threadIdx.x] += mean_buf[threadIdx.x + stride];
//         __syncthreads();
//     }
//     float mean = mean_buf[0] / HW;

//     // variance --------------------------------------------------------------
//     float v = 0.f;
//     for (int i = threadIdx.x; i < HW; i += blockDim.x) {
//         float diff = __bfloat162float(x_vec[i]) - mean;
//         v += diff * diff;
//     }
//     var_buf[threadIdx.x] = v;
//     __syncthreads();
//     for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
//         if (threadIdx.x < stride)
//             var_buf[threadIdx.x] += var_buf[threadIdx.x + stride];
//         __syncthreads();
//     }
//     float inv_std = rsqrtf(var_buf[0] / HW + eps);

//     // re‑scale --------------------------------------------------------------
//     float g = gamma[blockIdx.x];
//     float b = beta [blockIdx.x];
//     for (int i = threadIdx.x; i < HW; i += blockDim.x) {
//         float norm = (__bfloat162float(x_vec[i]) - mean) * inv_std;
//         y[blockIdx.x * HW + i] = __float2bfloat16(norm * g + b);
//     }
// }

}