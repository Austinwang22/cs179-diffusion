#pragma once
// DiffusionCommon.cuh
// --------------------------------------------------------------
// Common utilities, global CUDA handles, fused element‑wise kernels, and basic
// helper macros shared by all other diffusion headers.
// --------------------------------------------------------------
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <cuda_bf16.h>
#include <memory>
#include <vector>
#include <cmath>
#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"

// ---------------- Global cuBLAS / cuDNN handles ----------------- //
namespace dm_common {
struct Handles {
    cublasHandle_t blas{nullptr};
    cudnnHandle_t  dnn{nullptr};
    Handles() {
        checkCuda(cublasCreate(&blas));
        checkCuda(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH));
        checkCuda(cudnnCreate(&dnn));
    }
    ~Handles() {
        cublasDestroy(blas);
        cudnnDestroy(dnn);
    }
    static Handles &instance() { static Handles h; return h; }
};
} // namespace dm_common

// ---------------- bfloat16 helpers ------------------------------ //
__device__ __forceinline__ float 
bf2f(const __nv_bfloat16 &x) {
    return __bfloat162float(x); 
}
__device__ __forceinline__ __nv_bfloat16 
f2bf(float x) {
    return __float2bfloat16(x);
}

// ---------------- fused activation kernels ---------------------- //
__global__ inline void silu_kernel(__nv_bfloat16 *x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;

    float v = bf2f(x[i]); v = v / (1.f + expf(-v)); x[i] = f2bf(v);
}
inline void launch_silu(__nv_bfloat16 *x, size_t n, cudaStream_t s) {
    silu_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n);
}

// add per‑channel + per‑batch bias from time‑embedding
__global__ inline void add_time_bias_kernel(__nv_bfloat16 *y, const __nv_bfloat16 *t, int B, int C, int HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= size_t(B) * C * HW) return;

    int b = idx / (C * HW); 
    int c = (idx / HW) % C;
    y[idx] = f2bf(bf2f(y[idx]) + bf2f(t[b * C + c]));
}
inline void add_time_bias(__nv_bfloat16 *y, const __nv_bfloat16 *t, int B, int C, int H, int W, cudaStream_t s) {
    add_time_bias_kernel<<<(size_t(B) * C * H * W + 255) / 256, 256, 0, s>>>(y, t, B, C, H * W);
}

// nearest‑neighbour 2× upsample (NCHW)
__global__ inline void upsample_kernel(const __nv_bfloat16 *src, __nv_bfloat16 *dst, int C, int H, int W) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t HW = H * W;
    size_t N = C * HW * 4ULL;
    if (idx >= N) return;

    int c = idx / (HW * 4);
    size_t pos = idx % (HW * 4);
    int oh = (pos / (W * 2));
    int ow = (pos / 2) % W;
    size_t src_idx = c * HW + oh * W + ow;

    dst[idx] = src[src_idx];
}
inline void upsample2x(const __nv_bfloat16 *src, int B, int C, int H, int W, __nv_bfloat16 *dst, cudaStream_t s) {
    size_t elems = size_t(B) * C * H * W * 4ULL;

    upsample_kernel<<<(elems + 255) / 256, 256, 0, s>>>(src, dst, C, H, W);
}

// element‑wise scaling (per‑batch scalar) x * s
__global__ inline void scale_kernel(const __nv_bfloat16 *x, const float *s, __nv_bfloat16 *y, size_t per_img, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    size_t b = idx / per_img;
    y[idx] = f2bf(bf2f(x[idx]) * s[b]);
}

// final EDM blend  y = c_skip * x + c_out * raw
__global__ inline void blend_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *raw, const float *cskip, const float *cout, __nv_bfloat16 *y, size_t per_img, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t b = idx / per_img;
    float v = cskip[b] * bf2f(x[idx]) + cout[b] * bf2f(raw[idx]);
    y[idx] = f2bf(v);
}
