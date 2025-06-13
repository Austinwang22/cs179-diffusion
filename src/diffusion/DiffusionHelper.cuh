#pragma once
// DiffusionHelper.cuh
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

#include "DiffusionKernels.cuh"

namespace dm {
// ---------------------------------------------------------------------------
// Global cuBLAS / cuDNN handles (singleton)
// ---------------------------------------------------------------------------
struct Handles {
    cublasHandle_t blas;
    cudnnHandle_t  dnn;

    Handles() {
        // checkCuda(cublasCreate(&blas));
        // checkCuda(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH));
        // checkCuda(cudnnCreate(&dnn));
        cublasCreate(&blas);
        cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH);
        cudnnCreate(&dnn);
    }
    ~Handles() { cublasDestroy(blas); cudnnDestroy(dnn); }
    static Handles &get() { static Handles h; return h; }
};

// // -----------------------------------------------------------------------------
// // ensure_size: grow-or-keep for std::shared_ptr<CudaBuffer>
// // -----------------------------------------------------------------------------
// inline void ensure_size(std::shared_ptr<CudaBuffer>& buf, size_t bytes) {
//     if (bytes == 0) return;
//     if (!buf || buf->size < bytes)
//         buf = std::make_shared<CudaBuffer>(bytes);
// }

// // -----------------------------------------------------------------------------
// // ensure_size: grow-in-place for a value CudaBuffer
// // -----------------------------------------------------------------------------
// inline void ensure_size(CudaBuffer& buf, size_t bytes) {
//     if (bytes == 0 || buf.size >= bytes && buf.data) return;
//     void* new_ptr = nullptr;
//     checkCuda(cudaMallocManaged(&new_ptr, bytes));
//     if (buf.data) checkCuda(cudaFree(buf.data));
//     buf.data = new_ptr;
//     buf.size = bytes;
// }

// -----------------------------------------------------------------------------
// bytes: shorthand for buf.size
// -----------------------------------------------------------------------------
inline size_t bytes(const CudaBuffer &buf) {
    return buf.size;
}

static void dump(const char* tag,
                 const __nv_bfloat16* d, size_t n, cudaStream_t s = 0)
{
    std::vector<__nv_bfloat16> h(n);
    cudaMemcpyAsync(h.data(), d, n*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    std::cerr << tag << " : ";
    for (size_t i=0;i<n;++i) std::cerr << __bfloat162float(h[i]) << ' ';
    std::cerr << '\n';
}

static void dump_chw(const char* tag,
                     const __nv_bfloat16* d, int C,int H,int W, cudaStream_t s=0) { 
    dump(tag, d, std::min<size_t>(C*H*W,10), s); 
}

inline void fp32_to_bf16(const float *src, __nv_bfloat16 *dst,
                         size_t n, cudaStream_t s)
{
    constexpr int TPB = 256;
    int blocks = int((n + TPB - 1) / TPB);
    kernels::fp32_to_bf16_kernel<<<blocks, TPB, 0, s>>>(src, dst, n);
}

// ---------------------------------------------------------------------------
// Fused element‑wise kernels (SiLU, add‑time, upsample, scaling, blend)
// ---------------------------------------------------------------------------

inline void relu_inplace(__nv_bfloat16 *x, size_t n, cudaStream_t s)
{
    constexpr int TPB = 256;
    size_t blocks = (n + TPB - 1) / TPB;
    kernels::relu_bf16_kernel<<<blocks, TPB, 0, s>>>(x, n);
}

// inline void relu_inplace32(float *x, size_t n, cudaStream_t s)
// {
//     constexpr int TPB = 256;
//     size_t blocks = (n + TPB - 1) / TPB;
//     kernels::relu_fp32_kernel<<<blocks, TPB, 0, s>>>(x, n);
// }

inline void silu_inplace(__nv_bfloat16* x, size_t n, cudaStream_t s) {
    kernels::silu_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n);
}

inline void silu_inplace32(float *x, size_t n, cudaStream_t s)
{
    int tpb = 256; size_t blk = (n + tpb - 1) / tpb;
    kernels::silu_fp32_kernel<<<blk, tpb, 0, s>>>(x, n);
}

// inline void add_time_bias(__nv_bfloat16* y, const __nv_bfloat16* t, int B, int C, int H, int W, cudaStream_t s) {
//     kernels::add_time_bias_kernel<<<(size_t(B) * C * H * W + 255) / 256, 256, 0, s>>>(y, t, B, C, H * W);
// }

// inline void add_time_bias(__nv_bfloat16 *y, const __nv_bfloat16 *bias32, int B, int C, int H, int W, cudaStream_t s)
// {
//     constexpr int TPB = 256;
//     size_t blocks = (size_t(B) * C * H * W + TPB - 1) / TPB;
//     kernels::add_time_bias_kernel<<<blocks, TPB, 0, s>>>(y, bias32, B, C, H, W);
// }

inline void add_bias_conv(__nv_bfloat16 *y, const __nv_bfloat16 *bias32, int B, int C, int H, int W, cudaStream_t s)
{
    // std::cerr << "B: " << B;
    // std::cerr << " C: " << C;
    // std::cerr << " H: " << H;
    // std::cerr << " W: " << W;
    // std::cerr << "\nthe maximum index is: " << B * C * H * W << "\n";

    // constexpr int TPB = 256;
    // size_t blocks = (size_t(B) * C * H * W + TPB - 1) / TPB;
    kernels::add_bias_conv_kernel<<<256, 256, 0, s>>>(y, bias32, B, C, H, W);
}

inline void add_time_bias(__nv_bfloat16 *y, const __nv_bfloat16 *bias32, int B, int C, int H, int W, cudaStream_t s)
{
    // constexpr int TPB = 256;
    // size_t blocks = (size_t(B) * C * H * W + TPB - 1) / TPB;
    kernels::add_time_bias_kernel<<<256, 256, 0, s>>>(y, bias32, B, C, H, W);
}

// inline void add_time_bias32(__nv_bfloat16 *y, const float *bias32, int B, int C, int H, int W, cudaStream_t s)
// {
//     constexpr int TPB = 256;
//     size_t blocks = (size_t(B) * C * H * W + TPB - 1) / TPB;
//     kernels::add_time_bias32_kernel<<<blocks, TPB, 0, s>>>(y, bias32, B, C, H*W);
// }

// inline void add_time_bias(__nv_bfloat16 *y, const __nv_bfloat16 *bias32, int B, int C, int H, int W, cudaStream_t s)
// {
//     constexpr int TPB = 256;
//     size_t blocks = (size_t(B) * C * H * W + TPB - 1) / TPB;
//     kernels::add_time_bias_kernel<<<blocks, TPB, 0, s>>>(y, bias32, B, C, H, W);
// }

inline void upsample2x(const __nv_bfloat16* src, int B, int C, int H, int W, __nv_bfloat16* dst, cudaStream_t s) {
    size_t elems = size_t(B) * C * H * W * 4ULL;
    kernels::upsample_kernel<<<(elems + 255) / 256, 256, 0, s>>>(src, dst, C, H, W);
}

}