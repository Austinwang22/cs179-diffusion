// ────────────────────────────────────────────────────────────────────────────
// DiffusionEDMPrecond.cuh
// EDM pre-conditioning wrapper for BF16 U-Net
//
//    Dθ(x,σ) = c_skip · x  +  c_out · Fθ(c_in · x , ln σ)
//
// • Works with the minimal CudaBuffer (cudaMallocManaged / ~free)
// • Element-wise scale + blend kernels are defined here (BF16 arithmetic)
// • Accepts σ either on host or on device
// ────────────────────────────────────────────────────────────────────────────
#pragma once
#include <vector>
#include <cmath>
#include <cstring>     // <- for std::memcpy
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../CudaBuffer.cuh"
#include "DiffusionHelper.cuh"      // ensure_size
#include "DiffusionUNet.cuh"

namespace dm {

using bf16 = __nv_bfloat16;

// ───────────────────────────────── element-wise device helpers ──
namespace edm_kernels {

__global__ void scale_kernel(const bf16* x,
                             const float* s,      // [B]
                             bf16*       y,
                             size_t      per_img, // C·H·W
                             size_t      n_total) // B·C·H·W
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    size_t b = idx / per_img;
    y[idx] = __float2bfloat16(__bfloat162float(x[idx]) * s[b]);
}

__global__ void blend_kernel(const bf16* x,
                             const bf16* raw,
                             const float* c_skip, // [B]
                             const float* c_out,  // [B]
                             bf16*       y,
                             size_t      per_img,
                             size_t      n_total)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    size_t b = idx / per_img;
    float v = c_skip[b] * __bfloat162float(x[idx]) +
              c_out [b] * __bfloat162float(raw[idx]);
    y[idx] = __float2bfloat16(v);
}

} // namespace edm_kernels

// ────────────────────────────────────────────────────────────────
class EDMPrecondBF16 {
public:
    EDMPrecondBF16(UNetBF16* backbone, float sigma_data = 0.5f)
        : unet_(backbone), sigma_data2_(sigma_data * sigma_data) {}

    /**
     *  x_noisy  (device BF16)  : [B,C,H,W]
     *  sigma    (host float[B] or device float[B])
     *  out      (device BF16)  : [B,C,H,W]   (may alias x_noisy)
     *  sigma_on_device = true  if “sigma” already lives on the GPU
     */
    void forward(const bf16*  x_noisy,
                 const float* sigma,
                 int          B, int C, int H, int W,
                 bf16*        out,
                 cudaStream_t stream           = 0,
                 bool         sigma_on_device  = false)
    {
        const size_t per_img = size_t(C) * H * W;
        const size_t n_total = per_img * B;

        // ── 1. copy σ to host (if needed) & compute coeffs ────────────
        std::vector<float> h_sigma(B);
        if (sigma_on_device) {
            cudaMemcpyAsync(h_sigma.data(), sigma, B*sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        } else {
            std::memcpy(h_sigma.data(), sigma, B*sizeof(float));
        }

        std::vector<float> h_c_in(B), h_c_out(B), h_c_skip(B);
        std::vector<int32_t> h_ln_sigma(B);
        for (int i = 0; i < B; ++i) {
            float s   = h_sigma[i];
            float s2  = s * s;
            float denom = std::sqrt(s2 + sigma_data2_);

            h_c_in  [i] = 1.f / denom;
            h_c_out [i] = s * std::sqrt(sigma_data2_) / denom;
            h_c_skip[i] = sigma_data2_ / (s2 + sigma_data2_);
            h_ln_sigma[i] = static_cast<int32_t>(std::log(s)); // matches TimeEmbeddingBF16
        }

        // ── 2.  upload coeffs to device (CudaBuffer with ensure_size) ─
        upload_vec_to_gpu(c_in_buf_ , h_c_in , stream);
        upload_vec_to_gpu(c_out_buf_, h_c_out, stream);
        upload_vec_to_gpu(c_skip_buf_,h_c_skip,stream);

        // ── 3.  x_in = c_in · x_noisy  (into scratch x_in_buf_) ───────
        ensure_size(x_in_buf_, n_total * sizeof(bf16));
        launch_scale(x_noisy,
                     static_cast<float*>(c_in_buf_.data),
                     static_cast<bf16*>(x_in_buf_.data),
                     per_img, n_total, stream);

        // ── 4.  raw = U-Net(c_in·x , ln σ)  (in-place in x_in_buf_) ───
        unet_->forward(static_cast<bf16*>(x_in_buf_.data),
                       h_ln_sigma.data(),  // host int32 array
                       B, stream);

        // ── 5.  blend → out  :  c_skip·x + c_out·raw ──────────────────
        launch_blend(x_noisy,
                     static_cast<bf16*>(x_in_buf_.data),
                     static_cast<float*>(c_skip_buf_.data),
                     static_cast<float*>(c_out_buf_.data),
                     out, per_img, n_total, stream);
    }

private:
    // ── helpers ────────────────────────────────────────────────────
    template<typename Vec>
    void upload_vec_to_gpu(CudaBuffer& buf, const Vec& host, cudaStream_t st)
    {
        ensure_size(buf, host.size() * sizeof(typename Vec::value_type));
        cudaMemcpyAsync(buf.data, host.data(), buf.size,
                        cudaMemcpyHostToDevice, st);
    }

    inline void launch_scale(const bf16* x, const float* s, bf16* y,
                             size_t per_img, size_t n, cudaStream_t st)
    {
        constexpr int TPB = 256;
        edm_kernels::scale_kernel<<<(n + TPB - 1)/TPB, TPB, 0, st>>>(
            x, s, y, per_img, n);
    }

    inline void launch_blend(const bf16* x, const bf16* raw,
                             const float* c_skip, const float* c_out,
                             bf16* y, size_t per_img, size_t n, cudaStream_t st)
    {
        constexpr int TPB = 256;
        edm_kernels::blend_kernel<<<(n + TPB - 1)/TPB, TPB, 0, st>>>(
            x, raw, c_skip, c_out, y, per_img, n);
    }

    // ── members ────────────────────────────────────────────────────
    UNetBF16*   unet_;            // not owned
    const float sigma_data2_;     // σ_data²

    CudaBuffer  c_in_buf_{0}, c_out_buf_{0}, c_skip_buf_{0};
    CudaBuffer  x_in_buf_{0};     // scratch (holds c_in·x and then raw)
};

} // namespace dm
