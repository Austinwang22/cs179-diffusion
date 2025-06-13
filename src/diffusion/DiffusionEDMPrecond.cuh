#pragma once
// DiffusionEDMPrecond.cuh
// ────────────────────────────────────────────────────────────────────────────
// Implements the EDM wrapper
//
//   Dθ(x;σ) = c_skip · x + c_out · Fθ(c_in · x, log σ)
//
// where Fθ is the backbone UNet (BF16).  Everything runs in-place on CUDA.
//
// Dependencies:  DiffusionConfig.h, DiffusionUNet.cuh, DiffusionHelper.cuh
// ────────────────────────────────────────────────────────────────────────────
#include "DiffusionConfig.h"
#include "DiffusionUNet.cuh"
#include "DiffusionHelper.cuh"
#include <cstring>  // for std::memcpy

namespace dm {

/* ------------------------------------------------------------------------- */
/* small device helpers                                                      */
/* ------------------------------------------------------------------------- */
__global__ static void k_compute_coeffs(const float* sigma,
                                        float* c_skip,
                                        float* c_out,
                                        float* c_in,
                                        const float sigma_data,
                                        int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    float s   = sigma[i];
    float sd2 = sigma_data * sigma_data;
    float s2  = s * s;

    c_skip[i] = sd2 / (s2 + sd2);
    c_out [i] = s * sigma_data / sqrtf(s2 + sd2);
    c_in  [i] = 1.0f / sqrtf(s2 + sd2);
}

__global__ static void k_scale_img(const __nv_bfloat16* x,
                                   const float* scale,   // [B]
                                   __nv_bfloat16* y,
                                   size_t perImg,
                                   size_t nTotal)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTotal) return;
    size_t b   = idx / perImg;
    float  v   = __bfloat162float(x[idx]) * scale[b];
    y[idx] = __float2bfloat16(v);
}

__global__ static void k_blend(const __nv_bfloat16* x,     // original
                               const __nv_bfloat16* raw,   // UNet output
                               const float* c_skip,
                               const float* c_out,
                               __nv_bfloat16* y,           // dst (may alias x)
                               size_t perImg,
                               size_t nTotal)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTotal) return;
    size_t b = idx / perImg;

    float v = c_skip[b] * __bfloat162float(x[idx]) +
              c_out [b] * __bfloat162float(raw[idx]);
    y[idx] = __float2bfloat16(v);
}

/* ------------------------------------------------------------------------- */
/* EDM wrapper class                                                         */
/* ------------------------------------------------------------------------- */
class EDMPrecondBF16 {
public:
    explicit EDMPrecondBF16(std::shared_ptr<UNetBF16> backbone,
                            float sigma_data = 0.5f)
        : net(std::move(backbone)), sigma_data_(sigma_data) {}

    /**
     * x_noisy   – [B,C,H,W]  (device BF16), **over-written** with the denoised
     * sigma_h   – [B]        (host FP32 noise levels)
     * t_h       – [B]        (host int32 timestep,   existing UNet interface)
     * B         – batch size
     * stream    – CUDA stream to operate on
     */
    void forward(__nv_bfloat16* x_noisy,
                 const float*  sigma_h,
                 const int32_t* t_h,
                 int B,
                 cudaStream_t stream = 0)
    {
        using DC = DiffusionConfig;

        /* ------------ constants & shapes ---------------------------------- */
        constexpr size_t PER_IMG =
            size_t(DC::in_channels) * DC::img_resolution * DC::img_resolution;
        const size_t N_TOTAL = size_t(B) * PER_IMG;
        const size_t BYTES_IMG = N_TOTAL * sizeof(__nv_bfloat16);

        /* ------------ 1. copy sigma + compute coefficients ---------------- */
        CudaBuffer  d_sigma (sizeof(float) * B);
        CudaBuffer  d_cskip (sizeof(float) * B);
        CudaBuffer  d_cout  (sizeof(float) * B);
        CudaBuffer  d_cin   (sizeof(float) * B);

        checkCuda(cudaMemcpyAsync(d_sigma.data, sigma_h,
                                  sizeof(float) * B,
                                  cudaMemcpyHostToDevice, stream));

        // Debug: Print sigma values
        // std::vector<float> h_sigma(B);
        // std::memcpy(h_sigma.data(), sigma_h, sizeof(float) * B);  // Host to host copy
        // std::cout << "Sigma values: ";
        // for (int i = 0; i < B; ++i) {
        //     std::cout << h_sigma[i] << " ";
        // }
        // std::cout << std::endl;

        k_compute_coeffs<<<(B + 255) / 256, 256, 0, stream>>>(
            static_cast<float*>(d_sigma.data),
            static_cast<float*>(d_cskip.data),
            static_cast<float*>(d_cout.data),
            static_cast<float*>(d_cin.data),
            sigma_data_, B);

        // Debug: Print coefficients
        // std::vector<float> h_coeffs(B);
        // checkCuda(cudaMemcpy(h_coeffs.data(), d_cskip.data, sizeof(float) * B, cudaMemcpyDeviceToHost));
        // std::cout << "c_skip values: ";
        // for (int i = 0; i < B; ++i) {
        //     std::cout << h_coeffs[i] << " ";
        // }
        // std::cout << std::endl;

        // checkCuda(cudaMemcpy(h_coeffs.data(), d_cout.data, sizeof(float) * B, cudaMemcpyDeviceToHost));
        // std::cout << "c_out values: ";
        // for (int i = 0; i < B; ++i) {
        //     std::cout << h_coeffs[i] << " ";
        // }
        // std::cout << std::endl;

        // checkCuda(cudaMemcpy(h_coeffs.data(), d_cin.data, sizeof(float) * B, cudaMemcpyDeviceToHost));
        // std::cout << "c_in values: ";
        // for (int i = 0; i < B; ++i) {
        //     std::cout << h_coeffs[i] << " ";
        // }
        // std::cout << std::endl;

        /* ------------ 2. x_input = c_in · x_noisy ------------------------- */
        CudaBuffer d_xin(BYTES_IMG);

        // Debug: Print input values
        // std::vector<__nv_bfloat16> h_input(16);
        // checkCuda(cudaMemcpy(h_input.data(), x_noisy, 16 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        // std::cout << "Input values: ";
        // for (int i = 0; i < 16; ++i) {
        //     std::cout << __bfloat162float(h_input[i]) << " ";
        // }
        // std::cout << std::endl;

        k_scale_img<<<(N_TOTAL + 255) / 256, 256, 0, stream>>>(
            x_noisy,
            static_cast<float*>(d_cin.data),
            static_cast<__nv_bfloat16*>(d_xin.data),
            PER_IMG, N_TOTAL);

        // Debug: Print scaled input
        // checkCuda(cudaMemcpy(h_input.data(), d_xin.data, 16 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        // std::cout << "Scaled input values: ";
        // for (int i = 0; i < 16; ++i) {
        //     std::cout << __bfloat162float(h_input[i]) << " ";
        // }
        // std::cout << std::endl;

        /* ------------ 3. UNet forward  (in-place on d_xin) ---------------- */
        __nv_bfloat16* raw_output = net->forward(static_cast<__nv_bfloat16*>(d_xin.data),
                     t_h, B);

        // Debug: Print UNet output
        // checkCuda(cudaMemcpy(h_input.data(), raw_output, 16 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        // std::cout << "UNet output values: ";
        // for (int i = 0; i < 16; ++i) {
        //     std::cout << __bfloat162float(h_input[i]) << " ";
        // }
        // std::cout << std::endl;

        /* ------------ 4. blend   y = c_skip·x + c_out·raw ----------------- */
        k_blend<<<(N_TOTAL + 255) / 256, 256, 0, stream>>>(
            x_noisy,
            raw_output,
            static_cast<float*>(d_cskip.data),
            static_cast<float*>(d_cout.data),
            x_noisy,          // write back in-place
            PER_IMG, N_TOTAL);

        // Debug: Print final output
        // checkCuda(cudaMemcpy(h_input.data(), x_noisy, 16 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        // std::cout << "Final output values: ";
        // for (int i = 0; i < 16; ++i) {
        //     std::cout << __bfloat162float(h_input[i]) << " ";
        // }
        // std::cout << std::endl;

        cudaGetLastError();   // surface any async errors early
    }

private:
    std::shared_ptr<UNetBF16> net;
    float sigma_data_;
};

} // namespace dm
