/***********************************************************************
 *  sample.cu  –  Euler ancestral sampler for the BF16 EDM U-Net
 **********************************************************************/
#include "diffusion/DiffusionLoader.h"
#include "diffusion/DiffusionConfig.h"
#include "diffusion/DiffusionUNet.cuh"
#include "diffusion/DiffusionEDMPrecond.cuh"
#include "diffusion/DiffusionHelper.cuh"
#include "CudaBuffer.cuh"
#include "ErrorCheck.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>              // std::clamp

/* ───── tiny PNG writer ───── */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

using bf16 = __nv_bfloat16;
using DC = DiffusionConfig;

/* ---------- cuRAND error helper ---------------------------------- */
inline void curandAssert(curandStatus_t s, const char* f, int l)
{
    if (s != CURAND_STATUS_SUCCESS) {
        std::cerr << "cuRAND Error " << int(s) << "  at " << f << ':' << l << '\n';
        std::exit(1);
    }
}
#define checkCurand(ans)  curandAssert((ans), __FILE__, __LINE__)

/* ---------- kernels ---------------------------------------------- */
__global__ void diff_div_kernel(const bf16* x, const bf16* den,
                                float inv_sigma, bf16* out, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = (__bfloat162float(x[i]) - __bfloat162float(den[i])) * inv_sigma;
        out[i]  = __float2bfloat16(v);
    }
}
__global__ void add_scaled_kernel(const bf16* x, const bf16* d,
                                  float step, bf16* out, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __bfloat162float(x[i]) + step * __bfloat162float(d[i]);
        out[i]  = __float2bfloat16(v);
    }
}

/* ---------- σ-schedule ------------------------------------------- */
static std::vector<float> schedule(int N, float s_min, float s_max, int rho)
{
    std::vector<float> t(N + 1);
    float a = std::pow(s_max, 1.f / rho);
    float b = std::pow(s_min, 1.f / rho);
    for (int i = 0; i < N; ++i) {
        float f = float(i) / float(N - 1);
        t[i] = std::pow(a + f * (b - a), rho);
    }
    t[N] = 0.f;         // t_N = 0
    return t;
}

/* ---------- PNG helper (grayscale) --------------------------------*/
static void save_batch_png(const std::vector<bf16>& h,
                           int B, int H, int W, int C = 1)
{
    for (int b = 0; b < B; ++b) {
        std::vector<unsigned char> img(H * W * C);
        const bf16* src = h.data() + size_t(b) * H * W * C;
        for (int i = 0; i < H * W * C; ++i) {
            float v = __bfloat162float(src[i]);      // assume   v ∈ [-1,1]
            v = std::clamp((v + 1.f) * 0.5f, 0.f, 1.f);
            img[i] = static_cast<unsigned char>(v * 255.f + 0.5f);
        }
        std::string fname = "sample_" + std::to_string(b) + ".png";
        stbi_write_png(fname.c_str(), W, H, C, img.data(), W * C);
        std::cout << "  ↳ wrote " << fname << '\n';
    }
}

/* ---------- main sampler ----------------------------------------- */
int main()
{
    /* 1. constants & shapes -------------------------------------- */
    constexpr int   B   = DC::batch_size_infer;
    constexpr int   C   = DC::in_channels;
    constexpr int   H   = DC::img_resolution;
    constexpr int   W   = DC::img_resolution;
    constexpr int   N   = DC::num_steps;
    constexpr size_t PER = size_t(C) * H * W;
    const    size_t TOTAL = PER * B;
    const    size_t BYTES = TOTAL * sizeof(bf16);

    std::cout << "Euler sampler  B=" << B << "  " << H << 'x' << W
              << "  steps=" << N << '\n';

    /* 2. load weights & models ----------------------------------- */
    DiffusionLoader loader;
    DiffusionWeights Wt = loader.load_diffusion_weights(loader.get_model_dir());

    UNetBF16        net(H, DC::t_emb_dim);
    net.load_weights(Wt);
    EDMPrecondBF16  edm(&net, DC::sigma_data);

    /* 3. GPU buffers --------------------------------------------- */
    CudaBuffer x_buf  (BYTES);
    CudaBuffer den_buf(BYTES);
    CudaBuffer d_buf  (BYTES);
    CudaBuffer sigma_d(B * sizeof(float));

    /* 4. x₀ ← N(0, σ_max²) --------------------------------------- */
    {
        CudaBuffer tmp(TOTAL * sizeof(float));
        curandGenerator_t g{};
        checkCurand(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT));
        checkCurand(curandSetPseudoRandomGeneratorSeed(g, 2025ULL));
        checkCurand(curandGenerateNormal(g,
            static_cast<float*>(tmp.data), TOTAL, 0.f, DC::sigma_max));
        dm::fp32_to_bf16(static_cast<float*>(tmp.data),
                         static_cast<bf16*>(x_buf.data),
                         TOTAL, 0);
        curandDestroyGenerator(g);
    }

    /* 5. schedule ------------------------------------------------ */
    std::vector<float> t = schedule(N, DC::sigma_min, DC::sigma_max, DC::rho);

    std::vector<float> h_sigma(B);
    const int TPB = 256;
    const dim3 BLK((TOTAL + TPB - 1) / TPB);

    cudaStream_t st{};  checkCuda(cudaStreamCreate(&st));

    /* 6. Euler loop --------------------------------------------- */
    for (int k = 0; k < N; ++k) {
        float s_k   = t[k];
        float s_nxt = t[k + 1];
        float inv_s = 1.f / s_k;
        float d_s   = s_nxt - s_k;

        std::fill(h_sigma.begin(), h_sigma.end(), s_k);
        checkCuda(cudaMemcpyAsync(sigma_d.data, h_sigma.data(),
                                  B * sizeof(float),
                                  cudaMemcpyHostToDevice, st));

        edm.forward(static_cast<bf16*>(x_buf.data),
                    static_cast<float*>(sigma_d.data),
                    B, C, H, W,
                    static_cast<bf16*>(den_buf.data),
                    st, true);

        diff_div_kernel<<<BLK, TPB, 0, st>>>(static_cast<bf16*>(x_buf.data),
                                             static_cast<bf16*>(den_buf.data),
                                             inv_s,
                                             static_cast<bf16*>(d_buf.data),
                                             TOTAL);

        add_scaled_kernel<<<BLK, TPB, 0, st>>>(static_cast<bf16*>(x_buf.data),
                                               static_cast<bf16*>(d_buf.data),
                                               d_s,
                                               static_cast<bf16*>(x_buf.data),
                                               TOTAL);

        checkCuda(cudaStreamSynchronize(st));
        std::cout << "step " << k + 1 << '/' << N
                  << "   σ=" << s_k << " → " << s_nxt << '\n';
    }

    /* 7. copy to host & save PNGs -------------------------------- */
    std::vector<bf16> h_out(TOTAL);
    checkCuda(cudaMemcpy(h_out.data(), x_buf.data, BYTES, cudaMemcpyDeviceToHost));

    std::cout << "saving PNGs …\n";
    save_batch_png(h_out, B, H, W);

    checkCuda(cudaStreamDestroy(st));
    return 0;
}


// // sample.cu  –  Euler ancestral sampler for the BF16 EDM U-Net
// // ------------------------------------------------------------------
// #include "diffusion/DiffusionLoader.h"
// #include "diffusion/DiffusionConfig.h"
// #include "diffusion/DiffusionUNet.cuh"
// #include "diffusion/DiffusionEDMPrecond.cuh"
// #include "diffusion/DiffusionHelper.cuh"     // fp32_to_bf16
// #include "ErrorCheck.h"

// #include <cuda_runtime.h>
// #include <curand.h>

// #include <cstdlib>

// #include <vector>
// #include <iostream>
// #include <cmath>

// using bf16 = __nv_bfloat16;
// using DC = DiffusionConfig;

// // ------------------------------------------------------------------
// // small element-wise helpers
// // ------------------------------------------------------------------
// __global__ void diff_div_kernel(const bf16* x,
//                                 const bf16* den,
//                                 float inv_sigma,
//                                 bf16* out,
//                                 size_t n)
// {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) {
//         float v = (__bfloat162float(x[i]) - __bfloat162float(den[i])) * inv_sigma;
//         out[i]  = __float2bfloat16(v);
//     }
// }

// __global__ void add_scaled_kernel(const bf16* x,
//                                   const bf16* d,
//                                   float step,
//                                   bf16* out,
//                                   size_t n)
// {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) {
//         float v = __bfloat162float(x[i]) + step * __bfloat162float(d[i]);
//         out[i]  = __float2bfloat16(v);
//     }
// }

// // ------------------------------------------------------------------
// // build the EDM time-step schedule
// // ------------------------------------------------------------------
// static std::vector<float> build_schedule(int N,
//                                          float sigma_min,
//                                          float sigma_max,
//                                          int   rho)
// {
//     std::vector<float> t(N + 1);
//     float s_min_r = std::pow(sigma_min, 1.f / rho);
//     float s_max_r = std::pow(sigma_max, 1.f / rho);
//     for (int i = 0; i < N; ++i) {
//         float frac = float(i) / float(N - 1);
//         t[i] = std::pow(s_max_r + frac * (s_min_r - s_max_r), rho);
//     }
//     t[N] = 0.f;
//     return t;
// }

// inline void curandAssert(curandStatus_t stat, const char* file, int line)
// {
//     if (stat != CURAND_STATUS_SUCCESS) {
//         std::cerr << "cuRAND Error: " << static_cast<int>(stat)
//                   << "  " << file << ":" << line << '\n';
//         std::exit(1);
//     }
// }
// #define checkCurand(ans) { curandAssert((ans), __FILE__, __LINE__); }

// // ------------------------------------------------------------------
// // Euler sampler
// // ------------------------------------------------------------------
// void run_sampler()
// {
//     /* -------- constants from DiffusionConfig -------------------- */
//     constexpr int   B   = DC::batch_size_infer;
//     constexpr int   C   = DC::in_channels;
//     constexpr int   H   = DC::img_resolution;
//     constexpr int   W   = DC::img_resolution;

//     constexpr int   N_STEPS = DC::num_steps;
//     constexpr float SIG_MIN = DC::sigma_min;
//     constexpr float SIG_MAX = DC::sigma_max;
//     constexpr int   RHO     = DC::rho;

//     const size_t PER_IMG = size_t(C) * H * W;
//     const size_t TOTAL   = PER_IMG * B;
//     const size_t BYTES   = TOTAL * sizeof(bf16);

//     /* -------- 1)  Load weights ---------------------------------- */
//     DiffusionLoader loader;
//     DiffusionWeights weights = loader.load_diffusion_weights(loader.get_model_dir());

//     UNetBF16          net(H, DC::t_emb_dim);
//     net.load_weights(weights);
//     EDMPrecondBF16    edm(&net, DC::sigma_data);

//     /* -------- 2)  Allocate GPU buffers -------------------------- */
//     CudaBuffer x_buf   (BYTES);   // carries x through the loop
//     CudaBuffer den_buf (BYTES);   // denoised
//     CudaBuffer d_buf   (BYTES);   // derivative
//     CudaBuffer sigma_d (B * sizeof(float));

//     /* -------- 3)  Initialise x₀ ∼ N(0, σ_max²) ------------------ */
//     {
//         CudaBuffer tmp_f32(TOTAL * sizeof(float));
//         curandGenerator_t gen;
//         checkCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
//         checkCurand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
//         checkCurand(curandGenerateNormal(gen,
//                 static_cast<float*>(tmp_f32.data), TOTAL,
//                 0.f, SIG_MAX));
//         dm::fp32_to_bf16(static_cast<float*>(tmp_f32.data),
//                          static_cast<bf16*>(x_buf.data),
//                          TOTAL, 0);
//         curandDestroyGenerator(gen);
//     }

//     /* -------- 4)  Prepare schedule ------------------------------ */
//     std::vector<float> t = build_schedule(N_STEPS, SIG_MIN, SIG_MAX, RHO);

//     /* -------- 5)  Euler loop ------------------------------------ */
//     std::vector<float> h_sigma(B);
//     const int TPB = 256;
//     const dim3 blocks((TOTAL + TPB - 1) / TPB);

//     cudaStream_t stream;  checkCuda(cudaStreamCreate(&stream));

//     for (int k = 0; k < N_STEPS; ++k)
//     {
//         float sigma_k    = t[k];
//         float sigma_next = t[k + 1];
//         float inv_sigma  = 1.f / sigma_k;
//         float d_sigma    = sigma_next - sigma_k;

//         std::fill(h_sigma.begin(), h_sigma.end(), sigma_k);
//         checkCuda(cudaMemcpyAsync(sigma_d.data, h_sigma.data(),
//                                   B * sizeof(float),
//                                   cudaMemcpyHostToDevice, stream));

//         // denoised = Dθ(x, σ)
//         edm.forward(static_cast<bf16*>(x_buf.data),
//                     static_cast<float*>(sigma_d.data),
//                     B, C, H, W,
//                     static_cast<bf16*>(den_buf.data),
//                     stream, /*sigma_on_device=*/true);

//         // d = (x - den) / σ
//         diff_div_kernel<<<blocks, TPB, 0, stream>>>(
//             static_cast<bf16*>(x_buf.data),
//             static_cast<bf16*>(den_buf.data),
//             inv_sigma,
//             static_cast<bf16*>(d_buf.data),
//             TOTAL);

//         // x ← x + Δσ · d
//         add_scaled_kernel<<<blocks, TPB, 0, stream>>>(
//             static_cast<bf16*>(x_buf.data),
//             static_cast<bf16*>(d_buf.data),
//             d_sigma,
//             static_cast<bf16*>(x_buf.data),
//             TOTAL);

//         checkCuda(cudaStreamSynchronize(stream));
//         std::cout << "step " << k + 1 << "/" << N_STEPS
//                   << "   σ=" << sigma_k << " → " << sigma_next << '\n';
//     }

//     /* -------- 6)  Dump first few pixels ------------------------- */
//     std::vector<bf16> h_out(TOTAL);
//     checkCuda(cudaMemcpy(h_out.data(), x_buf.data, BYTES, cudaMemcpyDeviceToHost));

//     std::cout << "[Sample Output (first 16)]: ";
//     for (int i = 0; i < std::min<size_t>(16, TOTAL); ++i)
//         std::cout << __bfloat162float(h_out[i]) << ' ';
//     std::cout << '\n';

//     checkCuda(cudaStreamDestroy(stream));
// }

// // ------------------------------------------------------------------
// int main()
// {
//     run_sampler();
//     return 0;
// }
