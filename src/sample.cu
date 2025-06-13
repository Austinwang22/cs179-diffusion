// sample.cu ──────────────────────────────────────────────────────────────────
// Build:   nvcc -std=c++17 -O3 -o sample sample.cu -lcublas -lcudadevrt
// Depends: stb_image_write.h   (drop it next to this file)
//          the rest of your src/ tree (CudaBuffer, Diffusion*, …).
// ────────────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cassert>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#include "diffusion/DiffusionLoader.cu"      // loader + UNet
#include "diffusion/DiffusionEDMPrecond.cuh" // EDMPrecondBF16 wrapper

using namespace dm;     // all diffusion helpers live here

/* ========================================================================== */
/* GPU helpers                                                                */
/* ========================================================================== */
__global__ void k_init_rand(curandState *state, uint64_t seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void k_fill_normal(__nv_bfloat16 *dst,
                              curandState *state,
                              float sigma,
                              size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    
    // Add safety check for sigma
    sigma = fmaxf(sigma, 1e-6f);
    
    float rnd = curand_normal(&state[id]) * sigma;
    // Clamp the random values to prevent extreme values
    rnd = fmaxf(fminf(rnd, 100.0f), -100.0f);
    dst[id] = __float2bfloat16(rnd);
}

__global__ void update_kernel(float dt, float t_cur, 
                            __nv_bfloat16 *x_next,      // output buffer
                            const __nv_bfloat16 *x_cur, // current state
                            const __nv_bfloat16 *denoised, // model output
                            size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float x_cur_val = __bfloat162float(x_cur[idx]);
    float denoised_val = __bfloat162float(denoised[idx]);
    
    // Add small epsilon to prevent division by zero
    const float eps = 1e-6f;
    float t_cur_safe = fmaxf(t_cur, eps);
    
    // d_cur = (x_cur - denoised) / t_cur
    float d_cur = (x_cur_val - denoised_val) / t_cur_safe;
    
    // x_next = x_cur + dt * d_cur
    float x_next_val = x_cur_val + dt * d_cur;
    
    // Add value clamping to prevent extreme values
    x_next_val = fmaxf(fminf(x_next_val, 100.0f), -100.0f);
    
    x_next[idx] = __float2bfloat16(x_next_val);
}

/* ========================================================================== */
/* Configuration (matches DiffusionConfig defaults)                           */
/* ========================================================================== */
struct Opt {
    int   img_res   = 28;
    int   batch     = 16;
    int   num_steps = 100;
    float sigma_min = 0.02f;
    float sigma_max = 80.f;
    int   rho       = 7;
};

static std::vector<float> make_time_steps(const Opt &o)
{
    std::vector<float> t(o.num_steps + 1);
    for (int i = 0; i < o.num_steps; ++i) {
        double a = pow(o.sigma_max, 1.0/o.rho);
        double b = pow(o.sigma_min, 1.0/o.rho);
        double val = pow(a + (double)i / (o.num_steps - 1) * (b - a), o.rho);
        // Add safety check
        if (val <= 0.0 || !isfinite(val)) {
            std::cerr << "Warning: Invalid time step generated at index " << i 
                      << ": val=" << val << std::endl;
            val = o.sigma_min;  // Use minimum sigma as fallback
        }
        t[i] = static_cast<float>(val);
    }
    t[o.num_steps] = o.sigma_min;  // Use minimum sigma instead of 0
    return t;
}

/* ========================================================================== */
int main()
{
    Opt opt;  // tweak above if needed
    cudaStream_t stream;  checkCuda(cudaStreamCreate(&stream));

    /* ───────────────────── 1. load model weights ───────────────────────── */
    DiffusionLoader loader;
    auto  unet     = loader.load_diffusion_weights(loader.get_model_dir());
    EDMPrecondBF16 model(unet, DiffusionConfig::sigma_data);

    /* ───────────────────── 2. allocate tensors ─────────────────────────── */
    constexpr int C = DiffusionConfig::in_channels;
    const size_t PER_IMG    = size_t(C) * opt.img_res * opt.img_res;
    const size_t NUM_ELEMS  = size_t(opt.batch) * PER_IMG;
    const size_t BYTES_BF16 = NUM_ELEMS * sizeof(__nv_bfloat16);

    CudaBuffer d_x(BYTES_BF16);  // Main buffer for sampling
    CudaBuffer d_x_cur(BYTES_BF16);  // Temporary buffer for current state

    // host arrays for sigma & timestep (int32)
    std::vector<float>   h_sigma (opt.batch);
    std::vector<int32_t> h_tstep (opt.batch, 0);   // always 0 for UNet interface

    /* curand state for per-element RNG */
    CudaBuffer d_state(NUM_ELEMS * sizeof(curandState));
    k_init_rand<<<(NUM_ELEMS+255)/256, 256, 0, stream>>>(
        static_cast<curandState*>(d_state.data), 1234ULL);

    /* pre-compute time schedule on host */
    std::vector<float> t_schedule = make_time_steps(opt);

    /* ───────────────────── 3. Euler loop ───────────────────────────────── */
    for (int i = 0; i < opt.batch; ++i) h_sigma[i] = opt.sigma_max;

    std::cout << "Starting Euler sampling with " << opt.num_steps << " steps\n";
    std::cout << "Initial sigma: " << opt.sigma_max << "\n";

    // Verify buffer sizes
    if (d_x.size < BYTES_BF16) {
        throw std::runtime_error("Buffer d_x too small");
    }
    if (d_x_cur.size < BYTES_BF16) {
        throw std::runtime_error("Buffer d_x_cur too small");
    }

    // x_next ← N(0, σ_max² I)
    k_fill_normal<<<(NUM_ELEMS+255)/256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(d_x.data),
        static_cast<curandState*>(d_state.data),
        opt.sigma_max, NUM_ELEMS);
    checkCuda(cudaStreamSynchronize(stream));  // Ensure initialization is complete

    for (int step = 0; step < opt.num_steps; ++step) {
        float t_cur  = t_schedule[step];
        float t_next = t_schedule[step + 1];
        
        // Add safety check for time steps
        if (t_cur <= 0.0f || t_next <= 0.0f) {
            std::cerr << "Warning: Invalid time step detected at step " << step 
                      << ": t_cur=" << t_cur << ", t_next=" << t_next << std::endl;
            continue;
        }

        float dt = t_next - t_cur;

        if (step % 100 == 0 || step == opt.num_steps - 1) {
            std::cout << "Step " << std::setw(4) << step 
                      << "/" << opt.num_steps 
                      << " | sigma: " << std::fixed << std::setprecision(6) << t_cur
                      << " | dt: " << std::fixed << std::setprecision(6) << dt
                      << "\n";
        }

        // σ_i for every sample
        std::fill(h_sigma.begin(), h_sigma.end(), t_cur);
        
        // Create a temporary buffer for the current state
        checkCuda(cudaMemcpyAsync(d_x_cur.data, d_x.data, BYTES_BF16, cudaMemcpyDeviceToDevice, stream));
        checkCuda(cudaStreamSynchronize(stream));  // Ensure copy is complete
        
        // Forward pass - get denoised output
        model.forward(static_cast<__nv_bfloat16*>(d_x_cur.data),
                     h_sigma.data(), h_tstep.data(), opt.batch, stream);
        checkCuda(cudaStreamSynchronize(stream));  // Ensure forward pass is complete

        // Compute d_cur = (x_cur - denoised) / t_cur and update x_next
        size_t N = NUM_ELEMS;
        __nv_bfloat16 *x_ptr = static_cast<__nv_bfloat16*>(d_x.data);
        __nv_bfloat16 *x_cur_ptr = static_cast<__nv_bfloat16*>(d_x_cur.data);

        // Update kernel: x_next = x_cur + dt * (x_cur - denoised)/t_cur
        dim3 blk(256);
        dim3 grd((N + blk.x - 1) / blk.x);
        update_kernel<<<grd,blk,0,stream>>>(dt, t_cur, x_ptr, x_cur_ptr, x_cur_ptr, N);
        checkCuda(cudaStreamSynchronize(stream));  // Ensure update is complete
    }
    checkCuda(cudaStreamSynchronize(stream));  // Final synchronization

    /* ───────────────────── 4. dump to PNG files ────────────────────────── */
    std::vector<float> h_out(NUM_ELEMS);
    std::vector<__nv_bfloat16> h_bf(NUM_ELEMS);
    
    // Ensure host buffers are large enough
    if (h_out.size() < NUM_ELEMS || h_bf.size() < NUM_ELEMS) {
        throw std::runtime_error("Host buffers too small");
    }
    
    // BF16 → FP32 on host
    checkCuda(cudaMemcpy(h_bf.data(), d_x.data, BYTES_BF16, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < NUM_ELEMS; ++i)
        h_out[i] = __bfloat162float(h_bf[i]);

    // min-max per image → uint8
    for (int b = 0; b < opt.batch; ++b) {
        const float *img = h_out.data() + b * PER_IMG;

        float vmin =  1e9f, vmax = -1e9f;
        for (size_t p = 0; p < PER_IMG; ++p) {
            vmin = fminf(vmin, img[p]);
            vmax = fmaxf(vmax, img[p]);
        }
        float inv = (vmax > vmin) ? 1.f / (vmax - vmin) : 1.f;

        std::vector<uint8_t> u8(PER_IMG);
        for (size_t p = 0; p < PER_IMG; ++p)
            u8[p] = static_cast<uint8_t>(255.f * (img[p] - vmin) * inv);

        std::ostringstream oss;
        oss << "sample_" << std::setfill('0') << std::setw(3) << b << ".png";
        stbi_write_png(oss.str().c_str(),
                       opt.img_res, opt.img_res,
                       1,               // gray-scale
                       u8.data(),
                       opt.img_res);    // stride
    }

    std::cout << "Saved " << opt.batch << " PNGs.\n";
    return 0;
}
