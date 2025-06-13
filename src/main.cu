#include "diffusion/DiffusionLoader.h"
#include "diffusion/DiffusionConfig.h"

#include "diffusion/DiffusionUNet.cuh"

#include "ErrorCheck.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

void test_forward_with_weights() {
    using DC = DiffusionConfig;
    const int B = 1;                      // batch size
    const int H = DC::img_resolution;
    const int W = DC::img_resolution;
    const int C = DC::in_channels;       // typically 1
    const int numel = B * C * H * W;
    const size_t bytes = numel * sizeof(__nv_bfloat16);

    // ------------------------------
    // Load weights from safetensors
    // ------------------------------
    DiffusionLoader loader;

    const std::string weight_path = loader.get_model_dir();

    DiffusionWeights weights = loader.load_diffusion_weights(weight_path);

    // ------------------------------
    // Initialize model and load weights
    // ------------------------------
    UNetBF16 model(DC::img_resolution, DC::t_emb_dim);
    model.load_weights(weights);

    // ------------------------------
    // Allocate and fill dummy input
    // ------------------------------
    CudaBuffer input(bytes);
    std::vector<__nv_bfloat16> h_input(numel, __float2bfloat16(1.0f));  // Constant input
    checkCuda(cudaMemcpy(input.data, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // ------------------------------
    // Allocate and fill dummy time embedding
    // ------------------------------
    int32_t h_tstamp[B] = {32};  // arbitrary timestep
    int32_t *d_tstamp;
    checkCuda(cudaMalloc(&d_tstamp, sizeof(int32_t) * B));
    checkCuda(cudaMemcpy(d_tstamp, h_tstamp, sizeof(int32_t) * B, cudaMemcpyHostToDevice));

    // ------------------------------
    // Perform forward pass
    // ------------------------------

    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    model.forward(static_cast<__nv_bfloat16 *>(input.data), h_tstamp, B, stream);
    checkCuda(cudaStreamSynchronize(stream));

    // ------------------------------
    // Dump part of output
    // ------------------------------
    std::vector<__nv_bfloat16> h_out(numel);
    checkCuda(cudaMemcpy(h_out.data(), input.data, bytes, cudaMemcpyDeviceToHost));

    std::cout << "[Forward Output]: ";
    for (int i = 0; i < std::min(16, numel); ++i) {
        std::cout << __bfloat162float(h_out[i]) << " ";
    }
    std::cout << std::endl;

    // ------------------------------
    // Cleanup
    // ------------------------------
    checkCuda(cudaFree(d_tstamp));
    checkCuda(cudaStreamDestroy(stream));
}

int main() {
    test_forward_with_weights();
    return 0;
}


// ======== random weights forward pass ============================ ///
// // main.cpp – toy inference driver for the bf16 diffusion U-Net
// // ------------------------------------------------------------------
// // compile (CUDA 12+):
// // nvcc -std=c++17 -O3 -arch=sm_80 main.cpp -lcudnn -lcublas -lcurand
// // ------------------------------------------------------------------
// #include <cuda_runtime.h>
// #include <curand.h>
// #include <iostream>
// #include <vector>

// #include "diffusion/DiffusionConfig.h"
// #include "diffusion/DiffusionUNet.cuh"

// using namespace dm;

// // ------------------------------------------------------------------
// // quick helper: fill a raw bf16 buffer with U(0,1) noise
// // ------------------------------------------------------------------
// __global__ void rand2bf16(float *src, __nv_bfloat16 *dst, size_t n)
// {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) dst[i] = __float2bfloat16(src[i]);
// }

// void fill_random_bf16(void *dev_ptr, size_t elems, cudaStream_t st)
// {
//     static curandGenerator_t gen = nullptr;
//     if (!gen) curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//     curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

//     CudaBuffer tmp(elems * sizeof(float));
//     curandGenerateUniform(gen, static_cast<float*>(tmp.data), elems);

//     dim3 blk(256);
//     dim3 grd((elems + blk.x - 1) / blk.x);
//     rand2bf16<<<grd, blk, 0, st>>>(static_cast<float*>(tmp.data),
//                                    static_cast<__nv_bfloat16*>(dev_ptr),
//                                    elems);
// }

// // ------------------------------------------------------------------
// // run a single forward pass through the network
// // ------------------------------------------------------------------
// void forward_once(UNetBF16 &net, int B, cudaStream_t st)
// {
//     constexpr int H = DiffusionConfig::img_resolution;
//     constexpr int W = DiffusionConfig::img_resolution;

//     // x_noisy  (BF16)
//     CudaBuffer x_buf(size_t(B) * H * W * sizeof(__nv_bfloat16));
//     fill_random_bf16(x_buf.data, x_buf.size / sizeof(__nv_bfloat16), st);

//     // dummy timesteps (all zeros just for sanity-check)
//     std::vector<int32_t> t_host(B, 0);

//     net.forward(static_cast<__nv_bfloat16*>(x_buf.data), t_host.data(), B, st);
//     cudaStreamSynchronize(st);
// }

// int main()
// {
//     cudaStream_t st{}; cudaStreamCreate(&st);

//     // 1. build the diffusion U-Net
//     UNetBF16 net(28, 128); //resolution, time embedding

//     // 2. forward pass (no weight randomisation necessary for a compile test)
//     forward_once(net, 1, st); // single batch

//     std::cout << "Inference completed (random weights = implicit zeros).\n";

//     cudaStreamDestroy(st);
//     return 0;
// }
