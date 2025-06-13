#include "diffusion/DiffusionLoader.cu"

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
    // DiffusionWeights weights = loader.load_diffusion_weights(weight_path);

    // ------------------------------
    // Initialize model and load weights
    // ------------------------------
    // UNetBF16 model(DC::img_resolution, DC::t_emb_dim);
    // model.load_weights(weights);
    std::shared_ptr<UNetBF16> model = loader.load_diffusion_weights(weight_path);

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
    model->forward(static_cast<__nv_bfloat16 *>(input.data), h_tstamp, B, stream);
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
