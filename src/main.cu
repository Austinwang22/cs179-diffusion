#include "diffusion/DiffusionLoader.cu"

#include "ErrorCheck.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>

void test_forward_with_weights() {
    using DC = DiffusionConfig;
    const int B = 1;                      // batch size
    const int H = DC::img_resolution;
    const int W = DC::img_resolution;
    const int C = DC::in_channels;       // typically 1
    const int numel = B * C * H * W;
    const size_t bytes = numel * sizeof(__nv_bfloat16);

    // Create CUDA stream
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));

    // ------------------------------
    // Load weights from safetensors
    // ------------------------------
    DiffusionLoader loader;
    const std::string weight_path = loader.get_model_dir();
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

    // Print input values
    std::cout << "\n=== Input Values ===" << std::endl;
    std::cout << "Shape: [B=" << B << ", C=" << C << ", H=" << H << ", W=" << W << "]" << std::endl;
    std::cout << "First 16 values: ";
    for (int i = 0; i < std::min(16, numel); ++i) {
        std::cout << __bfloat162float(h_input[i]) << " ";
    }
    std::cout << std::endl;

    // ------------------------------
    // Perform forward pass
    // ------------------------------
    __nv_bfloat16* output = model->forward(static_cast<__nv_bfloat16 *>(input.data), h_tstamp, B);

    // ------------------------------
    // Print output values
    // ------------------------------
    std::vector<__nv_bfloat16> h_out(numel);
    checkCuda(cudaMemcpy(h_out.data(), output, bytes, cudaMemcpyDeviceToHost));

    std::cout << "\n=== Output Values ===" << std::endl;
    std::cout << "Shape: [B=" << B << ", C=1, H=" << H << ", W=" << W << "]" << std::endl;

    // Print first 16 values
    std::cout << "First 16 values: ";
    for (int i = 0; i < std::min(16, numel); ++i) {
        std::cout << __bfloat162float(h_out[i]) << " ";
    }
    std::cout << std::endl;

    // Print a small 5x5 grid from the center of the output
    std::cout << "\nCenter 5x5 grid of output:" << std::endl;
    int center_h = H / 2;
    int center_w = W / 2;
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            int idx = (center_h + i) * W + (center_w + j);
            if (idx >= 0 && idx < numel) {
                std::cout << std::fixed << std::setprecision(4) << __bfloat162float(h_out[idx]) << " ";
            }
        }
        std::cout << std::endl;
    }

    // Print statistics
    float min_val = __bfloat162float(h_out[0]);
    float max_val = __bfloat162float(h_out[0]);
    float sum = 0.0f;
    for (int i = 0; i < numel; ++i) {
        float val = __bfloat162float(h_out[i]);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    float mean = sum / numel;

    std::cout << "\nOutput Statistics:" << std::endl;
    std::cout << "Min: " << min_val << std::endl;
    std::cout << "Max: " << max_val << std::endl;
    std::cout << "Mean: " << mean << std::endl;

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
