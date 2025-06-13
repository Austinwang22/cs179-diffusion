// DiffusionLoader.h -----------------------------------------------------------
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include "../CudaBuffer.cuh"

#include "DiffusionConfig.h"
#include "DiffusionUNet.cuh"

#include "../vendor/safetensors.hh"

class DiffusionLoader {
public:
    /**  Look for the model path in DIFFUSION_MODEL_DIR or fall back.            */
    static std::string get_model_dir();

    /**  Load a BF16 tensor from a safetensors file and upload to GPU.           */
    static std::shared_ptr<const CudaBuffer> load_bf16_tensor(safetensors::safetensors_t &st,
                                                              const std::string &name,
                                                              const std::vector<size_t> &expected);

    /**  High-level helper: return *all* weights in a struct.                    */
    static std::shared_ptr<UNetBF16> load_diffusion_weights(const std::string& safetensors_file);
};