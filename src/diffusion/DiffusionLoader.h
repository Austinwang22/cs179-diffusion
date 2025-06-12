// DiffusionLoader.h -----------------------------------------------------------
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include "../CudaBuffer.cuh"

#include "DiffusionConfig.h"
#include "DiffusionWeights.h"

#include "../vendor/safetensors.hh"

class DiffusionLoader {
public:
    /**  Look for the model path in DIFFUSION_MODEL_DIR or fall back.            */
    static std::string get_model_dir();

    /**  Load a BF16 tensor from a safetensors file and upload to GPU.           */
    static std::shared_ptr<CudaBuffer> load_bf16_tensor(safetensors::safetensors_t& st,
                                                        const std::string& name,
                                                        size_t dim0,
                                                        size_t dim1 = 0,
                                                        size_t dim2 = 0,
                                                        size_t dim3 = 0);

    /**  High-level helper: return *all* weights in a struct.                    */
    static DiffusionWeights load_diffusion_weights(const std::string& safetensors_file);
};