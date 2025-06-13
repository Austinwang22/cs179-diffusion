// DiffusionWeights.h
#pragma once
#include <memory>
#include "../CudaBuffer.cuh"

struct ConvPair { std::shared_ptr<const CudaBuffer> conv1_w, conv1_b, conv2_w, conv2_b;
                  std::shared_ptr<const CudaBuffer> t_proj_w, t_proj_b, up_w, up_b; };

struct EncPair  { std::shared_ptr<const CudaBuffer> conv1_w, conv1_b, conv2_w, conv2_b;
                  std::shared_ptr<const CudaBuffer> t_proj_w, t_proj_b; };

struct DiffusionWeights {
    // time-MLP
    std::shared_ptr<const CudaBuffer> t_proj1_w, t_proj1_b, t_proj2_w, t_proj2_b;

    // encoders
    EncPair enc[2];

    // bottleneck
    std::shared_ptr<const CudaBuffer> bott1_w, bott1_b, bott2_w, bott2_b, bott_t_w, bott_t_b;

    // decoders
    ConvPair dec[2];

    // output
    std::shared_ptr<const CudaBuffer> out_w, out_b;
};