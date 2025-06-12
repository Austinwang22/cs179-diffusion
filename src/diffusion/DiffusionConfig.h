#pragma once

#include <stdexcept>
#include <array>

struct DiffusionConfig {
    // ------- architecture --------------------------------------------------
    static constexpr int img_resolution = 28;
    static constexpr int in_channels    = 1;
    static constexpr int out_channels   = 1;

    inline static constexpr std::array<int,3> enc_channels{1, 64, 128};
    inline static constexpr std::array<int,2> dec_channels{128, 64}; 

    static constexpr int t_emb_dim  = 128;
    static constexpr int t_mlp_mult = 4;    // TimeEmbedding default

    static constexpr float sigma_data = 0.5f;

    // ------- training hyper‑parameters ------------------------------------
    static constexpr float  Pmean           = -1.2f;
    static constexpr float  Pstd            = 1.2f;
    static constexpr double learning_rate   = 1e-3;
    static constexpr int    epochs          = 2;
    static constexpr int    batch_size_train= 256;

    // ------- sampling defaults --------------------------------------------
    static constexpr int   num_steps        = 1000;
    static constexpr float sigma_min        = 0.002f;
    static constexpr float sigma_max        = 80.f;
    static constexpr int   rho              = 7;
    static constexpr int   batch_size_infer = 16;
};