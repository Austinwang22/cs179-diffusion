// DiffusionLoader.cu ----------------------------------------------------------
#include "DiffusionLoader.h"

#include "../ErrorCheck.h"
#include "../HostBuffer.h"
#include <cstdlib>
#include <iostream>
#include <vector>

// auto dump_bf16 = [](const char* tag, const __nv_bfloat16* dev, int n = 16)
// {
//     std::vector<__nv_bfloat16> h(n);
//     checkCuda(cudaMemcpy(h.data(), dev, n * sizeof(__nv_bfloat16),
//                          cudaMemcpyDeviceToHost));
//     std::cout << tag << " : ";
//     for (int i = 0; i < n; ++i)
//         std::cout << __bfloat162float(h[i]) << ' ';
//     std::cout << '\n';
// };

// ------------------------------------------------------------------
// util: decide model root directory
// ------------------------------------------------------------------
std::string DiffusionLoader::get_model_dir()
{
    const char *model_dir_env = std::getenv("DIFFUSION_MODEL_DIR");
    if (model_dir_env) {
        return model_dir_env;
    } else {
        return "../models/model_bf16.safetensors";
    }
}

// ------------------------------------------------------------------
// util: locate tensor, shape-check, upload to GPU
// ------------------------------------------------------------------

std::shared_ptr<const CudaBuffer>
DiffusionLoader::load_bf16_tensor(safetensors::safetensors_t &st,
                 const std::string &name,
                 const std::vector<size_t> &expected)
{
    // 1. locate tensor entry -------------------------------------------------
    safetensors::tensor_t t;
    bool found = false;
    for (size_t i = 0; i < st.tensors.size(); ++i) {
        if (st.tensors.keys()[i] == name) {
            st.tensors.at(i, &t);   // fills struct
            found = true; break;
        }
    }
    if (!found)
        throw std::runtime_error("tensor not found : " + name);

    // 2. dtype check ---------------------------------------------------------
    if (t.dtype != safetensors::kBFLOAT16)
        throw std::runtime_error("tensor " + name + " must be BF16");

    // 3. shape check (0 == wildcard) ----------------------------------------
    if (t.shape.size() != expected.size())
        throw std::runtime_error("rank mismatch for " + name);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] && expected[i] != t.shape[i]) {
            std::string got = "("; for (auto s: t.shape) got += std::to_string(s)+","; got.back() = ')';
            throw std::runtime_error("shape mismatch for " + name + ", expected index " + std::to_string(i));
        }
    }

    // 4. copy to GPU ---------------------------------------------------------
    size_t numel = 1; for (auto s : t.shape) numel *= s;
    size_t bytes = numel * sizeof(__nv_bfloat16);

    const uint8_t *src = st.databuffer_addr + t.data_offsets[0];
    auto dst = std::make_shared<CudaBuffer>(bytes);
    checkCuda(cudaMemcpy(dst->data, src, bytes, cudaMemcpyDefault));

    return std::static_pointer_cast<const CudaBuffer>(dst);
}

// ------------------------------------------------------------------
// main public loader
// ------------------------------------------------------------------
std::shared_ptr<UNetBF16> DiffusionLoader::load_diffusion_weights(const std::string& safet_file)
{
    // DiffusionWeights W;

    // ---------- open safetensors ----------------------------------
    std::string warn, err;
    safetensors::safetensors_t st;
    bool ret = safetensors::mmap_from_file(safet_file, &st, &warn, &err);
    if (!warn.empty()) {
        std::cerr << "safetensors warning: " << warn << std::endl;
    }
    if (!ret) {
        throw std::runtime_error("safetensors error: " + err);
    }
    if (!safetensors::validate_data_offsets(st, err)) {
        throw std::runtime_error("safetensors: invalid data offsets: " + err);
    }

    // ---------- constants from config -----------------------------
    constexpr int T = DiffusionConfig::t_emb_dim;   // 128
    constexpr int C0 = DiffusionConfig::enc_channels[1];   // 64
    constexpr int C1 = DiffusionConfig::enc_channels[2];   // 128

    constexpr int D0O = DiffusionConfig::dec_channels[0];   // 128
    constexpr int D1O = DiffusionConfig::dec_channels[1];   //  64

    // // ---------- table-drive the loads -----------------------------
    // struct Entry { std::shared_ptr<const CudaBuffer> *field;
    //                std::string key;  int d0,d1,d2,d3; };
    // unet model
    auto net = std::make_shared<UNetBF16>(28);

    // ---- Time embedding ------------------------
    net->timeEmb = std::make_shared<TimeEmbeddingBF16>(T);
    net->timeEmb->proj1 = std::make_shared<LinearBF16>(T, 4*T);
    net->timeEmb->proj2 = std::make_shared<LinearBF16>(4*T, T);
    net->timeEmb->proj1->W_ = load_bf16_tensor(st, "unet.time_emb.mlp.0.weight", std::vector<size_t>{4*T, T});
    net->timeEmb->proj1->b_ = load_bf16_tensor(st, "unet.time_emb.mlp.0.bias",   std::vector<size_t>{4*T});
    net->timeEmb->proj2->W_ = load_bf16_tensor(st, "unet.time_emb.mlp.2.weight", std::vector<size_t>{T, 4*T});
    net->timeEmb->proj2->b_ = load_bf16_tensor(st, "unet.time_emb.mlp.2.bias",   std::vector<size_t>{T});

    // ---- Encoder‑0 -----------------------------
    net->enc0 = std::make_shared<EncoderBlockBF16>();
    net->enc0->conv1 = std::make_shared<Conv2dBF16>(1,  C0);
    net->enc0->conv1->W_ = load_bf16_tensor(st, "unet.enc_blocks.0.conv1.weight", std::vector<size_t>{C0, 1, 3, 3});
    net->enc0->conv1->b_ = load_bf16_tensor(st, "unet.enc_blocks.0.conv1.bias",   std::vector<size_t>{C0});
    
    net->enc0->conv2 = std::make_shared<Conv2dBF16>(C0, C0);
    net->enc0->conv2->W_ = load_bf16_tensor(st, "unet.enc_blocks.0.conv2.weight", std::vector<size_t>{C0, C0, 3, 3});
    net->enc0->conv2->b_ = load_bf16_tensor(st, "unet.enc_blocks.0.conv2.bias",   std::vector<size_t>{C0});
    
    net->enc0->t_proj = std::make_shared<LinearBF16>(T, C0);
    net->enc0->t_proj->W_ = load_bf16_tensor(st, "unet.enc_blocks.0.time_proj.weight", std::vector<size_t>{C0, T});
    net->enc0->t_proj->b_ = load_bf16_tensor(st, "unet.enc_blocks.0.time_proj.bias",   std::vector<size_t>{C0});

    // ---- Encoder‑1 -----------------------------
    net->enc1 = std::make_shared<EncoderBlockBF16>();
    net->enc1->conv1 = std::make_shared<Conv2dBF16>(C0, C1);
    net->enc1->conv1->W_ = load_bf16_tensor(st, "unet.enc_blocks.1.conv1.weight", std::vector<size_t>{C1, C0, 3, 3});
    net->enc1->conv1->b_ = load_bf16_tensor(st, "unet.enc_blocks.1.conv1.bias",   std::vector<size_t>{C1});
    net->enc1->conv2 = std::make_shared<Conv2dBF16>(C1, C1);
    net->enc1->conv2->W_ = load_bf16_tensor(st, "unet.enc_blocks.1.conv2.weight", std::vector<size_t>{C1, C1, 3, 3});
    net->enc1->conv2->b_ = load_bf16_tensor(st, "unet.enc_blocks.1.conv2.bias",   std::vector<size_t>{C1});
    net->enc1->t_proj = std::make_shared<LinearBF16>(T, C1);
    net->enc1->t_proj->W_ = load_bf16_tensor(st, "unet.enc_blocks.1.time_proj.weight", std::vector<size_t>{C1, T});
    net->enc1->t_proj->b_ = load_bf16_tensor(st, "unet.enc_blocks.1.time_proj.bias",   std::vector<size_t>{C1});

    // ---- Bottleneck ----------------------------
    net->bott = std::make_shared<BottleneckBF16>();
    net->bott->conv1 = std::make_shared<Conv2dBF16>(C1, C1);
    net->bott->conv1->W_ = load_bf16_tensor(st, "unet.bottleneck.conv1.weight", std::vector<size_t>{C1, C1, 3, 3});
    net->bott->conv1->b_ = load_bf16_tensor(st, "unet.bottleneck.conv1.bias",   std::vector<size_t>{C1});
    net->bott->conv2 = std::make_shared<Conv2dBF16>(C1, C1);
    net->bott->conv2->W_ = load_bf16_tensor(st, "unet.bottleneck.conv2.weight", std::vector<size_t>{C1, C1, 3, 3});
    net->bott->conv2->b_ = load_bf16_tensor(st, "unet.bottleneck.conv2.bias",   std::vector<size_t>{C1});
    net->bott->t_proj = std::make_shared<LinearBF16>(T, C1);
    net->bott->t_proj->W_ = load_bf16_tensor(st, "unet.bottleneck.time_proj.weight", std::vector<size_t>{C1, T});
    net->bott->t_proj->b_ = load_bf16_tensor(st, "unet.bottleneck.time_proj.bias",   std::vector<size_t>{C1});

    // ---- Decoder‑0 -----------------------------
    net->dec0 = std::make_shared<DecoderBlockBF16>(D0O);
    net->dec0->conv1 = std::make_shared<Conv2dBF16>(C1+C1, D0O);
    net->dec0->conv1->W_ = load_bf16_tensor(st, "unet.dec_blocks.0.conv1.weight", std::vector<size_t>{D0O, C1+C1, 3, 3});
    net->dec0->conv1->b_ = load_bf16_tensor(st, "unet.dec_blocks.0.conv1.bias",   std::vector<size_t>{D0O});

    net->dec0->conv2 = std::make_shared<Conv2dBF16>(D0O, D0O);
    net->dec0->conv2->W_ = load_bf16_tensor(st, "unet.dec_blocks.0.conv2.weight", std::vector<size_t>{D0O, D0O, 3, 3});
    net->dec0->conv2->b_ = load_bf16_tensor(st, "unet.dec_blocks.0.conv2.bias",   std::vector<size_t>{D0O});
    net->dec0->up = std::make_shared<ConvTrans2dBF16>(D0O, D0O);
    net->dec0->up->W_ = load_bf16_tensor(st, "unet.dec_blocks.0.upsample.weight", std::vector<size_t>{D0O, D0O, 2, 2});
    net->dec0->up->b_ = load_bf16_tensor(st, "unet.dec_blocks.0.upsample.bias",   std::vector<size_t>{D0O});

    net->dec0->t_proj = std::make_shared<LinearBF16>(T, D0O);
    net->dec0->t_proj->W_ = load_bf16_tensor(st, "unet.dec_blocks.0.time_proj.weight", std::vector<size_t>{D0O, T});
    net->dec0->t_proj->b_ = load_bf16_tensor(st, "unet.dec_blocks.0.time_proj.bias",   std::vector<size_t>{D0O});

    // ---- Decoder‑1 -----------------------------
    net->dec1 = std::make_shared<DecoderBlockBF16>(D1O);
    net->dec1->conv1 = std::make_shared<Conv2dBF16>(D0O + C0, D1O);
    net->dec1->conv1->W_ = load_bf16_tensor(st, "unet.dec_blocks.1.conv1.weight", std::vector<size_t>{D1O, D0O+C0, 3, 3});
    net->dec1->conv1->b_ = load_bf16_tensor(st, "unet.dec_blocks.1.conv1.bias",   std::vector<size_t>{D1O});
    net->dec1->conv2 = std::make_shared<Conv2dBF16>(D1O, D1O);
    net->dec1->conv2->W_ = load_bf16_tensor(st, "unet.dec_blocks.1.conv2.weight", std::vector<size_t>{D1O, D1O, 3, 3});
    net->dec1->conv2->b_ = load_bf16_tensor(st, "unet.dec_blocks.1.conv2.bias",   std::vector<size_t>{D1O});
    net->dec1->up = std::make_shared<ConvTrans2dBF16>(D1O, D1O);
    net->dec1->up->W_ = load_bf16_tensor(st, "unet.dec_blocks.1.upsample.weight", std::vector<size_t>{D1O, D1O, 2, 2});
    net->dec1->up->b_ = load_bf16_tensor(st, "unet.dec_blocks.1.upsample.bias",   std::vector<size_t>{D1O});
    
    net->dec1->t_proj = std::make_shared<LinearBF16>(T, D1O);
    net->dec1->t_proj->W_ = load_bf16_tensor(st, "unet.dec_blocks.1.time_proj.weight", std::vector<size_t>{D1O, T});
    net->dec1->t_proj->b_ = load_bf16_tensor(st, "unet.dec_blocks.1.time_proj.bias",   std::vector<size_t>{D1O});

    // ---- head conv -----------------------------
    net->out_conv1 = std::make_shared<Conv2dBF16>(D1O, DiffusionConfig::out_channels);
    net->out_conv1->W_ = load_bf16_tensor(st, "unet.out_conv.weight", std::vector<size_t>{DiffusionConfig::out_channels, D1O, 1, 1});
    net->out_conv1->b_ = load_bf16_tensor(st, "unet.out_conv.bias",   std::vector<size_t>{DiffusionConfig::out_channels});

    return net;
}