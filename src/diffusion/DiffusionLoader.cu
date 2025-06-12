// DiffusionLoader.cu ----------------------------------------------------------
#include "DiffusionLoader.h"

#include "../ErrorCheck.h"
#include "../HostBuffer.h"
#include <cstdlib>
#include <iostream>

auto dump_bf16 = [](const char* tag, const __nv_bfloat16* dev, int n = 16)
{
    std::vector<__nv_bfloat16> h(n);
    checkCuda(cudaMemcpy(h.data(), dev, n * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost));
    std::cout << tag << " : ";
    for (int i = 0; i < n; ++i)
        std::cout << __bfloat162float(h[i]) << ' ';
    std::cout << '\n';
};

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
std::shared_ptr<CudaBuffer> DiffusionLoader::load_bf16_tensor(safetensors::safetensors_t& st,
                                  const std::string& name,
                                  size_t d0, size_t d1,
                                  size_t d2, size_t d3)
{
    safetensors::tensor_t t;
    bool found=false;
    for (size_t i=0;i<st.tensors.size();++i) {
        std::string k = st.tensors.keys()[i];
        if (k==name) { st.tensors.at(i,&t); found=true; break; }
    }
    if (!found)
        throw std::runtime_error("tensor not found: "+name);

    if (t.dtype != safetensors::kBFLOAT16)
        throw std::runtime_error("tensor "+name+" not BF16");

    // ---- shape check (allow unused dims == 0) --------------------
    // ------------------------------------------------------------------
    // 1.  tweak the helper so d1 == 0 means “1-D tensor – no 2nd dim check”
    // ------------------------------------------------------------------
    auto bad_shape=[&]{
        std::string got="(";
        for (auto s: t.shape) got+=std::to_string(s)+",";
        got.back()=')';
        throw std::runtime_error("shape mismatch for "+name+" got "+got);
    };
    if (t.shape[0] != d0) bad_shape();
    if (d1 && (t.shape.size() < 2 || t.shape[1] != d1)) bad_shape();
    if (d2 && (t.shape.size() < 3 || t.shape[2] != d2)) bad_shape();
    if (d3 && (t.shape.size() < 4 || t.shape[3] != d3)) bad_shape();


    // ---- copy bytes ------------------------------------------------
    size_t numel = 1;
    for (auto s: t.shape) numel*=s;
    size_t bytes = numel*sizeof(__nv_bfloat16);

    const uint8_t* src = st.databuffer_addr + t.data_offsets[0];
    auto dst = std::make_shared<CudaBuffer>(bytes);
    checkCuda(cudaMemcpy(dst->data, src, bytes, cudaMemcpyDefault));

    // dump_bf16(name.c_str(),
    //           static_cast<const __nv_bfloat16*>(dst->data),
    //           std::min<size_t>(numel, 16));

    return dst;
}

// ------------------------------------------------------------------
// main public loader
// ------------------------------------------------------------------
DiffusionWeights DiffusionLoader::load_diffusion_weights(const std::string& safet_file)
{
    DiffusionWeights W;

    // ---------- open safetensors ----------------------------------
    safetensors::safetensors_t st;
    std::string warn, err;
    if (!safetensors::mmap_from_file(safet_file, &st, &warn, &err))
        throw std::runtime_error("safetensors: "+err);
    if (!warn.empty()) std::cerr<<"safetensors warn: "<<warn<<'\n';
    if (!safetensors::validate_data_offsets(st, err))
        throw std::runtime_error("safetensors: invalid offsets: "+err);

    // ---------- constants from config -----------------------------
    constexpr int T = DiffusionConfig::t_emb_dim;   // 128
    constexpr int C0 = DiffusionConfig::enc_channels[1];   // 64
    constexpr int C1 = DiffusionConfig::enc_channels[2];   // 128

    constexpr int DEC0_OUT = DiffusionConfig::dec_channels[0];   // 128
    constexpr int DEC1_OUT = DiffusionConfig::dec_channels[1];   //  64
    constexpr int DEC0_IN  = C1 + C1;               // 256  (128 skip + 128 in)
    constexpr int DEC1_IN  = DEC0_OUT + C0;         // 192  (128  + 64)

    // ---------- table-drive the loads -----------------------------
    struct Entry { std::shared_ptr<CudaBuffer>* field;
                   std::string key;  int d0,d1,d2,d3; };

    // ---- time MLP
    const Entry tbl[] = {
        // ---------- time embedding ------------------------------------
        { &W.t_proj1_w, "unet.time_emb.mlp.0.weight", 4*T, T },
        { &W.t_proj1_b, "unet.time_emb.mlp.0.bias",   4*T, 0    },
        { &W.t_proj2_w, "unet.time_emb.mlp.2.weight", T,   4*T },
        { &W.t_proj2_b, "unet.time_emb.mlp.2.bias",   T,   0    },

        // ---------- encoder-0 -----------------------------------------
        { &W.enc[0].conv1_w,  "unet.enc_blocks.0.conv1.weight", C0, 1, 3,3 },
        { &W.enc[0].conv1_b,  "unet.enc_blocks.0.conv1.bias",   C0, 0      },
        { &W.enc[0].conv2_w,  "unet.enc_blocks.0.conv2.weight", C0, C0,3,3 },
        { &W.enc[0].conv2_b,  "unet.enc_blocks.0.conv2.bias",   C0, 0      },
        { &W.enc[0].t_proj_w, "unet.enc_blocks.0.time_proj.weight", C0, T  },
        { &W.enc[0].t_proj_b, "unet.enc_blocks.0.time_proj.bias",   C0, 0  },

        // ---------- encoder-1 -----------------------------------------
        { &W.enc[1].conv1_w,  "unet.enc_blocks.1.conv1.weight", C1, C0,3,3 },
        { &W.enc[1].conv1_b,  "unet.enc_blocks.1.conv1.bias",   C1, 0      },
        { &W.enc[1].conv2_w,  "unet.enc_blocks.1.conv2.weight", C1, C1,3,3 },
        { &W.enc[1].conv2_b,  "unet.enc_blocks.1.conv2.bias",   C1, 0      },
        { &W.enc[1].t_proj_w, "unet.enc_blocks.1.time_proj.weight", C1, T  },
        { &W.enc[1].t_proj_b, "unet.enc_blocks.1.time_proj.bias",   C1, 0  },

        // ---------- bottleneck ----------------------------------------
        { &W.bott1_w,       "unet.bottleneck.conv1.weight", C1, C1,3,3 },
        { &W.bott1_b,       "unet.bottleneck.conv1.bias",   C1, 0      },
        { &W.bott2_w,       "unet.bottleneck.conv2.weight", C1, C1,3,3 },
        { &W.bott2_b,       "unet.bottleneck.conv2.bias",   C1, 0      },
        { &W.bott_t_w,      "unet.bottleneck.time_proj.weight", C1, T  },
        { &W.bott_t_b,      "unet.bottleneck.time_proj.bias",   C1, 0  },

        // ---------- decoder block 0 -----------------------------------
        { &W.dec[0].conv1_w,  "unet.dec_blocks.0.conv1.weight", DEC0_OUT, DEC0_IN,3,3 },
        { &W.dec[0].conv1_b,  "unet.dec_blocks.0.conv1.bias",   DEC0_OUT, 0        },
        { &W.dec[0].conv2_w,  "unet.dec_blocks.0.conv2.weight", DEC0_OUT, DEC0_OUT,3,3 },
        { &W.dec[0].conv2_b,  "unet.dec_blocks.0.conv2.bias",   DEC0_OUT, 0        },
        { &W.dec[0].t_proj_w, "unet.dec_blocks.0.time_proj.weight", DEC0_OUT, T },
        { &W.dec[0].t_proj_b, "unet.dec_blocks.0.time_proj.bias",   DEC0_OUT, 0 },
        { &W.dec[0].up_w,     "unet.dec_blocks.0.upsample.weight",  DEC0_OUT, DEC0_OUT,2,2 },
        { &W.dec[0].up_b,     "unet.dec_blocks.0.upsample.bias",    DEC0_OUT, 0        },

        // ---------- decoder block 1 -----------------------------------
        { &W.dec[1].conv1_w,  "unet.dec_blocks.1.conv1.weight", DEC1_OUT, DEC1_IN,3,3 },
        { &W.dec[1].conv1_b,  "unet.dec_blocks.1.conv1.bias",   DEC1_OUT, 0        },
        { &W.dec[1].conv2_w,  "unet.dec_blocks.1.conv2.weight", DEC1_OUT, DEC1_OUT,3,3 },
        { &W.dec[1].conv2_b,  "unet.dec_blocks.1.conv2.bias",   DEC1_OUT, 0        },
        { &W.dec[1].t_proj_w, "unet.dec_blocks.1.time_proj.weight", DEC1_OUT, T },
        { &W.dec[1].t_proj_b, "unet.dec_blocks.1.time_proj.bias",   DEC1_OUT, 0 },
        { &W.dec[1].up_w,     "unet.dec_blocks.1.upsample.weight",  DEC1_OUT, DEC1_OUT,2,2 },
        { &W.dec[1].up_b,     "unet.dec_blocks.1.upsample.bias",    DEC1_OUT, 0        },

        // ---------- final output conv ---------------------------------
        { &W.out_w, "unet.out_conv.weight", DiffusionConfig::out_channels, DEC1_OUT, 1,1 },
        { &W.out_b, "unet.out_conv.bias",   DiffusionConfig::out_channels, 0             },
    };

    for (const auto& e : tbl)
        *e.field = load_bf16_tensor(st, e.key, e.d0, e.d1, e.d2, e.d3);

    return W;
}