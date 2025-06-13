// DiffusionUNetBF16.h
// Full UNet implementation (BF16, CUDA/cuDNN) matching the PyTorch model
// ────────────────────────────────────────────────────────────────────────────
#pragma once

#include "DiffusionLayers.cuh"     // LinearBF16, Conv2dBF16, ConvTrans2dBF16, MaxPool2dBF16

#include <assert.h>

namespace dm {

struct Shape { int B, C, H, W; };

// ────────────────────────────────────────────────────────────────────────────
// Encoder Block (Conv‑Conv‑ReLU + MaxPool)                                   
// ────────────────────────────────────────────────────────────────────────────
class EncoderBlockBF16 {
public:
    std::shared_ptr<Conv2dBF16>   conv1, conv2;
    std::shared_ptr<MaxPool2dBF16> pool;
    std::shared_ptr<LinearBF16>    t_proj;

    EncoderBlockBF16() {}

    // -------------------------------------------------------------
    // Forward pass
    // -------------------------------------------------------------
    void forward(__nv_bfloat16 *x, Shape &s, const __nv_bfloat16 *temb,
                 std::vector<std::shared_ptr<CudaBuffer>> &skips,
                 cudaStream_t st)
    {
        /* ---- 1. project time-embedding → FP32 bias tensor ---- */        
        // ensure_size(tbuf, size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
        tbuf = std::make_unique<CudaBuffer>(size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
        // for (int b = 0; b < s.B; ++b)
        //     t_proj->forward(
        //         temb + b * t_proj->in_f,
        //         reinterpret_cast<__nv_bfloat16*>(tbuf->data) + b * conv1->outC_,
        //         st);
        t_proj->forward(
            temb,
            reinterpret_cast<__nv_bfloat16*>(tbuf->data), // modified
            st
        );
        const __nv_bfloat16* tbias = reinterpret_cast<const __nv_bfloat16*>(tbuf->data);

        dump_chw("===1FUCKING RETARD ASS BITCH", reinterpret_cast<const __nv_bfloat16*>(tbias), conv1->outC_, s.H, s.W, st);

        /* ---- 2. workspace for conv1 ---- */
        // ensure_size(tmp, size_t(s.B) * conv1->outC_ * s.H * s.W  * sizeof(__nv_bfloat16));
        tmp = std::make_unique<CudaBuffer>(size_t(s.B) * conv1->outC_ * s.H * s.W  * sizeof(__nv_bfloat16));

        /* ---- 3. conv-1 → add FP32 bias → ReLU ---- */
            dump_chw("1 enc0-conv1 CUDA", x, conv1->outC_, s.H, s.W, st);
        conv1->forward(x, s.B, s.H, s.W, reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
            dump_chw("2 enc0-conv1 CUDA", reinterpret_cast<__nv_bfloat16*>(tmp->data), conv1->outC_, s.H, s.W, st);
        add_time_bias(reinterpret_cast<__nv_bfloat16*>(tmp->data), tbias, s.B, conv1->outC_, s.H, s.W, st);
            dump_chw("3 enc0-conv1 CUDA", reinterpret_cast<__nv_bfloat16*>(tmp->data), conv1->outC_, s.H, s.W, st);
        relu_inplace(reinterpret_cast<__nv_bfloat16*>(tmp->data), size_t(s.B) * conv1->outC_ * s.H * s.W, st);
            dump_chw("4 enc0-conv1 CUDA", reinterpret_cast<__nv_bfloat16*>(tmp->data), conv1->outC_, s.H, s.W, st);
            std::cerr << "=======\n";

        dump_chw("===2FUCKING RETARD ASS BITCH", reinterpret_cast<const __nv_bfloat16*>(tbias), conv1->outC_, s.H, s.W, st);

        // conv2 + bias + ReLU (in‑place to x)
            dump_chw("1 enc0-conv2 CUDA",  reinterpret_cast<__nv_bfloat16*>(tmp->data), conv2->outC_, s.H, s.W, st);
            dump_chw("===3FUCKING RETARD ASS BITCH", reinterpret_cast<const __nv_bfloat16*>(tbias), conv1->outC_, s.H, s.W, st);
        conv2->forward(reinterpret_cast<__nv_bfloat16*>(tmp->data), s.B, s.H, s.W, x, st);
            dump_chw("2 enc0-conv2 CUDA", x, conv2->outC_, s.H, s.W, st);
            dump_chw("===4FUCKING RETARD ASS BITCH", reinterpret_cast<const __nv_bfloat16*>(tbias), conv1->outC_, s.H, s.W, st);
        add_time_bias(x, tbias, s.B, conv2->outC_, s.H, s.W, st);
            dump_chw("3 enc0-conv2 CUDA", x, conv2->outC_, s.H, s.W, st);
        relu_inplace(x, size_t(s.B) * conv2->outC_ * s.H * s.W, st);
            dump_chw("4 enc0-conv2 CUDA", x, conv2->outC_, s.H, s.W, st);

        std::cerr << "========================\n";

        // s.C = conv2->outC_;

        // // save skip
        // auto saved = std::make_shared<CudaBuffer>(size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16));
        // cudaMemcpyAsync(saved->data, x, saved->size, cudaMemcpyDeviceToDevice, st);
        // skips.push_back(saved);

        // // MaxPool downsample
        // pool->forward(x, s.B, s.C, s.H, s.W, reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
        // cudaMemcpyAsync(x, tmp->data, size_t(s.B) * s.C * (s.H/2) * (s.W/2) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, st);
        // s.H /= 2; s.W /= 2;
        // channels already updated: s.C stays = conv2->outC_
    }

private:
    std::shared_ptr<CudaBuffer> tmp, tbuf;
};

// ────────────────────────────────────────────────────────────────────────────
// Bottleneck Block (Conv‑Conv‑ReLU, no pooling)                              
// ────────────────────────────────────────────────────────────────────────────
class BottleneckBF16 {
public:
    // BottleneckBF16(int ch, int t_dim)
    //     : conv1(ch, ch), conv2(ch, ch), t_proj(t_dim, ch) {}

    std::shared_ptr<Conv2dBF16> conv1, conv2;
    std::shared_ptr<LinearBF16> t_proj;

    BottleneckBF16() {};

    void forward(__nv_bfloat16 *x, Shape &s, const __nv_bfloat16 *temb, cudaStream_t st)
    {
        // project time → bias
        // ensure_size(tbuf, size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
        tbuf = std::make_shared<CudaBuffer>(size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
        for (int b = 0; b < s.B; ++b)
            t_proj->forward(
                temb + b * t_proj->in_f,
                reinterpret_cast<__nv_bfloat16*>(tbuf->data) + b * conv1->outC_,
                st);
        const __nv_bfloat16* tbias = reinterpret_cast<const __nv_bfloat16*>(tbuf->data);

        // conv1
        // ensure_size(tmp, size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16));
        tmp = std::make_shared<CudaBuffer>(size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16));
        
        conv1->forward(x, s.B, s.H, s.W,
                      reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
        add_time_bias(reinterpret_cast<__nv_bfloat16*>(tmp->data), tbias,
                      s.B, s.C, s.H, s.W, st);
        relu_inplace(reinterpret_cast<__nv_bfloat16*>(tmp->data), size_t(s.B)*s.C*s.H*s.W, st);

        // conv2 (back to x)
        conv2->forward(reinterpret_cast<__nv_bfloat16*>(tmp->data), s.B, s.H, s.W,
                      x, st);
        add_time_bias(x, tbias, s.B, s.C, s.H, s.W, st);
        relu_inplace(x, size_t(s.B)*s.C*s.H*s.W, st);
    }

private:
    std::shared_ptr<CudaBuffer> tmp, tbuf;
};

// ────────────────────────────────────────────────────────────────────────────
// Decoder Block  (skip‑concat → Conv‑Conv‑ReLU → ConvT ×2)                   
// ────────────────────────────────────────────────────────────────────────────
class DecoderBlockBF16 {
public:
    int outC_;
    std::shared_ptr<Conv2dBF16>      conv1, conv2;
    std::shared_ptr<ConvTrans2dBF16> up;
    std::shared_ptr<LinearBF16>      t_proj;

    DecoderBlockBF16(int outC): outC_(outC) {};

    void forward(__nv_bfloat16 *&x, Shape &s, const std::shared_ptr<CudaBuffer> &skip,
                 const __nv_bfloat16 *temb, cudaStream_t st)
    {
        // concat skip & x along channel dim → catBuf
        const size_t skipElems = skip->size / sizeof(__nv_bfloat16);
        int skipC = static_cast<int>(skipElems / (size_t(s.B) * s.H * s.W));
        // ensure_size(catBuf, size_t(s.B) * (s.C + skipC) * s.H * s.W * sizeof(__nv_bfloat16));
        catBuf = std::make_shared<CudaBuffer>(size_t(s.B) * (s.C + skipC) * s.H * s.W * sizeof(__nv_bfloat16));
        
        cudaMemcpyAsync(catBuf->data, skip->data, skip->size, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(reinterpret_cast<__nv_bfloat16*>(catBuf->data) + skipC * s.B * s.H * s.W,
                        x, size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, st);

        // time proj
        // ensure_size(tbuf, size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
        tbuf = std::make_shared<CudaBuffer>(size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
        for (int b = 0; b < s.B; ++b)
            t_proj->forward(
                temb + b * t_proj->in_f,
                reinterpret_cast<__nv_bfloat16*>(tbuf->data) + b * conv1->outC_,
                st);
        const __nv_bfloat16* tbias = reinterpret_cast<const __nv_bfloat16*>(tbuf->data);

        // conv1
        conv1->forward(reinterpret_cast<__nv_bfloat16*>(catBuf->data), s.B, s.H, s.W, x, st);
        add_time_bias(x, tbias, s.B, outC_, s.H, s.W, st);
        relu_inplace(x, size_t(s.B)*outC_*s.H*s.W, st);

        // conv2
        conv2->forward(x, s.B, s.H, s.W, x, st);
        add_time_bias(x, tbias, s.B, outC_, s.H, s.W, st);
        relu_inplace(x, size_t(s.B)*outC_*s.H*s.W, st);

        // upsample (ConvTranspose2d)
        up->forward(x, s.B, s.H, s.W, x, st);
        s.H *= 2; s.W *= 2; s.C = outC_;
    }

private:
    std::shared_ptr<CudaBuffer>      catBuf, tbuf;
};

// ────────────────────────────────────────────────────────────────────────────
// Full UNet                                                                      
// ────────────────────────────────────────────────────────────────────────────
class UNetBF16 {
    cudaStream_t stream;
public:
    UNetBF16(int resolution, int t_dim = 128): img_res(resolution) {
        // img_res(resolution), 
        //   timeEmb(t_dim),
        //   enc0(1,  64,  t_dim),
        //   enc1(64, 128, t_dim),
        //   bott(128,       t_dim),
        //   dec0(256, 128, t_dim),   // 128(in) + 128(skip)
        //   dec1(192,  64, t_dim),   // 128(in) +  64(skip)
        //   out_conv(64, 1) 
        checkCuda(cudaStreamCreate(&stream))
    }

    ~UNetBF16() {
        checkCuda(cudaStreamDestroy(stream));
    }

    // // sub‑modules
    // TimeEmbeddingBF16 timeEmb;
    // EncoderBlockBF16  enc0, enc1;
    // BottleneckBF16    bott;
    // DecoderBlockBF16  dec0, dec1;
    // Conv2dBF16        out_conv;

    std::shared_ptr<TimeEmbeddingBF16> timeEmb;
    std::shared_ptr<EncoderBlockBF16> enc0, enc1;
    std::shared_ptr<BottleneckBF16> bott;
    std::shared_ptr<DecoderBlockBF16> dec0, dec1;
    std::shared_ptr<Conv2dBF16> out_conv1;

    // ---------------------------------------------------------
    // forward
    // ---------------------------------------------------------
    void forward(__nv_bfloat16 *x, const int32_t *t_host, int B, cudaStream_t st) {
        // prepare time embedding
        // ensure_size(tembBuf, size_t(B) * timeEmb->dim * sizeof(__nv_bfloat16));
        tembBuf = std::make_shared<CudaBuffer>(size_t(B) * timeEmb->dim * sizeof(__nv_bfloat16));
        
        timeEmb->forward(t_host, B, reinterpret_cast<__nv_bfloat16*>(tembBuf->data), st);
        const __nv_bfloat16 *temb = reinterpret_cast<__nv_bfloat16*>(tembBuf->data);

        Shape s{B, 1, img_res, img_res};
        skips.clear(); skips.reserve(2);

        // encoders
        enc0->forward(x, s, temb, skips, st);
        enc1->forward(x, s, temb, skips, st);

        // // bottleneck
        // bott->forward(x, s, temb, st);

        // // decoders (in reverse skip order)
        // dec0->forward(x, s, skips.back(), temb, st); skips.pop_back();
        // dec1->forward(x, s, skips.back(), temb, st); skips.pop_back();

        // // final 1×1 conv, no bias add-time
        // out_conv->forward(x, s.B, s.H, s.W, x, st);
    }

private:
    int img_res;

    // scratch & skips
    std::shared_ptr<CudaBuffer> tembBuf;
    std::vector<std::shared_ptr<CudaBuffer>> skips;
};

} // namespace dm
