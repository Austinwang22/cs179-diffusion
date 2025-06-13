// DiffusionUNetBF16.h
// Full UNet implementation (BF16, CUDA/cuDNN) matching the PyTorch model
// ────────────────────────────────────────────────────────────────────────────
#pragma once

#include "DiffusionLayers.cuh"     // LinearBF16, Conv2dBF16, ConvTrans2dBF16, MaxPool2dBF16

#include <assert.h>

namespace dm {


auto dump_bf16_2 = [](const char* tag, const __nv_bfloat16* dev, int n = 16)
{
    n = std::min(n, 10);

    std::vector<__nv_bfloat16> h(n);
    // std::cout << "dump_bf16: " << tag << " : ";
    // if (!dev) {
    //     std::cout << "null pointer\n";
    // }
    checkCuda(cudaMemcpy(h.data(), dev, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < n; ++i)
    //     std::cout << __bfloat162float(h[i]) << ' ';
    // std::cout << '\n';
};

struct Shape { int B, C, H, W; };

class EncoderBlockBF16 {
public:
    std::shared_ptr<Conv2dBF16>    conv1, conv2;
    std::shared_ptr<MaxPool2dBF16> pool;
    std::shared_ptr<LinearBF16>    t_proj;

    // scratch
    std::shared_ptr<CudaBuffer> tmp;    // conv workspace (after conv‑1)
    std::shared_ptr<CudaBuffer> xbuf;   // activation buffer large enough for conv‑2
    std::shared_ptr<CudaBuffer> tbuf;   // [B, outC]

    EncoderBlockBF16() {};

    void forward(__nv_bfloat16                 *&x,      // may be replaced
                 Shape                         &s,
                 const __nv_bfloat16           *temb,
                 std::vector<std::shared_ptr<CudaBuffer>> &skips,
                 cudaStream_t                   st)
    {
        const int B = s.B;

        //---------------- Time proj → tbias -------------------------
        const int outC1 = conv1->outC_;
        size_t tbBytes  = size_t(B) * outC1 * sizeof(__nv_bfloat16);
        if (!tbuf || tbuf->size < tbBytes) tbuf = std::make_shared<CudaBuffer>(tbBytes);
        auto tbias = reinterpret_cast<__nv_bfloat16*>(tbuf->data);
        for (int b = 0; b < B; ++b)
            t_proj->forward(temb + b * t_proj->in_f,
                            tbias + b * outC1,
                            st);

        //---------------- conv‑1 ------------------------------------
        // dump_chw("\nstarting x", x, conv1->outC_, s.H, s.W, st);
        
        size_t tmpBytes = size_t(B) * outC1 * s.H * s.W * sizeof(__nv_bfloat16);
        if (!tmp || tmp->size < tmpBytes) tmp = std::make_shared<CudaBuffer>(tmpBytes);
        
        conv1->forward(x, B, s.H, s.W, reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
            // dump_chw("\nafter first convolution x",  reinterpret_cast<__nv_bfloat16*>(tmp->data), conv2->outC_, s.H, s.W, st);
        add_time_bias(reinterpret_cast<__nv_bfloat16*>(tmp->data), tbias,
                      B, outC1, s.H, s.W, st);
            // dump_chw("\nafter first bias x",  reinterpret_cast<__nv_bfloat16*>(tmp->data), conv2->outC_, s.H, s.W, st);
        relu_inplace(reinterpret_cast<__nv_bfloat16*>(tmp->data), size_t(B)*outC1*s.H*s.W, st);

        //---------------- conv‑2 (may change channels) --------------
        // dump_chw("\nafter first conv-add block x",  reinterpret_cast<__nv_bfloat16*>(tmp->data), conv2->outC_, s.H, s.W, st);
        
        const int outC2 = conv2->outC_;
        size_t xbBytes  = size_t(B) * outC2 * s.H * s.W * sizeof(__nv_bfloat16);
        if (!xbuf || xbuf->size < xbBytes) xbuf = std::make_shared<CudaBuffer>(xbBytes);
        conv2->forward(reinterpret_cast<__nv_bfloat16*>(tmp->data), B, s.H, s.W,
                       reinterpret_cast<__nv_bfloat16*>(xbuf->data), st);
        add_time_bias(reinterpret_cast<__nv_bfloat16*>(xbuf->data), tbias,
                      B, outC2, s.H, s.W, st);
        relu_inplace(reinterpret_cast<__nv_bfloat16*>(xbuf->data), size_t(B)*outC2*s.H*s.W, st);

        // hand x back to caller and update shape
        x = reinterpret_cast<__nv_bfloat16*>(xbuf->data);
        s.C = outC2;

        // dump_chw("x after conv2", x, outC2, s.H, s.W, st);

        //---------------- MaxPool ↓2 --------------------------------
        pool->forward(x, B, s.C, s.H, s.W, reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
        cudaMemcpyAsync(x, tmp->data,
                        size_t(B)*s.C*(s.H/2)*(s.W/2)*sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, st);
        s.H /= 2;  s.W /= 2;

        // dump_chw("x pooling", x, outC2, s.H, s.W, st);

        //---------------- save skip AFTER pooling ------------------
        auto saved = std::make_shared<CudaBuffer>(size_t(B)*s.C*s.H*s.W*sizeof(__nv_bfloat16));
        cudaMemcpyAsync(saved->data, x, saved->size, cudaMemcpyDeviceToDevice, st);
        skips.push_back(saved);

        // std::cerr << "============================\n";

        // Print the shape of the weights of conv2
        // std::cout << "conv2 weights shape: [" << conv2->outC_ << ", " << conv2->inC_ << "]" << std::endl;
    }
};

//──────────────────────────────────────────────────────────────────────
// Bottleneck – fix tmp sizing (use conv1->outC_)
//──────────────────────────────────────────────────────────────────────
class BottleneckBF16 {
public:
    std::shared_ptr<Conv2dBF16> conv1, conv2;   // typically same C
    std::shared_ptr<LinearBF16> t_proj;
    std::shared_ptr<CudaBuffer> tmp, tbuf;

    void forward(__nv_bfloat16 *x, Shape &s, const __nv_bfloat16 *temb, cudaStream_t st) {
        const int B = s.B, H = s.H, W = s.W;
        const int C = conv1->outC_;

        size_t tbBytes = size_t(B) * C * sizeof(__nv_bfloat16);
        if (!tbuf || tbuf->size < tbBytes) tbuf = std::make_shared<CudaBuffer>(tbBytes);
        for (int b = 0; b < B; ++b)
            t_proj->forward(temb + b * t_proj->in_f,
                            reinterpret_cast<__nv_bfloat16*>(tbuf->data) + b * C,
                            st);
        const __nv_bfloat16 *tbias = reinterpret_cast<__nv_bfloat16*>(tbuf->data);

        size_t tmpBytes = size_t(B) * C * H * W * sizeof(__nv_bfloat16);
        if (!tmp || tmp->size < tmpBytes) tmp = std::make_shared<CudaBuffer>(tmpBytes);

        conv1->forward(x, B, H, W, reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
        add_time_bias(reinterpret_cast<__nv_bfloat16*>(tmp->data), tbias, B, C, H, W, st);
        relu_inplace(reinterpret_cast<__nv_bfloat16*>(tmp->data), size_t(B)*C*H*W, st);

        conv2->forward(reinterpret_cast<__nv_bfloat16*>(tmp->data), B, H, W, x, st);
        add_time_bias(x, tbias, B, C, H, W, st);
        relu_inplace(x, size_t(B)*C*H*W, st);
    }
};

//──────────────────────────────────────────────────────────────────────
// Decoder Block – correct skip‑channel computation
//──────────────────────────────────────────────────────────────────────
class DecoderBlockBF16 {
public:
    int outC_;
    std::shared_ptr<Conv2dBF16>      conv1, conv2;
    std::shared_ptr<ConvTrans2dBF16> up;
    std::shared_ptr<LinearBF16>      t_proj;

    // scratch
    std::shared_ptr<CudaBuffer> catBuf, tbuf, tmp, upBuf, xBuf;  // Added upBuf for upconv output

    explicit DecoderBlockBF16(int outC) : outC_(outC) {}

    void forward(__nv_bfloat16      *&x,
                 Shape              &s,
                 const std::shared_ptr<CudaBuffer> &skip,
                 const __nv_bfloat16*temb,
                 cudaStream_t        st)
    {
        const int B = s.B, H = s.H, W = s.W;
        const size_t skipElems = skip->size / sizeof(__nv_bfloat16);
        int skipC = static_cast<int>(skipElems / (size_t(s.B) * s.H * s.W));
        size_t catBytes = size_t(B) * (s.C + skipC) * H * W * sizeof(__nv_bfloat16);
        if (!catBuf || catBuf->size < catBytes) catBuf = std::make_shared<CudaBuffer>(catBytes);

        // concat along channel dim
        cudaMemcpyAsync(catBuf->data, x, size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(reinterpret_cast<__nv_bfloat16*>(catBuf->data) + s.C * s.B * s.H * s.W,
                        skip->data, skip->size, cudaMemcpyDeviceToDevice, st);

        // tbias for this block
        size_t tbBytes = size_t(B) * conv1->outC_ * sizeof(__nv_bfloat16);
        if (!tbuf || tbuf->size < tbBytes) tbuf = std::make_shared<CudaBuffer>(tbBytes);
        for (int b = 0; b < B; ++b)
            t_proj->forward(temb + b * t_proj->in_f,
                            reinterpret_cast<__nv_bfloat16*>(tbuf->data) + b * conv1->outC_, st);
        const __nv_bfloat16 *tbias = reinterpret_cast<__nv_bfloat16*>(tbuf->data);

        // conv‑1 / conv‑2 / upsample
        conv1->forward(reinterpret_cast<__nv_bfloat16*>(catBuf->data), B, H, W, x, st);

        add_time_bias(x, tbias, B, outC_, H, W, st);

        relu_inplace(x, size_t(B)*outC_*H*W, st);
        
        // Use temporary buffer for conv2 output
        size_t tmpBytes = size_t(B) * outC_ * H * W * sizeof(__nv_bfloat16);
        if (!tmp || tmp->size < tmpBytes) tmp = std::make_shared<CudaBuffer>(tmpBytes);
        
        conv2->forward(x, B, H, W, reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
        cudaMemcpyAsync(x, tmp->data, tmpBytes, cudaMemcpyDeviceToDevice, st);

        add_time_bias(x, tbias, B, outC_, H, W, st);

        relu_inplace(x, size_t(B)*outC_*H*W, st);

        // Use separate buffer for upconv output
        size_t upBytes = size_t(B) * outC_ * (H * 2) * (W * 2) * sizeof(__nv_bfloat16);
        if (!upBuf || upBuf->size < upBytes) upBuf = std::make_shared<CudaBuffer>(upBytes);
        
        up->forward(x, B, H, W, reinterpret_cast<__nv_bfloat16*>(upBuf->data), st);

        // resize x and copy
        xBuf = std::make_shared<CudaBuffer>(upBytes);
        cudaMemcpyAsync(xBuf->data, upBuf->data, upBytes, cudaMemcpyDeviceToDevice, st);
        x = reinterpret_cast<__nv_bfloat16*>(xBuf->data);

        s.H *= 2;  s.W *= 2;  s.C = outC_;    }
};

//──────────────────────────────────────────────────────────────────────
// Full UNet wrapper (only skip vector type changed)
//──────────────────────────────────────────────────────────────────────
class UNetBF16 {
    cudaStream_t stream;
public:
    UNetBF16(int res) : img_res(res) { checkCuda(cudaStreamCreate(&stream)); }
    ~UNetBF16()                              { checkCuda(cudaStreamDestroy(stream)); }

    // sub‑modules (prepare exactly as before)
    std::shared_ptr<TimeEmbeddingBF16> timeEmb;
    std::shared_ptr<EncoderBlockBF16>  enc0, enc1;
    std::shared_ptr<BottleneckBF16>    bott;
    std::shared_ptr<DecoderBlockBF16>  dec0, dec1;
    std::shared_ptr<Conv2dBF16>        out_conv1;

    std::shared_ptr<CudaBuffer> tembBuf;
    std::shared_ptr<CudaBuffer> outBuf;  // Add output buffer as member

    __nv_bfloat16* forward(__nv_bfloat16 *x, const int32_t *t_host, int B) {
        // convenience alias
        auto st = stream;

        //---------------- make sinusoid + MLP time embedding ----------
        if (!tembBuf || tembBuf->size < size_t(B) * timeEmb->dim * sizeof(__nv_bfloat16))
            tembBuf = std::make_shared<CudaBuffer>(size_t(B) * timeEmb->dim * sizeof(__nv_bfloat16));
        timeEmb->forward(t_host, B, reinterpret_cast<__nv_bfloat16*>(tembBuf->data), st);
        cudaStreamSynchronize(st);
        const __nv_bfloat16 *temb = reinterpret_cast<__nv_bfloat16*>(tembBuf->data);

        //---------------- stage‑by‑stage ----------------------------------
        Shape s{B, 1, img_res, img_res};
        skips.clear();
        skips.reserve(2);

        enc0->forward(x, s, temb, skips, st); // working
        enc1->forward(x, s, temb, skips, st); // working

        bott->forward(x, s, temb, st);

        // dump_chw("x after bottleneck", x, s.C, s.H, s.W, st);

        dec0->forward(x, s, skips.back(), temb, st); skips.pop_back();

        // dump_chw("x after decoder0", x, s.C, s.H, s.W, st);

        dec1->forward(x, s, skips.back(), temb, st); skips.pop_back();

        // dump_chw("x after decoder1", x, s.C, s.H, s.W, st);

        // Print weights of out_conv1
        // dump_bf16_2("out_conv1 weights", reinterpret_cast<__nv_bfloat16*>(out_conv1->W_->data), out_conv1->W_->size / sizeof(__nv_bfloat16));

        // Create or resize output buffer if needed
        size_t outBytes = size_t(B) * 1 * s.H * s.W * sizeof(__nv_bfloat16);
        if (!outBuf || outBuf->size < outBytes) {
            outBuf = std::make_shared<CudaBuffer>(outBytes);
        }
        
        // Perform the 1x1 convolution
        out_conv1->forward(x, s.B, s.H, s.W, reinterpret_cast<__nv_bfloat16*>(outBuf->data), st);
        
        // dump_chw("x after out_conv1", reinterpret_cast<__nv_bfloat16*>(outBuf->data), 1, s.H, s.W, st);
        
        return reinterpret_cast<__nv_bfloat16*>(outBuf->data);
    }

private:
    int img_res;
    std::vector<std::shared_ptr<CudaBuffer>> skips;  // channel‑aware skips
};

// // ────────────────────────────────────────────────────────────────────────────
// // Encoder Block (Conv‑Conv‑ReLU + MaxPool)                                   
// // ────────────────────────────────────────────────────────────────────────────
// class EncoderBlockBF16 {
// public:
//     std::shared_ptr<Conv2dBF16>    conv1, conv2;
//     std::shared_ptr<MaxPool2dBF16> pool;
//     std::shared_ptr<LinearBF16>    t_proj;

//     std::shared_ptr<CudaBuffer> tmp;
//     std::shared_ptr<CudaBuffer> tbuf;

//     EncoderBlockBF16() {}

//     // -------------------------------------------------------------
//     // Forward pass
//     // -------------------------------------------------------------
//     void forward(__nv_bfloat16 *x, Shape &s, const __nv_bfloat16 *temb,
//                  std::vector<std::shared_ptr<CudaBuffer>> &skips,
//                  cudaStream_t st,
//                  std::shared_ptr<EncoderBlockBF16> enc1 = nullptr) {

//         /* ---- 1. project time-embedding → FP32 bias tensor ---- */        
//         // auto tbuf = std::make_unique<CudaBuffer>(s.B * conv1->outC_ * sizeof(__nv_bfloat16));
//         // t_proj->forward(temb, reinterpret_cast<__nv_bfloat16*>(tbuf->data), st);
//         // auto tbuf = std::make_unique<CudaBuffer>(s.B * conv1->outC_ * sizeof(__nv_bfloat16));
//         // for (int b = 0; b < s.B; ++b) {
//         //     t_proj->forward(
//         //         temb + b * t_proj->in_f,                     //<– each batch's slice of temb
//         //         reinterpret_cast<__nv_bfloat16*>(tbuf->data) 
//         //         + b * conv1->outC_,                        //<– write into the b-th bias slot
//         //         st);
//         // }

//         const int outC = conv1->outC_;

//         if (!tbuf || tbuf->size < size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16)) {
//             tbuf = std::make_shared<CudaBuffer>(size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
//         }
//         auto tbias = reinterpret_cast<__nv_bfloat16*>(tbuf->data);

//         std::cerr << "THE PROJECTION\n";
//         for (int b = 0; b < s.B; ++b) {              // ←  restore this loop
//             const __nv_bfloat16* src = temb + b * t_proj->in_f;      // [T]
//             __nv_bfloat16*       dst = tbias + b * outC;             // [outC]
//             t_proj->forward(src, dst, st);                           // LinearBF16
//         }
//         std::cerr << "THE PROJECTION\n";

//         dump_bf16_2("THE WEIGHTS AFTER THE TIME PROJECTION model encoder1 conv1 weight",
//             reinterpret_cast<__nv_bfloat16*>(enc1->t_proj->W_->data), 
//             enc1->t_proj->W_->size / sizeof(__nv_bfloat16));

//         dump_chw("=========== temb", temb, conv1->outC_, s.H, s.W, st);
//         dump_chw("=========== TBUF", reinterpret_cast<const __nv_bfloat16*>(tbuf->data), conv1->outC_, s.H, s.W, st);

//         // for (int b = 0; b < s.B; b++)
//         //     t_proj->forward(
//         //         temb + b * t_proj->in_f,
//         //         reinterpret_cast<__nv_bfloat16*>(tbuf->data) + b * conv1->outC_,
//         //         st);
//         // const __nv_bfloat16* tbias = reinterpret_cast<const __nv_bfloat16*>(tbuf->data);
 
//         /* ---- 2. workspace for conv1 ---- */
//         if (!tmp || tmp->size < size_t(s.B) * conv1->outC_ * s.H * s.W * sizeof(__nv_bfloat16)) {
//             tmp = std::make_shared<CudaBuffer>(size_t(s.B) * conv1->outC_ * s.H * s.W * sizeof(__nv_bfloat16));
//         }

//         // std::cerr << "tbuf has size: " << s.B * conv1->outC_ << " " << tbuf->size << "\n";
//         // std::cerr << "multiply by " << sizeof(__nv_bfloat16) << " " << "\n";
//         // std::cerr << "tmp has size: " << s.B * conv1->outC_ * s.H * s.W << " " << tmp->size << "\n";
//         // std::cerr << "multiply by " << sizeof(__nv_bfloat16) << "\n";

//         /* ---- 3. conv-1 → add FP32 bias → ReLU ---- */
//             dump_chw("\nstarting x", x, conv1->outC_, s.H, s.W, st);
//         conv1->forward(x, s.B, s.H, s.W, reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
//             dump_chw("x after convolution (temp)", reinterpret_cast<__nv_bfloat16*>(tmp->data), conv1->outC_, s.H, s.W, st);
//         add_time_bias(reinterpret_cast<__nv_bfloat16*>(tmp->data), reinterpret_cast<const __nv_bfloat16*>(tbuf->data), s.B, conv1->outC_, s.H, s.W, st);
//             dump_chw("x after time bias (temp)", reinterpret_cast<__nv_bfloat16*>(tmp->data), conv1->outC_, s.H, s.W, st);
//         relu_inplace(reinterpret_cast<__nv_bfloat16*>(tmp->data), size_t(s.B) * conv1->outC_ * s.H * s.W, st);
//             dump_chw("x after relu (temp)", reinterpret_cast<__nv_bfloat16*>(tmp->data), conv1->outC_, s.H, s.W, st);
//             std::cerr << "=======\n";

//         cudaStreamSynchronize(st);

//         dump_bf16_2("THE WEIGHTS AFTER THE FIRST CONVOLUTION PASS model encoder1 conv1 weight",
//             reinterpret_cast<__nv_bfloat16*>(enc1->t_proj->W_->data), 
//             enc1->t_proj->W_->size / sizeof(__nv_bfloat16));

//         // dump_chw("============ tbuf after mini block", reinterpret_cast<const __nv_bfloat16*>(tbuf->data), conv1->outC_, s.H, s.W, st);

//         // conv2 + bias + ReLU (in‑place to x)
//             dump_chw("\nstarting x",  reinterpret_cast<__nv_bfloat16*>(tmp->data), conv2->outC_, s.H, s.W, st);
//         conv2->forward(reinterpret_cast<__nv_bfloat16*>(tmp->data), s.B, s.H, s.W, x, st);
//             dump_chw("x after convolution (x)", x, conv2->outC_, s.H, s.W, st);
        
//         dump_bf16_2("THE WEIGHTS AFTER THE SECOND CONVOLUTION PASS -- CONVOLUTION model encoder1 conv1 weight",
//             reinterpret_cast<__nv_bfloat16*>(enc1->t_proj->W_->data), 
//             enc1->t_proj->W_->size / sizeof(__nv_bfloat16));
        
//         add_time_bias(x, reinterpret_cast<const __nv_bfloat16*>(tbuf->data), s.B, conv2->outC_, s.H, s.W, st);
//             dump_chw("x after time bias (x)", x, conv2->outC_, s.H, s.W, st);
        
//         dump_bf16_2("THE WEIGHTS AFTER THE SECOND CONVOLUTION PASS -- ADD TIME BIAS model encoder1 conv1 weight",
//             reinterpret_cast<__nv_bfloat16*>(enc1->t_proj->W_->data), 
//             enc1->t_proj->W_->size / sizeof(__nv_bfloat16));
        
//         relu_inplace(x, size_t(s.B) * conv2->outC_ * s.H * s.W, st);
//             dump_chw("x after relu (x)", x, conv2->outC_, s.H, s.W, st);

//         cudaStreamSynchronize(st);

//         s.C = conv2->outC_;

//         dump_bf16_2("THE WEIGHTS AFTER THE SECOND CONVOLUTION PASS model encoder1 conv1 weight",
//             reinterpret_cast<__nv_bfloat16*>(enc1->t_proj->W_->data), 
//             enc1->t_proj->W_->size / sizeof(__nv_bfloat16));

//         // save skip
//         auto saved = std::make_shared<CudaBuffer>(size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16));
//         cudaMemcpyAsync(saved->data, x, saved->size, cudaMemcpyDeviceToDevice, st);
//         skips.push_back(saved);

//         // MaxPool downsample
//         pool->forward(x, s.B, s.C, s.H, s.W, reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
//         cudaMemcpyAsync(x, tmp->data, size_t(s.B) * s.C * (s.H/2) * (s.W/2) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, st);
//         s.H /= 2; 
//         s.W /= 2;
//         // channels already updated: s.C stays = conv2->outC_
        
//         dump_chw("=========== temb", temb, conv1->outC_, s.H, s.W, st);
//         dump_chw("=========== TBUF", reinterpret_cast<const __nv_bfloat16*>(tbuf->data), conv1->outC_, s.H, s.W, st);
//         dump_chw("x after downsample (x)", x, conv2->outC_, s.H, s.W, st);
//         std::cerr << "========================\n";

//         dump_bf16_2("THE WEIGHTS AFTER THE DOWNSAMPLE model encoder1 conv1 weight",
//             reinterpret_cast<__nv_bfloat16*>(enc1->t_proj->W_->data), 
//             enc1->t_proj->W_->size / sizeof(__nv_bfloat16));
//     }
// };

// // ────────────────────────────────────────────────────────────────────────────
// // Bottleneck Block (Conv‑Conv‑ReLU, no pooling)                              
// // ────────────────────────────────────────────────────────────────────────────
// class BottleneckBF16 {
// public:
//     // BottleneckBF16(int ch, int t_dim)
//     //     : conv1(ch, ch), conv2(ch, ch), t_proj(t_dim, ch) {}

//     std::shared_ptr<Conv2dBF16> conv1, conv2;
//     std::shared_ptr<LinearBF16> t_proj;

//     std::shared_ptr<CudaBuffer> tmp, tbuf;

//     BottleneckBF16() {};

//     void forward(__nv_bfloat16 *x, Shape &s, const __nv_bfloat16 *temb, cudaStream_t st)
//     {
//         // project time → bias
//         // ensure_size(tbuf, size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
//         if (!tbuf || tbuf->size < size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16)) {
//             tbuf = std::make_shared<CudaBuffer>(size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
//         }
//         for (int b = 0; b < s.B; ++b)
//             t_proj->forward(
//                 temb + b * t_proj->in_f,
//                 reinterpret_cast<__nv_bfloat16*>(tbuf->data) + b * conv1->outC_,
//                 st);
//         const __nv_bfloat16* tbias = reinterpret_cast<const __nv_bfloat16*>(tbuf->data);

//         // conv1
//         // ensure_size(tmp, size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16));
//         if (!tmp || tmp->size < size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16)) {
//             tmp = std::make_shared<CudaBuffer>(size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16));
//         }
        
//         conv1->forward(x, s.B, s.H, s.W,
//                       reinterpret_cast<__nv_bfloat16*>(tmp->data), st);
//         add_time_bias(reinterpret_cast<__nv_bfloat16*>(tmp->data), tbias,
//                       s.B, s.C, s.H, s.W, st);
//         relu_inplace(reinterpret_cast<__nv_bfloat16*>(tmp->data), size_t(s.B)*s.C*s.H*s.W, st);

//         // conv2 (back to x)
//         conv2->forward(reinterpret_cast<__nv_bfloat16*>(tmp->data), s.B, s.H, s.W,
//                       x, st);
//         add_time_bias(x, tbias, s.B, s.C, s.H, s.W, st);
//         relu_inplace(x, size_t(s.B)*s.C*s.H*s.W, st);
//     }
// };

// // ────────────────────────────────────────────────────────────────────────────
// // Decoder Block  (skip‑concat → Conv‑Conv‑ReLU → ConvT ×2)                   
// // ────────────────────────────────────────────────────────────────────────────
// class DecoderBlockBF16 {
// public:
//     int outC_;
//     std::shared_ptr<Conv2dBF16>      conv1, conv2;
//     std::shared_ptr<ConvTrans2dBF16> up;
//     std::shared_ptr<LinearBF16>      t_proj;

//     std::shared_ptr<CudaBuffer>      catBuf, tbuf;

//     DecoderBlockBF16(int outC): outC_(outC) {};

//     void forward(__nv_bfloat16 *&x, Shape &s, const std::shared_ptr<CudaBuffer> &skip,
//                  const __nv_bfloat16 *temb, cudaStream_t st)
//     {
//         // concat skip & x along channel dim → catBuf
//         const size_t skipElems = skip->size / sizeof(__nv_bfloat16);
//         int skipC = static_cast<int>(skipElems / (size_t(s.B) * s.H * s.W));
//         // ensure_size(catBuf, size_t(s.B) * (s.C + skipC) * s.H * s.W * sizeof(__nv_bfloat16));

//         if (!catBuf || catBuf->size < size_t(s.B) * (s.C + skipC) * s.H * s.W * sizeof(__nv_bfloat16)) {
//             catBuf = std::make_shared<CudaBuffer>(size_t(s.B) * (s.C + skipC) * s.H * s.W * sizeof(__nv_bfloat16));
//         }
//         // auto catBuf = std::make_shared<CudaBuffer>(size_t(s.B) * (s.C + skipC) * s.H * s.W * sizeof(__nv_bfloat16));
        
//         cudaMemcpyAsync(catBuf->data, skip->data, skip->size, cudaMemcpyDeviceToDevice, st);
//         cudaMemcpyAsync(reinterpret_cast<__nv_bfloat16*>(catBuf->data) + skipC * s.B * s.H * s.W,
//                         x, size_t(s.B) * s.C * s.H * s.W * sizeof(__nv_bfloat16),
//                         cudaMemcpyDeviceToDevice, st);

//         // time proj
//         // ensure_size(tbuf, size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
//         // auto tbuf = std::make_shared<CudaBuffer>(size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
//         if (!tbuf || tbuf->size < size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16)) {
//             tbuf = std::make_shared<CudaBuffer>(size_t(s.B) * conv1->outC_ * sizeof(__nv_bfloat16));
//         }
//         for (int b = 0; b < s.B; ++b)
//             t_proj->forward(
//                 temb + b * t_proj->in_f,
//                 reinterpret_cast<__nv_bfloat16*>(tbuf->data) + b * conv1->outC_,
//                 st);
//         const __nv_bfloat16* tbias = reinterpret_cast<const __nv_bfloat16*>(tbuf->data);

//         // conv1
//         conv1->forward(reinterpret_cast<__nv_bfloat16*>(catBuf->data), s.B, s.H, s.W, x, st);
//         add_time_bias(x, tbias, s.B, outC_, s.H, s.W, st);
//         relu_inplace(x, size_t(s.B)*outC_*s.H*s.W, st);

//         // conv2
//         conv2->forward(x, s.B, s.H, s.W, x, st);
//         add_time_bias(x, tbias, s.B, outC_, s.H, s.W, st);
//         relu_inplace(x, size_t(s.B)*outC_*s.H*s.W, st);

//         // upsample (ConvTranspose2d)
//         up->forward(x, s.B, s.H, s.W, x, st);
//         s.H *= 2; s.W *= 2; s.C = outC_;
//     }    
// };

// // ────────────────────────────────────────────────────────────────────────────
// // Full UNet                                                                      
// // ────────────────────────────────────────────────────────────────────────────
// class UNetBF16 {
//     cudaStream_t stream;
// public:
//     UNetBF16(int resolution, int t_dim = 128): img_res(resolution) {
//         // img_res(resolution), 
//         //   timeEmb(t_dim),
//         //   enc0(1,  64,  t_dim),
//         //   enc1(64, 128, t_dim),
//         //   bott(128,       t_dim),
//         //   dec0(256, 128, t_dim),   // 128(in) + 128(skip)
//         //   dec1(192,  64, t_dim),   // 128(in) +  64(skip)
//         //   out_conv(64, 1) 
//         checkCuda(cudaStreamCreate(&stream))
//     }

//     ~UNetBF16() {
//         checkCuda(cudaStreamDestroy(stream));
//     }

//     // // sub‑modules
//     // TimeEmbeddingBF16 timeEmb;
//     // EncoderBlockBF16  enc0, enc1;
//     // BottleneckBF16    bott;
//     // DecoderBlockBF16  dec0, dec1;
//     // Conv2dBF16        out_conv;

//     std::shared_ptr<TimeEmbeddingBF16> timeEmb;
//     std::shared_ptr<EncoderBlockBF16> enc0, enc1;
//     std::shared_ptr<BottleneckBF16> bott;
//     std::shared_ptr<DecoderBlockBF16> dec0, dec1;
//     std::shared_ptr<Conv2dBF16> out_conv1;

//     std::shared_ptr<CudaBuffer> tembBuf;

//     // ---------------------------------------------------------
//     // forward
//     // ---------------------------------------------------------
//     void forward(__nv_bfloat16 *x, const int32_t *t_host, int B, cudaStream_t st) {
//         // prepare time embedding
//         // ensure_size(tembBuf, size_t(B) * timeEmb->dim * sizeof(__nv_bfloat16));
//         tembBuf = std::make_shared<CudaBuffer>(size_t(B) * timeEmb->dim * sizeof(__nv_bfloat16));
        
//         timeEmb->forward(t_host, B, reinterpret_cast<__nv_bfloat16*>(tembBuf->data), st);\
//         cudaStreamSynchronize(st);
//         const __nv_bfloat16 *temb = reinterpret_cast<__nv_bfloat16*>(tembBuf->data);

//         Shape s{B, 1, img_res, img_res};
//         skips.clear(); skips.reserve(2);

//         dump_bf16_2("FIRST model encoder1 conv1 weight",
//             reinterpret_cast<__nv_bfloat16*>(enc1->t_proj->W_->data), 
//             enc1->t_proj->W_->size / sizeof(__nv_bfloat16));

//         // encoders
//         enc0->forward(x, s, temb, skips, st, 
//                      enc1);
//         cudaStreamSynchronize(st);

//         dump_bf16_2("SECOND model encoder1 conv1 weight",
//             reinterpret_cast<__nv_bfloat16*>(enc1->t_proj->W_->data), 
//             enc1->t_proj->W_->size / sizeof(__nv_bfloat16));

//         enc1->forward(x, s, temb, skips, st,
//                      enc1);
//         cudaStreamSynchronize(st);

//         // // // bottleneck
//         // // bott->forward(x, s, temb, st);

//         // // // decoders (in reverse skip order)
//         // // dec0->forward(x, s, skips.back(), temb, st); skips.pop_back();
//         // // dec1->forward(x, s, skips.back(), temb, st); skips.pop_back();

//         // // // final 1×1 conv, no bias add-time
//         // // out_conv->forward(x, s.B, s.H, s.W, x, st);
//     }

// private:
//     int img_res;

//     // scratch & skips
//     std::vector<std::shared_ptr<CudaBuffer>> skips;
// };

} // namespace dm
