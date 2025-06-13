#pragma once
// DiffusionLayers.cuh – Linear, Conv2d, MaxPool, LayerNorm (bf16), TimeEmbedding
#include "DiffusionHelper.cuh"
#include "DiffusionKernels.cuh"

#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"

using namespace dm;
using namespace kernels;

// ========== LinearBF16 =======================================================
class LinearBF16 {
public:
    int in_f, out_f;
    std::shared_ptr<CudaBuffer> W;  // column‑major [out_f, in_f]
    std::shared_ptr<CudaBuffer> b;  // [out_f]

    LinearBF16(int inFeat, int outFeat): in_f(inFeat), out_f(outFeat)
    {
        W = std::make_shared<CudaBuffer>(size_t(out_f) * in_f * sizeof(__nv_bfloat16));
        b = std::make_shared<CudaBuffer>(out_f * sizeof(__nv_bfloat16));
    }

    void forward(const __nv_bfloat16 *x,       // [in_f]
                 __nv_bfloat16 *y,             // [out_f]
                 cudaStream_t s) const
    {
        auto &blas = Handles::get().blas;  cublasSetStream(blas, s);
        const float alpha = 1.f, beta = 0.f;
        // y = W * x  ;  W column‑major (out_f×in_f)
        // // GEMM: (out_f×1) = (out_f×in_f) * (in_f×1)
        // cublasGemmEx(blas,
        //              CUBLAS_OP_N, CUBLAS_OP_N,
        //              out_f, 1, in_f,
        //              &alpha,
        //              W->data, CUDA_R_16BF, out_f,
        //              x,        CUDA_R_16BF, in_f,
        //              &beta,
        //              y,        CUDA_R_16BF, out_f,
        //              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        // // ==== row major ====
        cublasGemmEx(blas, 
                    CUBLAS_OP_T,  // <-- transpose W on-the-fly
                    CUBLAS_OP_N,
                    out_f, 1, in_f,
                    &alpha,
                    W->data, CUDA_R_16BF, in_f,   // lda = in_f
                    x,        CUDA_R_16BF, in_f,
                    &beta,
                    y,        CUDA_R_16BF, out_f,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        lin_add_bias<<<(out_f + 255) / 256, 256, 0, s>>>(y,
            static_cast<const __nv_bfloat16*>(b->data), out_f);
    }

    void forward32(const __nv_bfloat16 *x_bf16,   // [in_f]  (BF16)
                   float                *y_fp32,  // [out_f] (FP32)
                   cudaStream_t s) const {
        auto &blas = Handles::get().blas;  cublasSetStream(blas, s);

        const float alpha = 1.f, beta = 0.f;

        /*  GEMM:  y_fp32 = W_bf16^T  *  x_bf16   (accumulate in FP32) */
        cublasGemmEx(blas,
                    CUBLAS_OP_T,  CUBLAS_OP_N,
                    out_f, 1, in_f,
                    &alpha,
                    W->data,                CUDA_R_16BF,  in_f,   // **BF16 weights**
                    x_bf16,                 CUDA_R_16BF,  in_f,   // BF16 input
                    &beta,
                    y_fp32,                 CUDA_R_32F,   out_f,  // FP32 output
                    CUDA_R_32F,  CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        /* add bias in FP32 */
        const __nv_bfloat16* b_bf16 = static_cast<const __nv_bfloat16*>(b->data);
        for (int i = 0; i < out_f; ++i) {
            y_fp32[i] += __bfloat162float(b_bf16[i]);
        }
    }

    void forward32_fp32(const float *x_fp32,   // [in_f]  (FP32)
                       float *y_fp32,          // [out_f] (FP32)
                       cudaStream_t s) const {
        auto &blas = Handles::get().blas;  cublasSetStream(blas, s);

        const float alpha = 1.f, beta = 0.f;

        /*  GEMM:  y_fp32 = W_bf16^T  *  x_fp32   (accumulate in FP32) */
        cublasGemmEx(blas,
                    CUBLAS_OP_T,  CUBLAS_OP_N,
                    out_f, 1, in_f,
                    &alpha,
                    W->data,                CUDA_R_16BF,  in_f,   // **BF16 weights**
                    x_fp32,                 CUDA_R_32F,   in_f,   // FP32 input
                    &beta,
                    y_fp32,                 CUDA_R_32F,   out_f,  // FP32 output
                    CUDA_R_32F,  CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        /* add bias in FP32 */
        const __nv_bfloat16* b_bf16 = static_cast<const __nv_bfloat16*>(b->data);
        for (int i = 0; i < out_f; ++i) {
            y_fp32[i] += __bfloat162float(b_bf16[i]);
        }
    }

    void set_weights(std::shared_ptr<CudaBuffer> w, std::shared_ptr<CudaBuffer> b_) {
        W = std::move(w);
        b = std::move(b_);
    }

    __nv_bfloat16*       weights()       { return static_cast<__nv_bfloat16*>(W->data); }
    const __nv_bfloat16* weights() const { return static_cast<const __nv_bfloat16*>(W->data); }

    __nv_bfloat16*       bias()       { return static_cast<__nv_bfloat16*>(b->data); }
    const __nv_bfloat16* bias() const { return static_cast<const __nv_bfloat16*>(b->data); }
};

// -----------------------------------------------------------------------------
// 2.  Conv2dBF16  (same as before, but bias pointer cached)
// -----------------------------------------------------------------------------
class Conv2dBF16 {
public:
    int in, out;
    std::shared_ptr<CudaBuffer> W;
    std::shared_ptr<CudaBuffer> b;

    Conv2dBF16(int inC, int outC): in(inC), out(outC)
    {
        // allocate weight buffer
        W = std::make_shared<CudaBuffer>(size_t(out) * in * 3 * 3 * sizeof(__nv_bfloat16));
        b = std::make_shared<CudaBuffer>(size_t(out) * sizeof(__nv_bfloat16));

        auto &dnn = Handles::get().dnn;

        /* 1. filter desc */
        cudnnCreateFilterDescriptor(&w_desc);
        cudnnSetFilter4dDescriptor(w_desc,
            CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW,
            out, in, 3, 3);

        /* 2. convolution desc */
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnSetConvolution2dDescriptor(conv_desc,
            /*padH,W=*/1, 1,
            /*strideH,W=*/1, 1,
            /*dilationH,W=*/1, 1,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT); // BFLOAT16
        cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

        /* 3. tensor descs (created here, sizes filled in forward()) */
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateTensorDescriptor(&y_desc);

        /* pick default forward algo */
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }

    void forward(const __nv_bfloat16 *x, 
                 int B, int H, int W_,
                 __nv_bfloat16 *y, cudaStream_t s)
    {
        auto &dnn = Handles::get().dnn; cudnnSetStream(dnn,s);
        cudnnSetTensor4dDescriptor(x_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,
                                   B,in,H,W_);
        cudnnSetTensor4dDescriptor(y_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,
                                   B,out,H,W_);
        size_t wsB=0;
        cudnnGetConvolutionForwardWorkspaceSize(dnn,x_desc,w_desc,
                                                conv_desc,y_desc,algo,&wsB);
        CudaBuffer ws(wsB);
        const float alpha=1.f, beta=0.f;
        cudnnConvolutionForward(dnn,&alpha,
                                x_desc,x,
                                w_desc,W->data,
                                conv_desc,algo,
                                ws.data,wsB,
                                &beta,
                                y_desc,y);

        conv_add_bias<<<(size_t(B)*out*H*W_+255)/256,256,0,s>>>(y,static_cast<const __nv_bfloat16*>(b->data),B,out,H*W_);
    }

    void set_weights(std::shared_ptr<CudaBuffer> w_){ W=std::move(w_); }
    void set_bias   (std::shared_ptr<CudaBuffer> b_){ b=std::move(b_); }

    __nv_bfloat16* weights() { return static_cast<__nv_bfloat16*>(W->data); }
    const __nv_bfloat16* weights() const { return static_cast<const __nv_bfloat16*>(W->data); }

    __nv_bfloat16* bias() { return static_cast<__nv_bfloat16*>(b->data); }
    const __nv_bfloat16* bias() const { return static_cast<const __nv_bfloat16*>(b->data); }

    int out_channels() const { return out; }

private:
    cudnnTensorDescriptor_t x_desc{},y_desc{};
    cudnnFilterDescriptor_t w_desc{};
    cudnnConvolutionDescriptor_t conv_desc{};
    cudnnConvolutionFwdAlgo_t algo{};
};

// -----------------------------------------------------------------------------
// 3.  ConvTranspose2dBF16 (2×2, stride-2) – for decoder up-sampling
// -----------------------------------------------------------------------------
class ConvTrans2dBF16 {
public:
    int in{1},out{1};
    std::shared_ptr<CudaBuffer> W,b;

    ConvTrans2dBF16(int inC, int outC): in(inC), out(outC)
    {
        // 1. allocate  (Cout, Cin, 2, 2)
        W = std::make_shared<CudaBuffer>(size_t(out) * in * 4 * sizeof(__nv_bfloat16));
        b = std::make_shared<CudaBuffer>(size_t(out) * sizeof(__nv_bfloat16));

        auto &dnn = Handles::get().dnn;

        // 2. filter descriptor (Cout, Cin, kH, kW)
        cudnnCreateFilterDescriptor(&w_desc);
        cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_BFLOAT16,
                                CUDNN_TENSOR_NCHW,
                                out, in, 2, 2);            // ← fixed order

        // 3. convolution descriptor: stride=2, pad=0
        cudnnCreateConvolutionDescriptor(&deconv_desc);
        cudnnSetConvolution2dDescriptor(deconv_desc,
            /*padH,W=*/0, 0,
            /*strH,W=*/2, 2,
            /*dilH,W=*/1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_BFLOAT16); // BFLOAT16

        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateTensorDescriptor(&y_desc);

        algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;   // fast for small kernels
    }

    ~ConvTrans2dBF16(){
        cudnnDestroyTensorDescriptor(x_desc); cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyFilterDescriptor(w_desc);  cudnnDestroyConvolutionDescriptor(deconv_desc);
    }

    void forward(const __nv_bfloat16 *x, int B, int H, int W_,
                 __nv_bfloat16 *y, cudaStream_t s)
    {
        auto &dnn = Handles::get().dnn;  cudnnSetStream(dnn, s);

        cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_BFLOAT16, B, in,  H,  W_);
        cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                CUDNN_DATA_BFLOAT16, B, out, H*2, W_*2);

        size_t wsB = 0;
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            dnn, w_desc, x_desc, deconv_desc, y_desc, algo, &wsB);

        CudaBuffer ws(wsB);

        const float alpha = 1.f, beta = 0.f;
        cudnnConvolutionBackwardData(
            dnn, &alpha,
            w_desc, W->data,          // filter
            x_desc, x,                // dy
            deconv_desc, algo,
            ws.data, wsB,
            &beta,
            y_desc, y);               // dx  (our upsampled output)

        conv_add_bias<<<(size_t(B) * out * H * 2 * W_ * 2 + 255) / 256, 256, 0, s>>>(
                y, static_cast<const __nv_bfloat16*>(b->data), B, out, (H * 2) * (W_ * 2));
    }

    void set_weights(std::shared_ptr<CudaBuffer> w_){ W=std::move(w_); }
    void set_bias   (std::shared_ptr<CudaBuffer> b_){ b=std::move(b_); }

    __nv_bfloat16* weights() { return static_cast<__nv_bfloat16*>(W->data); }
    const __nv_bfloat16* weights() const { return static_cast<const __nv_bfloat16*>(W->data); }
    __nv_bfloat16* bias() { return static_cast<__nv_bfloat16*>(b->data); }
    const __nv_bfloat16* bias() const { return static_cast<const __nv_bfloat16*>(b->data); }


private:
    cudnnTensorDescriptor_t x_desc{},y_desc{};
    cudnnFilterDescriptor_t w_desc{};
    cudnnConvolutionDescriptor_t deconv_desc{};
    cudnnConvolutionBwdDataAlgo_t algo{};
};

// ========== MaxPool2d (2×2, stride‑2) ========================================
class MaxPool2dBF16 {
public:
    MaxPool2dBF16() {
        cudnnCreatePoolingDescriptor(&p_desc);
        cudnnSetPooling2dDescriptor(p_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                    2, 2, 0, 0, 2, 2);
    }
    ~MaxPool2dBF16() { cudnnDestroyPoolingDescriptor(p_desc); }
    void forward(const __nv_bfloat16 *x, int B, int C, int H, int W_,
                 __nv_bfloat16 *y, cudaStream_t s) {
        auto &dnn = Handles::get().dnn; cudnnSetStream(dnn, s);
        cudnnTensorDescriptor_t ti, to; cudnnCreateTensorDescriptor(&ti); cudnnCreateTensorDescriptor(&to);
        cudnnSetTensor4dDescriptor(ti, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, B, C, H,   W_);
        cudnnSetTensor4dDescriptor(to, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, B, C, H/2, W_/2);
        const float alpha = 1.f, beta = 0.f;
        cudnnPoolingForward(dnn, p_desc, &alpha, ti, x, &beta, to, y);
        cudnnDestroyTensorDescriptor(ti); cudnnDestroyTensorDescriptor(to);
    }
private:
    cudnnPoolingDescriptor_t p_desc{};
};

class TimeEmbeddingBF16 {
public:
    const int dim;
    LinearBF16 proj1, proj2;

    CudaBuffer tmp_bf16{0};
    CudaBuffer out_fp32{0};                 //  <-- new (stores bias in FP32)
    CudaBuffer tmp_fp32{0};

    explicit TimeEmbeddingBF16(int d, int mlp = -1)
        : dim(d), proj1(d, mlp < 0 ? 4 * d : mlp),
          proj2(mlp < 0 ? 4 * d : mlp, d) {}

    /**
     * t_host – host int32 array of length B
     * out     – device bf16 [B, dim]
     */
    void forward(const int32_t *t_host, int B,
                 __nv_bfloat16 *out_bf16,     // destination
                 cudaStream_t  s) {
        /* ---------- sinusoid in FP32 ---------- */
        std::vector<float> h(B * dim);
        int half = dim / 2;
        for (int b = 0; b < B; ++b) {
            float tt = float(t_host[b]);
            for (int i = 0; i < half; ++i) {
                float f = expf(-logf(10000.f) * i / (half - 1));
                h[b * dim + i]        = sinf(tt * f);
                h[b * dim + half + i] = cosf(tt * f);
            }
            if (dim & 1) h[b * dim + dim - 1] = 0.f;
        }
        ensure_size(out_fp32, h.size() * sizeof(float));
        cudaMemcpyAsync(out_fp32.data, h.data(), out_fp32.size,
                        cudaMemcpyHostToDevice, s);

        ensure_size(tmp_bf16, out_fp32.size / 2);            // same #elements, half the bytes
        dm::fp32_to_bf16(reinterpret_cast<float*>(out_fp32.data),
                         reinterpret_cast<__nv_bfloat16*>(tmp_bf16.data),
                         size_t(B) * dim, s);

        /* ---------- proj-1 ---------- */
        ensure_size(tmp_fp32, size_t(B) * proj1.out_f * sizeof(float));
        for (int b = 0; b < B; ++b)
            proj1.forward32(                               // FP32 out
                reinterpret_cast<__nv_bfloat16*>(tmp_bf16.data) + b * dim,
                reinterpret_cast<float*>(tmp_fp32.data)    + b * proj1.out_f, s);

        dm::silu_inplace32(                         // ← new FP32 SiLU
                reinterpret_cast<float*>(tmp_fp32.data),
                size_t(B) * proj1.out_f, s);

        /* ---------- proj-2 ---------- */
        ensure_size(tmp_bf16, size_t(B) * proj1.out_f * sizeof(__nv_bfloat16));
        dm::fp32_to_bf16(reinterpret_cast<float*>(tmp_fp32.data),
                         reinterpret_cast<__nv_bfloat16*>(tmp_bf16.data),
                         size_t(B) * proj1.out_f, s);

        for (int b = 0; b < B; ++b)
            proj2.forward32(
                reinterpret_cast<__nv_bfloat16*>(tmp_bf16.data) + b * proj1.out_f,
                reinterpret_cast<float*>(out_fp32.data) + b * dim, s);

        /* ---------- one cast to BF16 ---------- */
        dm::fp32_to_bf16(reinterpret_cast<float*>(out_fp32.data),
                        out_bf16, size_t(B) * dim, s);
    }

    void set_weights(std::shared_ptr<CudaBuffer> w1, std::shared_ptr<CudaBuffer> b1,
                     std::shared_ptr<CudaBuffer> w2, std::shared_ptr<CudaBuffer> b2) {
        proj1.set_weights(std::move(w1), std::move(b1));
        proj2.set_weights(std::move(w2), std::move(b2));
    }
};

