#pragma once
// DiffusionLayers.cuh – Linear, Conv2d, MaxPool, LayerNorm (bf16), TimeEmbedding
#include "DiffusionHelper.cuh"

#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"

using namespace dm;
using namespace kernels;

// ========== LinearBF16 =======================================================
class LinearBF16 {
public:
    const int in_f, out_f;
    std::shared_ptr<const CudaBuffer> W_;  // column‑major [out_f, in_f]
    std::shared_ptr<const CudaBuffer> b_;  // [out_f]

    LinearBF16(int inFeat, int outFeat): in_f(inFeat), out_f(outFeat)
    {
        W_ = std::make_shared<const CudaBuffer>(size_t(out_f) * in_f * sizeof(__nv_bfloat16));
        b_ = std::make_shared<const CudaBuffer>(size_t(out_f) * sizeof(__nv_bfloat16));
    }

    void forward(const __nv_bfloat16 *x,       // [in_f]
                 __nv_bfloat16 *y,             // [out_f]
                 cudaStream_t s) const 
    {
        auto &blas = Handles::get().blas;  cublasSetStream(blas, s);

        // dump_chw("=========== weights", reinterpret_cast<__nv_bfloat16*>(W_->data), out_f, 128, 128, s);
        // dump_chw("=========== bias", reinterpret_cast<__nv_bfloat16*>(b_->data), out_f, 128, 128, s);
        // dump_chw("=========== tbias (random)", y, out_f, 128, 128, s);
        // dump_chw("=========== temb  (notran)", x, out_f, 128, 128, s);
        
        // y = W * x  ;  W column‑major (out_f×in_f)
        // GEMM: (out_f×1) = (out_f×in_f) * (in_f×1)
        const float alpha = 1.f, beta = 0.f;
        cublasGemmEx(blas, 
                    CUBLAS_OP_T,  // <-- transpose W on-the-fly
                    CUBLAS_OP_N,
                    out_f, 1, in_f,
                    &alpha,
                    W_->data,  CUDA_R_16BF, in_f,   // lda = in_f
                    x,        CUDA_R_16BF,  in_f,
                    &beta,
                    y,        CUDA_R_16BF, out_f,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        lin_add_bias<<<(out_f + 255) / 256, 256, 0, s>>>(y, static_cast<const __nv_bfloat16*>(b_->data), out_f);
        // add_time_bias(y, static_cast<const __nv_bfloat16*>(b_->data),
        //               1, out_f, 1, 1, s);  // B=1, C=out_f, H=W=1
    }

    // void forward32(const __nv_bfloat16 *x_bf16,   // [in_f] (BF16)
    //             float               *y_fp32,      // [out_f] (FP32)
    //             cudaStream_t         s) const
    // {
    //     auto &blas = Handles::get().blas;  cublasSetStream(blas, s);
    //     const float alpha = 1.f, beta = 0.f;

    //     /* GEMM:   y_fp32 = W_bf16ᵀ · x_bf16   (accumulate in FP32) */
    //     cublasGemmEx(blas,
    //                 CUBLAS_OP_N, CUBLAS_OP_N,
    //                 out_f, 1, in_f,
    //                 &alpha,
    //                 W_->data, CUDA_R_16BF, in_f,   // BF16 weights
    //                 x_bf16,  CUDA_R_16BF,  in_f,   // BF16 input
    //                 &beta,
    //                 y_fp32,  CUDA_R_32F,  out_f,  // FP32 output
    //                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    //     /* add BF16 bias **in FP32** */
    //     int tpb = 256, blk = (out_f + tpb - 1) / tpb;
    //     kernels::add_bias_fp32_kernel<<<blk, tpb, 0, s>>>(
    //         y_fp32,
    //         static_cast<const __nv_bfloat16*>(b_->data),
    //         out_f);
    // }
};

// -----------------------------------------------------------------------------
// 2.  Conv2dBF16  (same as before, but bias pointer cached)
// -----------------------------------------------------------------------------

class Conv2dBF16 {
public:
    int inC_, outC_;
    std::shared_ptr<const CudaBuffer> W_, b_;
    std::shared_ptr<CudaBuffer> ws_;               // <- ***persistent scratch***
    
    /* parameters are allocated & owned by the layer */
    Conv2dBF16(int inC, int outC): inC_(inC), outC_(outC)
    {
        W_ = std::make_shared<const CudaBuffer>(size_t(outC) * inC * 3 * 3 * sizeof(__nv_bfloat16));
        b_ = std::make_shared<const CudaBuffer>(size_t(outC) * sizeof(__nv_bfloat16));

        cudnnCreateFilterDescriptor(&wDesc_);
        cudnnSetFilter4dDescriptor(wDesc_, CUDNN_DATA_BFLOAT16,
                                   CUDNN_TENSOR_NCHW, outC_, inC_, 3, 3);

        cudnnCreateConvolutionDescriptor(&convDesc_);
        cudnnSetConvolution2dDescriptor(convDesc_,
                                        1,1,   // pad
                                        1,1,   // stride
                                        1,1,   // dilation
                                        CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT);
        cudnnSetConvolutionMathType(convDesc_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

        cudnnCreateTensorDescriptor(&xDesc_); 
        cudnnCreateTensorDescriptor(&yDesc_);
        algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }

    ~Conv2dBF16() noexcept {
        cudnnDestroyTensorDescriptor(xDesc_);
        cudnnDestroyTensorDescriptor(yDesc_);
        cudnnDestroyFilterDescriptor(wDesc_);
        cudnnDestroyConvolutionDescriptor(convDesc_);
    }

    //------------------------------------------------ forward
    void forward(const __nv_bfloat16 *x,
                 int N, int H, int W,
                 __nv_bfloat16 *y,
                 cudaStream_t stream)
    {
        auto &dnn = Handles::get().dnn;
        cudnnSetStream(dnn, stream);

        cudnnSetTensor4dDescriptor(xDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
                                   N, inC_, H, W);

        int nOut,cOut,hOut,wOut;
        cudnnGetConvolution2dForwardOutputDim(convDesc_, xDesc_, wDesc_,
                                              &nOut,&cOut,&hOut,&wOut);
        if (nOut!=N || cOut!=outC_)
            throw std::runtime_error("Conv2dBF16: channel mismatch");

        cudnnSetTensor4dDescriptor(yDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
                                   nOut,cOut,hOut,wOut);

        // persistent workspace (grown once, never freed)
        size_t need=0;
        cudnnGetConvolutionForwardWorkspaceSize(dnn, xDesc_, wDesc_, convDesc_,
                                                yDesc_, algo_, &need);
        // ensure_size(ws_, need);
        if (!ws_ || ws_->size < need)
            ws_ = std::make_shared<CudaBuffer>(need);
        // auto ws_ = std::make_shared<CudaBuffer>(need);

        const float alpha=1.f, beta=0.f;
        cudnnConvolutionForward(dnn,&alpha,
                                xDesc_, x,
                                wDesc_, W_->data,
                                convDesc_, algo_,
                                ws_->data, ws_->size,
                                &beta,
                                yDesc_, y);

        add_bias_conv(y, static_cast<const __nv_bfloat16*>(b_->data), N, outC_, H, W, stream);
        // add_bias_conv_kernel<<<128,128,0,stream>>>(
        //     y,
        //     static_cast<const __nv_bfloat16*>(b_->data),
        //     size_t(nOut), cOut,           // B , C
        //     hOut, wOut);                  // H , W

        cudaGetLastError();
    }
private:
    cudnnFilterDescriptor_t      wDesc_{};  
    cudnnConvolutionDescriptor_t convDesc_{};
    cudnnTensorDescriptor_t      xDesc_{}, yDesc_{};
    cudnnConvolutionFwdAlgo_t    algo_{};
};

// -----------------------------------------------------------------------------
// 3.  ConvTranspose2dBF16 (2×2, stride-2) – for decoder up-sampling
// -----------------------------------------------------------------------------
// ConvTranspose2dBF16 – writes straight into y, pure-device scratch
class ConvTrans2dBF16 {
public:
    int in, out;

    // filter + bias (owned)
    std::shared_ptr<const CudaBuffer> W_, b_;

    // pure‐device workspace
    void*   ws_data{nullptr};
    size_t  ws_bytes{0};

    // cuDNN descriptors
    cudnnTensorDescriptor_t        x_desc{}, y_desc{};
    cudnnFilterDescriptor_t        w_desc{};
    cudnnConvolutionDescriptor_t   deconv_desc{};
    cudnnConvolutionBwdDataAlgo_t  algo{CUDNN_CONVOLUTION_BWD_DATA_ALGO_0};

    explicit ConvTrans2dBF16(int inC, int outC)
      : in(inC), out(outC)
    {
        // 1) allocate filter [Cin, Cout, 2, 2]
        W_ = std::make_shared<const CudaBuffer>(
                size_t(in) * out * 2 * 2 * sizeof(__nv_bfloat16));
        b_ = std::make_shared<const CudaBuffer>(
                size_t(out) * sizeof(__nv_bfloat16));

        auto &dnn = Handles::get().dnn;

        // 2) filter descriptor: K==in, C==out
        cudnnCreateFilterDescriptor(&w_desc);
        cudnnSetFilter4dDescriptor(
            w_desc,
            CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW,
            /*K=*/in, /*C=*/out, /*kH=*/2, /*kW=*/2);

        // 3) convolution descriptor: stride=2, pad=0
        cudnnCreateConvolutionDescriptor(&deconv_desc);
        cudnnSetConvolution2dDescriptor(
            deconv_desc,
            /*padH=*/0, /*padW=*/0,
            /*strideH=*/2, /*strideW=*/2,
            /*dilationH=*/1, /*dilationW=*/1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        cudnnSetConvolutionMathType(
            deconv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

        // 4) tensor descriptors (empty for now)
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateTensorDescriptor(&y_desc);
    }

    ~ConvTrans2dBF16() noexcept {
        if (ws_data) cudaFree(ws_data);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyConvolutionDescriptor(deconv_desc);
    }

    void forward(const __nv_bfloat16 *x,
                 int B, int H, int W,
                 __nv_bfloat16       *y,
                 cudaStream_t         s)
    {
        auto &dnn = Handles::get().dnn;
        cudnnSetStream(dnn, s);

        // set input/output shapes
        cudnnSetTensor4dDescriptor(
            x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
            /*N=*/B, /*C=*/in, /*H=*/H, /*W=*/W);
        cudnnSetTensor4dDescriptor(
            y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
            /*N=*/B, /*C=*/out, /*H=*/H*2, /*W=*/W*2);

        // pick best algo
        cudnnConvolutionBwdDataAlgoPerf_t perf;
        int found=0;
        cudnnFindConvolutionBackwardDataAlgorithm(
            dnn, w_desc, x_desc, deconv_desc, y_desc,
            /*requestAlgoCount=*/1, &found, &perf);
        algo = perf.algo;

        // workspace size
        size_t want;
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            dnn, w_desc, x_desc, deconv_desc, y_desc,
            algo, &want);

        // grow device workspace if needed
        if (want > ws_bytes) {
            if (ws_data) cudaFree(ws_data);
            cudaMalloc(&ws_data, want);
            ws_bytes = want;
        }

        // run deconv→y
        const float alpha=1.f, beta=0.f;
        cudnnConvolutionBackwardData(
            dnn, &alpha,
            w_desc, W_->data,   // filter
            x_desc, x,          // dy
            deconv_desc, algo,
            ws_data, ws_bytes,
            &beta,
            y_desc, y);         // dx == output

        // add per-channel bias in-place
        add_bias_conv(y,
                     static_cast<const __nv_bfloat16*>(b_->data),
                     /*B=*/B, /*C=*/out,
                     /*H=*/H*2, /*W=*/W*2,
                     s);

        // Ensure all operations are complete before next use
        cudaStreamSynchronize(s);
        cudaGetLastError();
    }
};

// class ConvTrans2dBF16 {
// public:
//     int in, out;
//     std::shared_ptr<const CudaBuffer> W_, b_;
//     std::shared_ptr<CudaBuffer>       ws_;
//     std::shared_ptr<CudaBuffer>       tmp_;  // Temporary buffer for output

//     explicit ConvTrans2dBF16(int inC, int outC) : in(inC), out(outC)
//     {
//         /* 1.  Allocate filter in  (Cin, Cout, kH, kW) order  */
//         W_ = std::make_shared<const CudaBuffer>(
//                 size_t(in) * out * 4 * sizeof(__nv_bfloat16));       // 2×2
//         b_ = std::make_shared<const CudaBuffer>(
//                 size_t(out) * sizeof(__nv_bfloat16));

//         auto &dnn = Handles::get().dnn;

//         /* 2.  Descriptors -------------------------------------------------- */
//         cudnnCreateFilterDescriptor(&w_desc);
//         cudnnSetFilter4dDescriptor(w_desc,            // K == in, C == out
//             CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW,
//             in, out, 2, 2);

//         cudnnCreateConvolutionDescriptor(&deconv_desc);
//         cudnnSetConvolution2dDescriptor(
//             deconv_desc,
//             0, 0,          // padH, padW
//             2, 2,          // stride
//             1, 1,          // dilation
//             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
//         cudnnSetConvolutionMathType(
//             deconv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

//         cudnnCreateTensorDescriptor(&x_desc);
//         cudnnCreateTensorDescriptor(&y_desc);
//         algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;     // default
//     }

//     ~ConvTrans2dBF16() {
//         cudnnDestroyTensorDescriptor(x_desc);
//         cudnnDestroyTensorDescriptor(y_desc);
//         cudnnDestroyFilterDescriptor(w_desc);
//         cudnnDestroyConvolutionDescriptor(deconv_desc);
//     }

//     void forward(const __nv_bfloat16 *x, int B, int H, int W,
//                  __nv_bfloat16 *y, cudaStream_t s)
//     {
//         auto &dnn = Handles::get().dnn;
//         cudnnSetStream(dnn, s);

//         /* 3.  Tensor descriptors  */
//         cudnnSetTensor4dDescriptor(
//             x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
//             B, in, H, W);
//         cudnnSetTensor4dDescriptor(
//             y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
//             B, out, H * 2, W * 2);

//         /* 4.  Workspace selection  */
//         cudnnConvolutionBwdDataAlgoPerf_t perf;
//         int nAlgo;
//         cudnnFindConvolutionBackwardDataAlgorithm(
//             dnn, w_desc, x_desc, deconv_desc, y_desc,
//             1, &nAlgo, &perf);
//         algo = perf.algo;

//         size_t wsBytes = 0;
//         cudnnGetConvolutionBackwardDataWorkspaceSize(
//             dnn, w_desc, x_desc, deconv_desc, y_desc, algo, &wsBytes);
//         if (!ws_ || ws_->size < wsBytes)
//             ws_ = std::make_shared<CudaBuffer>(wsBytes);

//         /* 5.  Allocate temporary output buffer */
//         size_t tmpBytes = size_t(B) * out * (H * 2) * (W * 2) * sizeof(__nv_bfloat16);
//         if (!tmp_ || tmp_->size < tmpBytes)
//             tmp_ = std::make_shared<CudaBuffer>(tmpBytes);

//         /* 6.  Actual compute  */
//         const float alpha = 1.f, beta = 0.f;
//         cudnnConvolutionBackwardData(
//             dnn, &alpha,
//             w_desc, W_->data,
//             x_desc, x,
//             deconv_desc, algo,
//             ws_->data, ws_->size,
//             &beta,
//             y_desc, reinterpret_cast<__nv_bfloat16*>(tmp_->data));

//         /* 7.  Add per-channel bias  */
//         add_bias_conv(reinterpret_cast<__nv_bfloat16*>(tmp_->data), 
//                      static_cast<const __nv_bfloat16*>(b_->data),
//                      B, out, H * 2, W * 2, s);

//         /* 8.  Copy result to output and ensure completion */
//         cudaMemcpyAsync(y, tmp_->data, tmpBytes, cudaMemcpyDeviceToDevice, s);
//         cudaStreamSynchronize(s);  // Ensure the operation is complete before next use
//     }

// private:
//     cudnnTensorDescriptor_t        x_desc{}, y_desc{};
//     cudnnFilterDescriptor_t        w_desc{};
//     cudnnConvolutionDescriptor_t   deconv_desc{};
//     cudnnConvolutionBwdDataAlgo_t  algo{};
// };

// ========== MaxPool2d (2×2, stride‑2) ========================================
class MaxPool2dBF16 {
public:
    MaxPool2dBF16() {
        cudnnCreatePoolingDescriptor(&p_desc);
        cudnnSetPooling2dDescriptor(p_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                    2, 2, 0, 0, 2, 2);
        cudnnCreateTensorDescriptor(&ti_desc);
        cudnnCreateTensorDescriptor(&to_desc);
    }
    
    ~MaxPool2dBF16() { 
        cudnnDestroyPoolingDescriptor(p_desc);
        cudnnDestroyTensorDescriptor(ti_desc);
        cudnnDestroyTensorDescriptor(to_desc);
    }
    
    void forward(const __nv_bfloat16 *x, int B, int C, int H, int W_,
                 __nv_bfloat16 *y, cudaStream_t s) {
        auto &dnn = Handles::get().dnn; 
        cudnnSetStream(dnn, s);
        
        cudnnSetTensor4dDescriptor(ti_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, B, C, H, W_);
        cudnnSetTensor4dDescriptor(to_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, B, C, H/2, W_/2);
        
        const float alpha = 1.f, beta = 0.f;
        cudnnPoolingForward(dnn, p_desc, &alpha, ti_desc, x, &beta, to_desc, y);
    }

private:
    cudnnPoolingDescriptor_t p_desc{};
    cudnnTensorDescriptor_t ti_desc{}, to_desc{};
};
// class MaxPool2dBF16 {
// public:
//     MaxPool2dBF16() {
//         cudnnCreatePoolingDescriptor(&p_desc);
//         cudnnSetPooling2dDescriptor(p_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
//                                     2, 2, 0, 0, 2, 2);
//     }
//     ~MaxPool2dBF16() { cudnnDestroyPoolingDescriptor(p_desc); }
//     void forward(const __nv_bfloat16 *x, int B, int C, int H, int W_,
//                  __nv_bfloat16 *y, cudaStream_t s) {
//         auto &dnn = Handles::get().dnn; cudnnSetStream(dnn, s);
//         ti, to; cudnnCreateTensorDescriptor(&ti); cudnnCreateTensorDescriptor(&to);
//         cudnnSetTensor4dDescriptor(ti, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, B, C, H,   W_);
//         cudnnSetTensor4dDescriptor(to, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, B, C, H/2, W_/2);
//         const float alpha = 1.f, beta = 0.f;
//         cudnnPoolingForward(dnn, p_desc, &alpha, ti, x, &beta, to, y);
//         cudnnDestroyTensorDescriptor(ti); cudnnDestroyTensorDescriptor(to);
//     }
// private:
//     cudnnPoolingDescriptor_t p_desc{};
//     cudnnTensorDescriptor_t ti{}, to{};
// };

class TimeEmbeddingBF16 {
public:
    const int dim;
    std::shared_ptr<LinearBF16> proj1, proj2;
    std::shared_ptr<CudaBuffer> tmp;

    // CudaBuffer out_fp32{0};                 //  <-- new (stores bias in FP32)
    // CudaBuffer tmp_fp32{0};

    // proj1(d, mlp < 0 ? 4 * d : mlp),
    // proj2(mlp < 0 ? 4 * d : mlp, d) 
    TimeEmbeddingBF16(int d, int mlp = -1): dim(d) {}

    void forward(const int32_t *t_host, int B,
                 __nv_bfloat16 *out,     // [B, dim]  (device)
                 cudaStream_t  s)
    {
        /* ---------- 1. raw sinusoid on host (FP32) ---------- */
        std::vector<__nv_bfloat16> h(B * dim);
        const int half = dim / 2;

        for (int b = 0; b < B; ++b) {
            float tt = float(t_host[b]);
            for (int i = 0; i < half; ++i) {
                float f = expf(-logf(10000.f) * i / (half - 1));
                h[b * dim + i]        = __float2bfloat16(sinf(tt * f));
                h[b * dim + half + i] = __float2bfloat16(cosf(tt * f));
            }
            if (dim & 1) h[b * dim + dim - 1] = __float2bfloat16(0.f);
        }
        checkCuda(cudaMemcpyAsync(out, h.data(),
                                  h.size() * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice, s));

        /* ---------- 2. proj-1 → SiLU → proj-2 --------------- */
        // ensure_size(tmp, size_t(B) * proj1->out_f * sizeof(__nv_bfloat16));
        if (!tmp || tmp->size < size_t(B) * proj1->out_f * sizeof(__nv_bfloat16))
            tmp = std::make_shared<CudaBuffer>(size_t(B) * proj1->out_f * sizeof(__nv_bfloat16));
        // auto tmp = std::make_shared<CudaBuffer>(size_t(B) * proj1->out_f * sizeof(__nv_bfloat16));

        for (int b = 0; b < B; ++b)
            proj1->forward(out + b * dim,
                          reinterpret_cast<__nv_bfloat16*>(tmp->data) + b * proj1->out_f,
                          s);

        silu_inplace(reinterpret_cast<__nv_bfloat16*>(tmp->data),
                     size_t(B) * proj1->out_f, s);

        for (int b = 0; b < B; ++b)
            proj2->forward(reinterpret_cast<__nv_bfloat16*>(tmp->data) + b * proj1->out_f,
                          out + b * dim, s);
    }

    private:
};

