#pragma once
// DiffusionLayers.cuh – Linear, Conv2d, MaxPool, LayerNorm (bf16), TimeEmbedding
#include "DiffusionHelper.cuh"
#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"

// ---------------- LinearBF16 -----------------
class LinearBF16 {
public:
    std::shared_ptr<CudaBuffer> W, b; int in_f, out_f;

    LinearBF16(int in_features,int out_features,bool bias=true): in_f(in_features), out_f(out_features) {
        W=std::make_shared<CudaBuffer>(size_t(out_f)*in_f*sizeof(__nv_bfloat16));
        if(bias) b=std::make_shared<CudaBuffer>(out_f*sizeof(__nv_bfloat16));
    }

    void forward(const __nv_bfloat16 *x,__nv_bfloat16 *y,cudaStream_t s) const {
        auto &h=dm_common::Handles::instance().blas; cublasSetStream(h,s);
        
        const float a=1.f,beta=0.f;
        checkCuda(cublasGemmEx(h,
                               CUBLAS_OP_N,CUBLAS_OP_N,
                               out_f,1,in_f,&a,
                               W->data,CUDA_R_16BF,out_f,x,CUDA_R_16BF,in_f,&beta,y,CUDA_R_16BF,out_f,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if(b) add_bias<<<(out_f+255)/256,256,0,s>>>(y,(__nv_bfloat16*)b->data,out_f); }

private: 
    static __global__ void add_bias(__nv_bfloat16 *y,const __nv_bfloat16 *b,int n) {
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i < n) {
            y[i] = f2bf(bf2f(y[i])+bf2f(b[i]));
        }
    }
};

// --------------- Conv2dBF16 -------------------
class Conv2dBF16 {
public:
    Conv2dBF16(int in_c,int out_c);

    ~Conv2dBF16();

    void forward(const __nv_bfloat16 *x,int B,int H,int W,const __nv_bfloat16 *bias,__nv_bfloat16 *y,cudaStream_t s);
private:
    int in,out; 
    std::shared_ptr<CudaBuffer> W, ws;
    size_t ws_bytes{0};
    cudnnTensorDescriptor_t x_desc{},y_desc{};
    cudnnFilterDescriptor_t w_desc{};
    cudnnConvolutionDescriptor_t conv_desc{};
    cudnnConvolutionFwdAlgo_t algo;
};

// implementation inline for header‑only
inline Conv2dBF16::Conv2dBF16(int in_c, int out_c): in(in_c), out(out_c) {
    W=std::make_shared<CudaBuffer>(size_t(out)*in*3*3*sizeof(__nv_bfloat16));
    auto &d=dm_common::Handles::instance().dnn;

    cudnnCreateFilterDescriptor(&w_desc);
    cudnnSetFilter4dDescriptor(w_desc,CUDNN_DATA_BFLOAT16,CUDNN_TENSOR_NCHW,out,in,3,3);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc,1,1,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; 
}

inline Conv2dBF16::~Conv2dBF16() {
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc); 
    cudnnDestroyFilterDescriptor(w_desc); 
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

inline void Conv2dBF16::forward(const __nv_bfloat16 *x,int B,int H,int W_,const __nv_bfloat16 *bias,__nv_bfloat16 *y,cudaStream_t s) {
    auto &d=dm_common::Handles::instance().dnn;

    cudnnSetStream(d,s);
    cudnnSetTensor4dDescriptor(x_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,B,in ,H,W_);
    cudnnSetTensor4dDescriptor(y_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,B,out,H,W_);
    
    size_t sz;
    cudnnGetConvolutionForwardWorkspaceSize(d,x_desc,w_desc,conv_desc,y_desc,algo,&sz);
    if(sz>ws_bytes) {
        ws_bytes=sz;
        ws=std::make_shared<CudaBuffer>(ws_bytes)
    }
    
    const float a=1.f,b=0.f;

    cudnnConvolutionForward(d,&a,x_desc,x,w_desc,W->data,conv_desc,algo,ws->data,ws_bytes,&b,y_desc,y);

    if(bias) add_bias<<<(size_t(B)*out*H*W_+255)/256,256,0,s>>>(y,bias,B,out,H*W_);
}

__global__ inline void add_bias(__nv_bfloat16 *y,const __nv_bfloat16 *b,int B,int C,int HW) {
    size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=size_t(B)*C*HW) return;
    
    int c=(idx/HW)%C;
    y[idx]=f2bf(bf2f(y[idx])+bf2f(b[c]));
}

// ---------------- MaxPool2dBF16 ----------------
class MaxPool2dBF16 { 
public: 
    MaxPool2dBF16() { 
        cudnnCreatePoolingDescriptor(&p);
        cudnnSetPooling2dDescriptor(p,CUDNN_POOLING_MAX,CUDNN_NOT_PROPAGATE_NAN,2,2,0,0,2,2);
    } 
    
    ~MaxPool2dBF16() {
        cudnnDestroyPoolingDescriptor(p);
    } 
    
    void forward(const __nv_bfloat16 *x,int B,int C,int H,int W,__nv_bfloat16 *y,cudaStream_t s) {
        auto &d=dm_common::Handles::instance().dnn;
        cudnnSetStream(d,s);
        cudnnTensorDescriptor_t ti,to;

        cudnnCreateTensorDescriptor(&ti);
        cudnnCreateTensorDescriptor(&to);
        cudnnSetTensor4dDescriptor(ti,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,B,C,H,W);
        cudnnSetTensor4dDescriptor(to,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,B,C,H/2,W/2);
        const float a=1.f,b=0.f;
        cudnnPoolingForward(d,p,&a,ti,x,&b,to,y);
        cudnnDestroyTensorDescriptor(ti);
        cudnnDestroyTensorDescriptor(to);
    } 
    
private: 
    cudnnPoolingDescriptor_t p;
};

// ---------------- LayerNormBF16 --------------
class LayerNormBF16{ public: explicit LayerNormBF16(int C_):C(C_){ gamma=std::make_shared<CudaBuffer>(C*sizeof(float)); beta=std::make_shared<CudaBuffer>(C*sizeof(float)); std::vector<float> g(C,1.f),z(C,0.f); cudaMemcpy(gamma->data,g.data(),C*sizeof(float),cudaMemcpyHostToDevice); cudaMemcpy(beta->data,z.data(),C*sizeof(float),cudaMemcpyHostToDevice);} void forward(const __nv_bfloat16 *x,int B,int H,int W,__nv_bfloat16 *y,cudaStream_t s){ auto &d=dm_common::Handles::instance().dnn; cudnnSetStream(d,s); cudnnTensorDescriptor_t td; cudnnCreateTensorDescriptor(&td); cudnnSetTensor4dDescriptor(td,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,B,C,H,W); const float a=1.f,b=0.f; cudnnNormalizationForward(d,CUDNN_NORM_LAYERNORM,&a,&b,td,x,td,y,gamma->data,beta->data,nullptr,nullptr,1e-5f,nullptr); cudnnDestroyTensorDescriptor(td);} private:int C; std::shared_ptr<CudaBuffer> gamma,beta;};

// ---------------- TimeEmbeddingBF16 --------------
class TimeEmbeddingBF16 {
public:
    const int dim;
    LinearBF16 l1, l2;
    CudaBuffer tmp;

    TimeEmbeddingBF16(int d, int mlp=-1): dim(d), l1(d, mlp<0?4*d:mlp), l2(mlp<0?4*d:mlp, d) {}

    void forward(const int32_t *t_host, int B, __nv_bfloat16 *out, cudaStream_t s) {
        /* 1. raw sinusoid on host */
        std::vector<__nv_bfloat16> h(B*dim);
        const int half = dim/2;
        for(int b=0;b<B;++b){ float tt=float(t_host[b]); for(int i=0;i<half;++i){ float f=expf(-logf(10000.f)*i/(half-1)); h[b*dim+i]=f2bf(sinf(tt*f)); h[b*dim+half+i]=f2bf(cosf(tt*f)); } if(dim%2) h[b*dim+dim-1]=f2bf(0.f);}        
        cudaMemcpyAsync(out,h.data(),B*dim*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice,s);
        
        /* 2. proj1 -> SiLU -> proj2 */
        tmp.resize(size_t(B)*l1.out_f*sizeof(__nv_bfloat16));
        for(int b=0;b<B;++b) l1.forward(out+b*dim,(__nv_bfloat16*)tmp.data+b*l1.out_f,s);
        launch_silu((__nv_bfloat16*)tmp.data,size_t(B)*l1.out_f,s);
        for(int b=0;b<B;++b) l2.forward((__nv_bfloat16*)tmp.data+b*l1.out_f,out+b*dim,s);
    }
};
