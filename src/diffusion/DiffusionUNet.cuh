#pragma once
// DiffusionUNet.cuh – 2‑level encoder‑decoder U‑Net (bf16)
#include "DiffusionHelper.cuh"
#include "DiffusionLayers.cuh"

struct Shape { int B,C,H,W; };

// ------------ EncoderBlock --------------
class EncoderBlockBF16 {
public:
    EncoderBlockBF16(int in_c,int out_c,int t_dim):
        conv1(in_c,out_c),conv2(out_c,out_c),ln(out_c),
        tproj(out_c,false),pool() { (void)t_dim; }

    void forward(__nv_bfloat16 *x,Shape &s,const __nv_bfloat16 *temb,cudaStream_t st,std::vector<std::shared_ptr<CudaBuffer>>&skip) {
        tmp.resize(size_t(s.B)*conv1_out()*s.H*s.W*sizeof(__nv_bfloat16));
        conv1.forward(x,s.B,s.H,s.W,nullptr,(__nv_bfloat16*)tmp.data,st);
        add_time_bias((__nv_bfloat16*)tmp.data,temb,s.B,conv1_out(),s.H,s.W,st);
        launch_silu((__nv_bfloat16*)tmp.data,size_t(s.B)*conv1_out()*s.H*s.W,st);
        conv2.forward((__nv_bfloat16*)tmp.data,s.B,s.H,s.W,nullptr,x,st);
        add_time_bias(x,temb,s.B,conv2_out(),s.H,s.W,st);
        launch_silu(x,size_t(s.B)*conv2_out()*s.H*s.W,st);
        ln.forward(x,s.B,s.H,s.W,x,st);
        auto saved=std::make_shared<CudaBuffer>(size_t(s.B)*s.C*s.H*s.W*sizeof(__nv_bfloat16)); cudaMemcpyAsync(saved->data,x,saved->byteSize(),cudaMemcpyDeviceToDevice,st); skip.push_back(saved);
        pool.forward(x,s.B,s.C,s.H,s.W,(__nv_bfloat16*)tmp.data,st);
        cudaMemcpyAsync(x,tmp.data,tmp.byteSize(),cudaMemcpyDeviceToDevice,st);
        s.H/=2; s.W/=2;
    }
private:
    Conv2dBF16 conv1,conv2; LayerNormBF16 ln; LinearBF16 tproj; MaxPool2dBF16 pool; CudaBuffer tmp; int conv1_out(){return ln_C;} int conv2_out(){return ln_C;} int ln_C;};

// ------------ Decoder helpers ------------
static inline void upsample_inplace(__nv_bfloat16 *x,Shape &s,cudaStream_t st) { 
    CudaBuffer tmp(size_t(s.B)*s.C*s.H*s.W*4*sizeof(__nv_bfloat16));
    upsample2x(x,s.B,s.C,s.H,s.W,(__nv_bfloat16*)tmp.data,st);
    cudaMemcpyAsync(x,tmp.data,tmp.byteSize(),cudaMemcpyDeviceToDevice,st);
    
    s.H*=2;
    s.W*=2; 
}

static inline void cat_channels(__nv_bfloat16 *x,const void *skip,Shape &s,int skipC,CudaBuffer &cat,cudaStream_t st) {
    cat.resize(size_t(s.B)*(s.C+skipC)*s.H*s.W*sizeof(__nv_bfloat16));
    cudaMemcpyAsync(cat.data,skip,size_t(s.B)*skipC*s.H*s.W*sizeof(__nv_bfloat16),cudaMemcpyDeviceToDevice,st);
    cudaMemcpyAsync(((__nv_bfloat16*)cat.data)+size_t(s.B)*skipC*s.H*s.W,x,size_t(s.B)*s.C*s.H*s.W*sizeof(__nv_bfloat16),cudaMemcpyDeviceToDevice,st);
    
    s.C+=skipC; 
}

// ------------ UNetBF16 -------------------
class UNetBF16 {
public:
    UNetBF16(int res,int tdim=128): img_res(res), timeEmb(tdim) {
        enc.emplace_back(std::make_unique<EncoderBlockBF16>(1,64,tdim));
        enc.emplace_back(std::make_unique<EncoderBlockBF16>(64,128,tdim));
        bott1=std::make_unique<Conv2dBF16>(128,128);
        bott2=std::make_unique<Conv2dBF16>(128,128);
        bott_ln=std::make_unique<LayerNormBF16>(128);
        dec1a=std::make_unique<Conv2dBF16>(256,128);
        dec1b=std::make_unique<Conv2dBF16>(128,128);
        dec2a=std::make_unique<Conv2dBF16>(192,64);
        dec2b=std::make_unique<Conv2dBF16>(64,64);
        out_conv=std::make_unique<Conv2dBF16>(64,1);
    }

    void forward(__nv_bfloat16 *x,const int32_t *t,int B,cudaStream_t st){
        // time embedding
        temb_buf.resize(size_t(B)*timeEmb.dim*sizeof(__nv_bfloat16)); timeEmb.forward(t,B,(__nv_bfloat16*)temb_buf.data,st);
        const __nv_bfloat16 *temb=(__nv_bfloat16*)temb_buf.data;
        // encoder
        Shape s{B,1,img_res,img_res}; skip.clear(); for(auto &blk:enc) blk->forward(x,s,temb,st,skip);
        // bottleneck
        bott1->forward(x,s.B,s.H,s.W,nullptr,(__nv_bfloat16*)bot_tmp.data,st);
        bott2->forward((__nv_bfloat16*)bot_tmp.data,s.B,s.H,s.W,nullptr,x,st); bott_ln->forward(x,s.B,s.H,s.W,x,st);
        // decoder stage‑1
        upsample_inplace(x,s,st); CudaBuffer cat; cat_channels(x,skip.back()->data,s,128,cat,st);
        dec1a->forward((__nv_bfloat16*)cat.data,s.B,s.H,s.W,nullptr,x,st); dec1b->forward(x,s.B,s.H,s.W,nullptr,x,st); skip.pop_back();
        // decoder stage‑2
        upsample_inplace(x,s,st); cat_channels(x,skip.back()->data,s,64,cat,st);
        dec2a->forward((__nv_bfloat16*)cat.data,s.B,s.H,s.W,nullptr,x,st); dec2b->forward(x,s.B,s.H,s.W,nullptr,x,st); skip.pop_back();
        // out conv
        out_conv->forward(x,s.B,s.H,s.W,nullptr,x,st);
    }

private:
    int img_res;
    TimeEmbeddingBF16 timeEmb;
    CudaBuffer temb_buf,bot_tmp;
    std::vector<std::unique_ptr<EncoderBlockBF16>> enc;
    std::vector<std::shared_ptr<CudaBuffer>> skip;
    std::unique_ptr<Conv2dBF16> bott1,bott2,dec1a,dec1b,dec2a,dec2b,out_conv;
    std::unique_ptr<LayerNormBF16>bott_ln;
};
