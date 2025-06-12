#pragma once

// DiffusionEDM.cuh – EDMPrecondBF16 wrapper
#include "DiffusionHelper.cuh"
#include "DiffusionUNet.cuh"

class EDMPrecondBF16 {
public:
    EDMPrecondBF16(int res,float sd=0.5f):net(res),sigma_data(sd){}

    void forward(__nv_bfloat16 *x_noisy,const float *sigma,int B,int H,int W,cudaStream_t st) {
        size_t elems=size_t(B)*H*W; const size_t perImg=H*W;
        // host compute coeff
        std::vector<float> hsig(B); cudaMemcpyAsync(hsig.data(),sigma,B*sizeof(float),cudaMemcpyDeviceToHost,st); cudaStreamSynchronize(st);
        std::vector<float> cskip(B),cout(B),cinv(B); std::vector<int32_t> ln(B);
        float sd2=sigma_data*sigma_data; for(int i=0;i<B;++i){ cskip[i]=sd2/(hsig[i]*hsig[i]+sd2); cout[i]=hsig[i]*sigma_data/std::sqrt(hsig[i]*hsig[i]+sd2); cinv[i]=1.f/std::sqrt(hsig[i]*hsig[i]+sd2); ln[i]=int32_t(std::log(hsig[i])); }
        CudaBuffer cskip_d(B*sizeof(float)),cout_d(B*sizeof(float)),cinv_d(B*sizeof(float)),ln_d(B*sizeof(int32_t));
        cudaMemcpyAsync(cskip_d.data,cskip.data(),B*sizeof(float),cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(cout_d.data ,cout.data() ,B*sizeof(float),cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(cinv_d.data ,cinv.data() ,B*sizeof(float),cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(ln_d.data   ,ln.data()   ,B*sizeof(int32_t),cudaMemcpyHostToDevice,st);
        // scale
        CudaBuffer x_in(elems*sizeof(__nv_bfloat16)); scale_kernel<<<(elems+255)/256,256,0,st>>>(x_noisy,(float*)cinv_d.data,(__nv_bfloat16*)x_in.data,perImg,elems);
        // net
        net.forward((__nv_bfloat16*)x_in.data,(int32_t*)ln_d.data,B,st);
        // blend (write back to x_noisy)
        blend_kernel<<<(elems+255)/256,256,0,st>>>(x_noisy,(__nv_bfloat16*)x_in.data,(float*)cskip_d.data,(float*)cout_d.data,x_noisy,perImg,elems);
    }

private: UNetBF16 net; float sigma_data;};
