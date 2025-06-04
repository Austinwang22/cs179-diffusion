#include <iostream>
#include <random>
#include <fstream>
#include <nvtx3/nvToolsExt.h>

#include "vendor/argparse.hpp"
#include "DiffusionModel.cuh"  // something else!
#include "CudaBuffer.cuh"
#include "ErrorCheck.h"

using json = nlohmann::json;

struct DiffusionRunner {
    int     batch;
    int     res;
    float   sigma_data;
    float   test_sigma;
    int     iterations;

    EDMPrecondBF16 model;
    cudaStream_t   stream;

    CudaBuffer x_noisy; // bf16 (B,C,H,W) where C=1
    CudaBuffer sigmas;  // float32 per sample

    DiffusionRunner(int b,int r,float sd,float sig,int it)
        : batch(b),res(r),sigma_data(sd),test_sigma(sig),iterations(it),model(r,sd),
          x_noisy(size_t(b)*r*r*sizeof(__nv_bfloat16)),
          sigmas (b*sizeof(float))
    {
        checkCuda(cudaStreamCreate(&stream));
        // init host data
        std::vector<__nv_bfloat16> h_img(size_t(b)*r*r);
        std::vector<float>         h_sig(b, test_sigma);
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> un(-1.f,1.f);
        for(auto &px:h_img) px = __float2bfloat16(un(gen));
        cudaMemcpy(x_noisy.data,h_img.data(),h_img.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice);
        cudaMemcpy(sigmas.data ,h_sig.data(),b*sizeof(float),cudaMemcpyHostToDevice);
    }

    void run() {
        size_t elem_per_img = size_t(res)*res;
        for(int it=0; it<iterations; ++it){
            if(it==iterations-1) nvtxRangePush("last_iter");
            model.forward((__nv_bfloat16*)x_noisy.data,(float*)sigmas.data,batch,res,res,stream);
            if(it==iterations-1) nvtxRangePop();
        }
        // pull few pixels to show output
        std::vector<__nv_bfloat16> h_out(8);
        cudaMemcpyAsync(h_out.data(),x_noisy.data,8*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        std::cout << "First 8 output pixels: ";
        for(auto bf:h_out) std::cout << __bfloat162float(bf) << " ";
        std::cout << std::endl;
    }

    ~DiffusionRunner(){ cudaStreamDestroy(stream);} };

int main(int argc,const char* argv[]){
    argparse::ArgumentParser program("diffusion_test");
    program.add_argument("--batch").default_value(2).scan<'i',int>();
    program.add_argument("--res").default_value(32).scan<'i',int>();
    program.add_argument("--sigma").default_value(1.0f).scan<'g',float>();
    program.add_argument("--sigma-data").default_value(0.5f).scan<'g',float>();
    program.add_argument("--iters").default_value(5).scan<'i',int>();
    try{ program.parse_args(argc,argv);}catch(const std::exception& e){ std::cerr<<e.what()<<"\n"<<program; return 1;}

    int B   = program.get<int>("--batch");
    int RES = program.get<int>("--res");
    float sig   = program.get<float>("--sigma");
    float sig_d = program.get<float>("--sigma-data");
    int iters   = program.get<int>("--iters");

    DiffusionRunner runner(B,RES,sig_d,sig,iters);
    runner.run();
    return 0; }
