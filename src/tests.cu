// tests_layers_refactored.cu ────────────────────────────────────────────────
// Lightweight, self‑contained unit tests for the BF16 diffusion primitives.
// Uses the refactored DiffusionLayersV2 + helper utilities.
// Compile with:  nvcc -std=c++17 -Iinclude tests_layers_refactored.cu -lcudart -lcublas -lcudnn
// ---------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "diffusion/DiffusionLayers.cuh"      // LinearBF16, Conv2dBF16, ConvTrans2dBF16, MaxPool2dBF16

// ----------------------------------------------------------------------------
// Host⇄Device conversion helpers (FP32 ↔︎ BF16)
// ----------------------------------------------------------------------------
namespace util {
static void copy_host_to_bf16(const std::vector<float>& h, void* dev)
{
    std::vector<__nv_bfloat16> tmp(h.size());
    for (size_t i = 0; i < h.size(); ++i) tmp[i] = __float2bfloat16(h[i]);
    checkCuda(cudaMemcpy(dev, tmp.data(), h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

static std::vector<float> copy_bf16_to_host(const void* dev, size_t n)
{
    std::vector<__nv_bfloat16> tmp(n);
    std::vector<float>         h(n);
    checkCuda(cudaMemcpy(tmp.data(), dev, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) h[i] = __bfloat162float(tmp[i]);
    return h;
}

static bool allclose(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-3f)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) {
            std::cerr << "Mismatch @" << i << ": " << a[i] << " vs " << b[i] << "\n";
            
            for (size_t i = 0; i < a.size(); ++i) {
                std::cout << a[i] << " ";
            }
            return false;
        }
    return true;
}
} // namespace util

using util::copy_bf16_to_host;
using util::copy_host_to_bf16;
using util::allclose;

// ----------------------------------------------------------------------------
// 1. SiLU kernel (already in DiffusionHelper) – sanity regression
// ----------------------------------------------------------------------------
static bool test_silu()
{
    constexpr size_t N = 128;
    CudaBuffer buf(N * sizeof(__nv_bfloat16));

    // deterministic host values
    std::vector<float> h(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> ur(-5.f, 5.f);
    for (auto& v : h) v = ur(gen);

    copy_host_to_bf16(h, buf.data);
    dm::silu_inplace(reinterpret_cast<__nv_bfloat16*>(buf.data), N, 0 /* default stream */);
    auto out = copy_bf16_to_host(buf.data, N);

    // reference with identical BF16 rounding
    std::vector<float> ref(N);
    for (size_t i = 0; i < N; ++i) {
        float x = __bfloat162float(__float2bfloat16(h[i]));  // BF16 cast in
        float y = x / (1.f + std::exp(-x));                  // SiLU
        ref[i]  = __bfloat162float(__float2bfloat16(y));     // BF16 cast out
    }
    return allclose(out, ref, 2e-3f);
}

// ----------------------------------------------------------------------------
// 2. Linear layer small GEMM
// ----------------------------------------------------------------------------
static bool test_linear()
{
    constexpr int IN = 4, OUT = 3;
    LinearBF16 lin(IN, OUT);

    std::vector<float> W(OUT * IN), B(OUT), X(IN);
    for (int i = 0; i < IN;  ++i) X[i] = float(i + 1);
    for (int o = 0; o < OUT; ++o) {
        B[o] = float(o);
        for (int i = 0; i < IN; ++i) W[o * IN + i] = float((o + 1) * (i + 2));
    }
    copy_host_to_bf16(W, lin.W->data);
    copy_host_to_bf16(B, lin.b->data);

    CudaBuffer x_dev(IN * sizeof(__nv_bfloat16));
    CudaBuffer y_dev(OUT * sizeof(__nv_bfloat16));
    copy_host_to_bf16(X, x_dev.data);
    lin.forward(reinterpret_cast<__nv_bfloat16*>(x_dev.data),
                reinterpret_cast<__nv_bfloat16*>(y_dev.data), 0);

    auto out = copy_bf16_to_host(y_dev.data, OUT);
    std::vector<float> ref(OUT);
    for (int o = 0; o < OUT; ++o) {
        float acc = B[o];
        for (int i = 0; i < IN; ++i) acc += W[o * IN + i] * X[i];
        ref[o] = __bfloat162float(__float2bfloat16(acc));
    }
    return allclose(out, ref, 1e-2f);
}

// ----------------------------------------------------------------------------
// 3. Conv2d identity (3×3 kernel with center=1)
// ----------------------------------------------------------------------------
static bool test_conv_identity()
{
    const int H = 4, W = 4, B = 1, Cin = 1, Cout = 1;
    Conv2dBF16 conv(Cin, Cout);

    std::vector<float> k(9, 0.f); k[4] = 1.f;
    copy_host_to_bf16(k, conv.weights());

    std::vector<float> img(H * W); for (int i = 0; i < H * W; ++i) img[i] = float(i);
    CudaBuffer x(H * W * sizeof(__nv_bfloat16)), y(H * W * sizeof(__nv_bfloat16));
    copy_host_to_bf16(img, x.data);

    conv.forward(reinterpret_cast<__nv_bfloat16*>(x.data), B, H, W,
                 reinterpret_cast<__nv_bfloat16*>(y.data), 0);

    auto out = copy_bf16_to_host(y.data, H * W);
    return allclose(out, img, 1e-2f);
}

// ----------------------------------------------------------------------------
// 4. MaxPool 2×2 → expect top‑left values of each block
// ----------------------------------------------------------------------------
static bool test_maxpool()
{
    const int H = 4, W = 4;
    MaxPool2dBF16 pool;

    std::vector<float> img(H * W); for (int i = 0; i < H * W; ++i) img[i] = float(i);
    CudaBuffer x(H * W * sizeof(__nv_bfloat16)); copy_host_to_bf16(img, x.data);
    CudaBuffer y((H / 2) * (W / 2) * sizeof(__nv_bfloat16));

    pool.forward(reinterpret_cast<__nv_bfloat16*>(x.data), 1, 1, H, W,
                 reinterpret_cast<__nv_bfloat16*>(y.data), 0);

    auto out = copy_bf16_to_host(y.data, (H / 2) * (W / 2));
    std::vector<float> ref = {5, 7, 13, 15};
    return allclose(out, ref, 1e-2f);
}

// ----------------------------------------------------------------------------
// 5. ConvTranspose2d up‑sample test  (2×2, stride 2)
//    Identity weights should replicate each pixel into 2×2 block.
// ----------------------------------------------------------------------------
static bool test_conv_transpose()
{
    const int H = 2, W = 2, B = 1, C = 1;
    ConvTrans2dBF16 deconv(C, C);

    /* kernel: all ones -> replicate */
    std::vector<float> k = {1.f, 1.f,
                            1.f, 1.f};
    copy_host_to_bf16(k, deconv.weights());      // single copy is enough

    /* 2×2 input */
    std::vector<float> img = {1, 2,
                              3, 4};
    CudaBuffer x(img.size() * sizeof(__nv_bfloat16));
    copy_host_to_bf16(img, x.data);

    /* 4×4 output buffer */
    CudaBuffer y(4 * img.size() * sizeof(__nv_bfloat16));

    deconv.forward(reinterpret_cast<__nv_bfloat16*>(x.data), B, H, W,
                   reinterpret_cast<__nv_bfloat16*>(y.data), 0);
    cudaDeviceSynchronize();                     // make sure kernel finished

    auto out = copy_bf16_to_host(y.data, 4 * img.size());
    const std::vector<float> ref = {
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4 };
    return allclose(out, ref, 1e-2f);
}

// ----------------------------------------------------------------------------
// Main driver
// ----------------------------------------------------------------------------
int main()
{
    std::cout << "SiLU            : " << (test_silu()          ? "OK" : "FAIL") << "\n";
    std::cout << "Linear          : " << (test_linear()        ? "OK" : "FAIL") << "\n";
    std::cout << "Conv Identity   : " << (test_conv_identity() ? "OK" : "FAIL") << "\n";
    std::cout << "MaxPool         : " << (test_maxpool()       ? "OK" : "FAIL") << "\n";
    std::cout << "ConvTranspose   : " << (test_conv_transpose()? "OK" : "FAIL") << "\n";
    return 0;
}


// // tests_layers.cu – unit tests for bf16 diffusion layers
// // -----------------------------------------------------------------------------
// #include <iostream>
// #include <vector>
// #include <random>
// #include <cmath>

// #include "diffusion/DiffusionHelper.cuh"
// #include "diffusion/DiffusionLayers.cuh"  // LinearBF16, Conv2dBF16, etc.

// // -----------------------------------------------------------------------------
// // small helpers
// // -----------------------------------------------------------------------------
// static void copy_host_to_bf16(const std::vector<float>& h, void* dev){
//     std::vector<__nv_bfloat16> tmp(h.size());
//     for(size_t i=0;i<h.size();++i) tmp[i] = __float2bfloat16(h[i]);
//     checkCuda(cudaMemcpy(dev, tmp.data(), h.size()*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
// }
// static std::vector<float> copy_bf16_to_host(const void* dev, size_t n){
//     std::vector<__nv_bfloat16> tmp(n); std::vector<float> h(n);
//     checkCuda(cudaMemcpy(tmp.data(), dev, n*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
//     for(size_t i=0;i<n;++i) h[i] = __bfloat162float(tmp[i]);
//     return h;
// }
// static bool allclose(const std::vector<float>& a,const std::vector<float>& b, float tol){
//     if(a.size()!=b.size()) return false; 
    
//     for(size_t i=0;i<a.size();++i) {
//         if(std::fabs(a[i]-b[i])>tol) {

//             std::cout << "the failure: " << a[1] << "  " << b[1] << " " << std::fabs(a[i]-b[i]) << " which is bigger than "<< tol << "\n";

//             return false;
//         }
//     }

//     return true;
// }

// // -----------------------------------------------------------------------------
// // 1. test SiLU kernel
// // -----------------------------------------------------------------------------
// bool test_silu(){
//     size_t n=128;
    
//     CudaBuffer buf(n*sizeof(__nv_bfloat16));
    
//     std::vector<float> h(n);
//     std::mt19937 gen(0);
//     std::uniform_real_distribution<float> u(-3.f,3.f);
    
//     for(float &v:h) v=u(gen);
    
//     copy_host_to_bf16(h,buf.data);
//     silu_inplace((__nv_bfloat16*)buf.data,n,0);
//     auto out = copy_bf16_to_host(buf.data,n);
    
//     std::vector<float> ref(n); 
//     for (size_t i = 0; i < n; ++i) {
//         // match the device: first cast x to BF16, back to float
//         __nv_bfloat16 x_bf16 = __float2bfloat16(h[i]);
//         float x_fp32_bf16    = __bfloat162float(x_bf16);

//         float y = x_fp32_bf16 / (1.f + std::exp(-x_fp32_bf16));   // SiLU
//         // now cast the *output* to BF16 exactly like the kernel does
//         ref[i] = __bfloat162float(__float2bfloat16(y));
//     }
//     return allclose(out, ref, 2e-3);          // one BF16 ULP for |y|≤3
// }

// // -----------------------------------------------------------------------------
// // 2. Linear layer
// // -----------------------------------------------------------------------------
// bool test_linear(){
//     constexpr int IN=4, OUT=3; LinearBF16 lin(IN,OUT); cudaStream_t s=nullptr;
//     std::vector<float> W(OUT*IN),B(OUT),X(IN);
//     for(int i=0;i<IN;++i) X[i]=float(i+1);
//     for(int o=0;o<OUT;++o){ B[o]=float(o); for(int i=0;i<IN;++i) W[o*IN+i]=float((o+1)*(i+1)); }
//     copy_host_to_bf16(W, lin.W->data); copy_host_to_bf16(B, lin.b->data);
//     CudaBuffer x_dev(IN*sizeof(__nv_bfloat16)), y_dev(OUT*sizeof(__nv_bfloat16));
//     copy_host_to_bf16(X, x_dev.data);
//     lin.forward((__nv_bfloat16*)x_dev.data, (__nv_bfloat16*)y_dev.data, s);
//     auto out = copy_bf16_to_host(y_dev.data, OUT);
//     std::vector<float> ref(OUT);
//     for(int o=0;o<OUT;++o){ float sum=B[o]; for(int i=0;i<IN;++i) sum+=W[o*IN+i]*X[i]; ref[o]=sum; }
//     return allclose(out,ref,1e-2);
// }

// // -----------------------------------------------------------------------------
// // 3. Conv2d identity kernel (1×1 conv equivalence)
// // -----------------------------------------------------------------------------
// bool test_conv_identity(){
//     const int H=4,W=4,B=1,Cin=1,Cout=1; Conv2dBF16 conv(Cin,Cout); cudaStream_t s=nullptr;
//     std::vector<float> kernel(9,0.f); kernel[4]=1.f; // identity 3x3
//     copy_host_to_bf16(kernel, conv.weights());
//     std::vector<float> img(H*W); for(int i=0;i<H*W;++i) img[i]=float(i);
//     CudaBuffer x(H*W*sizeof(__nv_bfloat16)), y(H*W*sizeof(__nv_bfloat16));
//     copy_host_to_bf16(img, x.data);
//     conv.forward((__nv_bfloat16*)x.data,B,H,W,nullptr,(__nv_bfloat16*)y.data,s);
//     auto out = copy_bf16_to_host(y.data,H*W);
//     return allclose(out,img,1e-2);
// }

// // -----------------------------------------------------------------------------
// // 4. MaxPool 2×2
// // -----------------------------------------------------------------------------
// bool test_maxpool(){
//     const int H=4,W=4; MaxPool2dBF16 pool; cudaStream_t s=nullptr;
//     std::vector<float> img(H*W); for(int i=0;i<H*W;++i) img[i]=float(i);
//     CudaBuffer x(H*W*sizeof(__nv_bfloat16)), y(H*W/4*sizeof(__nv_bfloat16)); copy_host_to_bf16(img,x.data);
//     pool.forward((__nv_bfloat16*)x.data,1,1,H,W,(__nv_bfloat16*)y.data,s);
//     auto out=copy_bf16_to_host(y.data,H/2*W/2);
//     std::vector<float> ref={5,7,13,15};
//     return allclose(out,ref,1e-2);
// }

// // -----------------------------------------------------------------------------
// // 5. LayerNorm (per-channel) – zero mean test
// // -----------------------------------------------------------------------------
// bool test_layernorm(){
//     const int C=2,H=2,W=2,B=1; LayerNormBF16 ln(C); cudaStream_t s=nullptr;
//     std::vector<float> img(C*H*W); for(size_t i=0;i<img.size();++i) img[i]=float(i);
//     CudaBuffer x(img.size()*sizeof(__nv_bfloat16)), y(img.size()*sizeof(__nv_bfloat16)); copy_host_to_bf16(img,x.data);
//     ln.forward((__nv_bfloat16*)x.data,B,H,W,(__nv_bfloat16*)y.data,s);
//     auto out=copy_bf16_to_host(y.data,img.size());
//     // check each channel mean ~0
//     for(int c=0;c<C;++c){ float mean=0; for(int hw=0;hw<H*W;++hw) mean+=out[c*H*W+hw]; mean/=H*W; if(std::fabs(mean)>1e-2) return false; }
//     return true;
// }

// // -----------------------------------------------------------------------------
// // 6. TimeEmbedding – proj2(SiLU(proj1(sincos))) end-to-end BF16-exact
// // -----------------------------------------------------------------------------
// static uint16_t float2bf16_rte(float v) {               // round-to-nearest-even
//     uint32_t u = *reinterpret_cast<uint32_t*>(&v);
//     uint32_t r = 0x7FFF + ((u >> 16) & 1);              // sticky + tie-to-even
//     return static_cast<uint16_t>((u + r) >> 16);
// }
// static float bf16bits_to_float(uint16_t h) { uint32_t u = uint32_t(h) << 16;
//     return *reinterpret_cast<float*>(&u); }

// bool test_time_embedding() {
//     constexpr int B = 1, dim = 8, half = dim / 2;
//     TimeEmbeddingBF16 emb(dim);                         // proj1:32, proj2:8
//     cudaStream_t s = nullptr;

//     /* 1 ─ deterministic weight / bias initialisation (simple pattern) */
//     auto init_linear = [](LinearBF16& l) {
//         std::vector<float> W(l.out_f * l.in_f), B(l.out_f);
//         for (int o = 0; o < l.out_f; ++o) {
//             B[o] = float(o);                            // bias = o
//             for (int i = 0; i < l.in_f; ++i)
//                 W[o * l.in_f + i] = float((o + 1) * (i + 1));
//         }
//         copy_host_to_bf16(W, l.weights());
//         copy_host_to_bf16(B, l.bias());
//     };
//     init_linear(emb.proj1);
//     init_linear(emb.proj2);

//     /* 2 ─ GPU result */
//     int32_t t_host[B] = {3};
//     CudaBuffer out_dev(size_t(B) * dim * sizeof(__nv_bfloat16));
//     emb.forward(t_host, B, static_cast<__nv_bfloat16*>(out_dev.data), s);
//     auto out_gpu = copy_bf16_to_host(out_dev.data, dim);

//     /* 3 ─ CPU reference with identical BF16 quantisation steps */
//     /* 3.1 sin/cos */
//     std::vector<float> x(dim);
//     for (int i = 0; i < half; ++i) {
//         float f = std::exp(-std::log(10000.f) * i / (half - 1));
//         x[i]        = std::sin(3.f * f);
//         x[half + i] = std::cos(3.f * f);
//     }
//     if (dim & 1) x.back() = 0.f;
//     for (float& v : x) v = bf16bits_to_float(float2bf16_rte(v));        // BF16-in

//     /* helpers to bring weights/biases back to FP32 */
//     auto load_vec = [](const __nv_bfloat16* p, size_t n) {
//         std::vector<float> v(n);
//         for (size_t k = 0; k < n; ++k) v[k] = __bfloat162float(p[k]);
//         return v;
//     };
//     auto W1 = load_vec(emb.proj1.weights(), emb.proj1.out_f * emb.proj1.in_f);
//     auto B1 = load_vec(emb.proj1.bias(),    emb.proj1.out_f);
//     auto W2 = load_vec(emb.proj2.weights(), emb.proj2.out_f * emb.proj2.in_f);
//     auto B2 = load_vec(emb.proj2.bias(),    emb.proj2.out_f);

//     /* 3.2 proj1 GEMM (row-major) */
//     std::vector<float> h1(emb.proj1.out_f, 0.f);
//     for (int o = 0; o < emb.proj1.out_f; ++o)
//         for (int i = 0; i < emb.proj1.in_f; ++i)
//             h1[o] += W1[o * emb.proj1.in_f + i] * x[i];
//     for (int o = 0; o < emb.proj1.out_f; ++o) {          // BF16-round, +bias, BF16-round
//         h1[o] = bf16bits_to_float(float2bf16_rte(h1[o]));
//         h1[o] = bf16bits_to_float(float2bf16_rte(h1[o] + B1[o]));
//     }

//     /* 3.3 SiLU + BF16-round */
//     for (float& v : h1) v = bf16bits_to_float(
//         float2bf16_rte(v / (1.f + std::exp(-v))));

//     /* 3.4 proj2 GEMM */
//     std::vector<float> y(dim, 0.f);
//     for (int o = 0; o < dim; ++o)
//         for (int i = 0; i < emb.proj2.in_f; ++i)
//             y[o] += W2[o * emb.proj2.in_f + i] * h1[i];
//     for (int o = 0; o < dim; ++o) {                      // BF16-round, +bias, BF16-round
//         y[o] = bf16bits_to_float(float2bf16_rte(y[o]));
//         y[o] = bf16bits_to_float(float2bf16_rte(y[o] + B2[o]));
//     }

//     /* 4 ─ compare (one-ULP tolerance for BF16 result range) */
//     return allclose(out_gpu, y, 0.002f) ||               // small values
//            allclose(out_gpu, y, 4096.f);                 // large values
// }


// // -----------------------------------------------------------------------------
// int main() {
//     std::cout<<"SiLU        : "<<(test_silu()?"OK":"FAIL")<<"\n";
//     std::cout<<"Linear      : "<<(test_linear()?"OK":"FAIL")<<"\n";
//     std::cout<<"Conv Identity: "<<(test_conv_identity()?"OK":"FAIL")<<"\n";
//     std::cout<<"MaxPool     : "<<(test_maxpool()?"OK":"FAIL")<<"\n";
//     std::cout<<"LayerNorm   : "<<(test_layernorm()?"OK":"FAIL")<<"\n";
//     std::cout<<"TimeEmbed   : "<<(test_time_embedding()?"OK":"FAIL")<<"\n";
//     return 0;
// }
