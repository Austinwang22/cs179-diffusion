#pragma once

#include <cuda_bf16.h>
#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <memory>
#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"
#include "../ErrorCheck.h"


template<Qwen2Size SIZE>
class Qwen2Layer {
public:
    using CFG = Qwen2Config<SIZE>;

    LayerNorm input_layernorm{CFG::hidden_size()};
    std::shared_ptr<CudaBuffer> q_proj_weight, q_proj_bias;
    std::shared_ptr<CudaBuffer> k_proj_weight, k_proj_bias;
    std::shared_ptr<CudaBuffer> v_proj_weight, v_proj_bias;
    std::shared_ptr<CudaBuffer> o_proj_weight;
    LayerNorm post_attention_layernorm{CFG::hidden_size()};
    std::shared_ptr<CudaBuffer> up_proj_weight, gate_proj_weight, down_proj_weight;

    explicit Qwen2Layer(uint32_t idx, uint32_t max_seq_len) : layer_num(idx) {
        size_t HS = CFG::hidden_size();
        size_t QS = CFG::queries_size();
        size_t KS = CFG::keys_size();
        size_t VS = CFG::values_size();
        size_t IS = CFG::intermediate_size();

        q_proj_weight = std::make_shared<CudaBuffer>(QS * HS * sizeof(__nv_bfloat16));
        q_proj_bias   = std::make_shared<CudaBuffer>(QS * sizeof(__nv_bfloat16));
        k_proj_weight = std::make_shared<CudaBuffer>(KS * HS * sizeof(__nv_bfloat16));
        k_proj_bias   = std::make_shared<CudaBuffer>(KS * sizeof(__nv_bfloat16));
        v_proj_weight = std::make_shared<CudaBuffer>(VS * HS * sizeof(__nv_bfloat16));
        v_proj_bias   = std::make_shared<CudaBuffer>(VS * sizeof(__nv_bfloat16));
        o_proj_weight = std::make_shared<CudaBuffer>(HS * QS * sizeof(__nv_bfloat16));
        up_proj_weight   = std::make_shared<CudaBuffer>(IS * HS * sizeof(__nv_bfloat16));
        gate_proj_weight = std::make_shared<CudaBuffer>(IS * HS * sizeof(__nv_bfloat16));
        down_proj_weight = std::make_shared<CudaBuffer>(HS * IS * sizeof(__nv_bfloat16));

        q_buf  = std::make_shared<CudaBuffer>(QS * sizeof(__nv_bfloat16));
        k_buf  = std::make_shared<CudaBuffer>(KS * sizeof(__nv_bfloat16));
        v_buf  = std::make_shared<CudaBuffer>(VS * sizeof(__nv_bfloat16));
        attn_out = std::make_shared<CudaBuffer>(QS * sizeof(float));
        ffn_up_buf   = std::make_shared<CudaBuffer>(IS * sizeof(__nv_bfloat16));
        ffn_gate_buf = std::make_shared<CudaBuffer>(IS * sizeof(__nv_bfloat16));
        residual_buf = std::make_shared<CudaBuffer>(HS * sizeof(__nv_bfloat16));

        gqa = std::make_unique<GroupQueryAttention<SIZE>>(max_seq_len);
    }

    void forward(const std::shared_ptr<CudaBuffer> &k_cache,
                 const std::shared_ptr<CudaBuffer> &v_cache,
                 const std::shared_ptr<CudaBuffer> &hidden,
                 int32_t seq_len,
                 cudaStream_t stream) {
        const int HS = CFG::hidden_size();
        const int QS = CFG::queries_size();
        const int KS = CFG::keys_size();
        const int VS = CFG::values_size();
        const int IS = CFG::intermediate_size();

        checkCuda(cudaMemcpyAsync(residual_buf->data, hidden->data, HS * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));

        input_layernorm.normalize_hidden_state(hidden, hidden, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(QS, HS, (__nv_bfloat16 *)q_proj_weight->data, (__nv_bfloat16 *)q_proj_bias->data, (__nv_bfloat16 *)hidden->data, (__nv_bfloat16 *)q_buf->data, stream);
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(KS, HS, (__nv_bfloat16 *)k_proj_weight->data, (__nv_bfloat16 *)k_proj_bias->data, (__nv_bfloat16 *)hidden->data, (__nv_bfloat16 *)k_buf->data, stream);
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(VS, HS, (__nv_bfloat16 *)v_proj_weight->data, (__nv_bfloat16 *)v_proj_bias->data, (__nv_bfloat16 *)hidden->data, (__nv_bfloat16 *)v_buf->data, stream);

        int pos = seq_len - 1;
        RoPE::apply_rope_to_qk((__nv_bfloat16 *)q_buf->data, CFG::num_query_heads(), CFG::head_size(), pos, CFG::rope_theta_base(), stream);
        RoPE::apply_rope_to_qk((__nv_bfloat16 *)k_buf->data, CFG::num_kv_heads(), CFG::head_size(), pos, CFG::rope_theta_base(), stream);

        size_t kv_elems = size_t(CFG::num_kv_heads()) * CFG::head_size();
        size_t layer_stride = size_t(CFG::num_layers()) * kv_elems;
        size_t dst = size_t(pos) * layer_stride + size_t(layer_num) * kv_elems;
        checkCuda(cudaMemcpyAsync(((__nv_bfloat16 *)k_cache->data) + dst, (__nv_bfloat16 *)k_buf->data, kv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        checkCuda(cudaMemcpyAsync(((__nv_bfloat16 *)v_cache->data) + dst, (__nv_bfloat16 *)v_buf->data, kv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));

        gqa->sdpa((__nv_bfloat16 *)q_buf->data, (__nv_bfloat16 *)k_cache->data, (__nv_bfloat16 *)v_cache->data, (float *)attn_out->data, layer_num, seq_len, stream);

        MatrixVectorMultiply::bf16_matmul<float>(HS, QS, (__nv_bfloat16 *)o_proj_weight->data, (__nv_bfloat16 *)residual_buf->data, (float *)attn_out->data, (__nv_bfloat16 *)hidden->data, stream);

        checkCuda(cudaMemcpyAsync(residual_buf->data, hidden->data, HS * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));

        post_attention_layernorm.normalize_hidden_state(hidden, hidden, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(IS, HS, (__nv_bfloat16 *)up_proj_weight->data, nullptr, (__nv_bfloat16 *)hidden->data, (__nv_bfloat16 *)ffn_up_buf->data, stream);
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(IS, HS, (__nv_bfloat16 *)gate_proj_weight->data, nullptr, (__nv_bfloat16 *)hidden->data, (__nv_bfloat16 *)ffn_gate_buf->data, stream);

        SiLUMult::silu_mult_in_place(ffn_gate_buf, ffn_up_buf, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(HS, IS, (__nv_bfloat16 *)down_proj_weight->data, (__nv_bfloat16 *)residual_buf->data, (__nv_bfloat16 *)ffn_gate_buf->data, (__nv_bfloat16 *)hidden->data, stream);
    }

private:
    uint32_t layer_num;
    std::shared_ptr<CudaBuffer> q_buf, k_buf, v_buf, attn_out, ffn_up_buf, ffn_gate_buf, residual_buf;
    std::unique_ptr<GroupQueryAttention<SIZE>> gqa;
};
