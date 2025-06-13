#pragma once

#include <memory>
#include <stdexcept>

#include "Qwen2Layer.cuh"
#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/ArgMax.cuh"

template<Qwen2Size QWEN2_SIZE>
class Qwen2Model {
    cudaStream_t stream;
public:
    u1ing C     = Qwen2Config<QWEN2_SIZE>;
    using Layer = Qwen2Layer<QWEN2_SIZE>;

    Qwen2Model() {
        checkCuda(cudaStreamCreate(&stream));
    }
    ~Qwen2Model() {
        checkCuda(cudaStreamDestroy(stream));
    }

    std::shared_ptr<CudaBuffer> embedding_weight;                  // (vocab, hidden)
    std::shared_ptr<Layer>      layers[C::num_layers()];           // decoder blocks
    LayerNorm                   final_layernorm{C::hidden_size()}; // last LN (no bias)

    /**
     * Generate the next token (greedy argmax).
     *
     * @param k_cache  (seq_len, L, G, d_k)  – keys for all past positions
     * @param v_cache  (seq_len, L, G, d_v)  – values cache
     * @param seq_len  Current sequence length (after *appending* input_tok_id)
     * @param input_tok_id  ID of the last token in the prompt so far
     * @param temperature  Must be 0 (only greedy decode supported)
     */
    int32_t forward(const std::shared_ptr<CudaBuffer>& k_cache,
                    const std::shared_ptr<CudaBuffer>& v_cache,
                    int32_t  seq_len,
                    int32_t  input_tok_id,
                    float    temperature)
    {
        if (temperature != 0.f)
            throw std::runtime_error("Only greedy decoding (temperature=0) is implemented");

        auto hidden_state = std::make_shared<CudaBuffer>(C::hidden_size() * sizeof(__nv_bfloat16));
        size_t row_bytes  = C::hidden_size() * sizeof(__nv_bfloat16);
        const __nv_bfloat16* row_src = reinterpret_cast<const __nv_bfloat16*>(embedding_weight->data)
                                        + static_cast<size_t>(input_tok_id) * C::hidden_size();
        checkCuda(cudaMemcpyAsync(hidden_state->data, row_src, row_bytes,
                                   cudaMemcpyDeviceToDevice, stream));

        for (uint32_t l = 0; l < C::num_layers(); ++l)
            layers[l]->forward(k_cache, v_cache, hidden_state, seq_len, stream);

        auto hidden_norm = std::make_shared<CudaBuffer>(C::hidden_size() * sizeof(__nv_bfloat16));
        final_layernorm.normalize_hidden_state(hidden_state, hidden_norm, stream);

        auto logits = std::make_shared<CudaBuffer>(C::vocab_size() * sizeof(__nv_bfloat16));
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            C::vocab_size(), C::hidden_size(),
            reinterpret_cast<__nv_bfloat16*>(embedding_weight->data),
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(hidden_norm->data),
            reinterpret_cast<__nv_bfloat16*>(logits->data), stream);

        ArgMax argmax_op(C::vocab_size());
        int32_t* dev_idx = argmax_op.bf16_argmax(logits, stream);

        int32_t next_token;
        checkCuda(cudaMemcpy(&next_token, dev_idx, sizeof(int32_t), cudaMemcpyDeviceToHost));
        return next_token;
    }
};
