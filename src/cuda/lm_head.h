#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// untied：embed 和 lm_head 各存各的（tie_word_embeddings=0）。吃最后一层 hidden，吐出词表 logits。
// 你可以想成：hidden 跟 lm_head 权重做内积，词表有几个词就有几个分数。
//
// 最小例子：hidden 2 维、词表 3 个词，只看第 0 个 token。hidden[:,0] 是 [1.0, 2.0]；
// lm_head 里 hidden 第 0 维对三个词的权重 [0.1, 0.2, 0.3]，第 1 维是 [0.4, 0.5, 0.6]。
// 词 0 的分数 1.0*0.1 + 2.0*0.4 = 0.9，词 1 得 1.2，词 2 得 1.5，所以 logits[:,0] =
// [0.9, 1.2, 1.5]。 cuBLAS 怎么拼见 lm_head.cu。
void untied_lm_head_forward_device(void *stream, void *cublas_handle, const float *d_lm_head,
                                   const float *d_hidden, float *d_logits, int hidden_size,
                                   int vocab_size, int num_tokens);

// tied：不单独存 lm_head，输出打分直接复用 embed 那张表（tie_word_embeddings=1）。
//
// 同样 hidden[:,0]=[1.0, 2.0]，词表 2 个词。embed 里词 0 向量 [0.1, 0.2]，词 1 是 [0.3, 0.4]。
// 词 0 分数 1.0*0.1 + 2.0*0.2 = 0.5，词 1 得 1.1，logits[:,0] = [0.5, 1.1]。
// 认字和猜词用的是同一套向量，只是这次拿 hidden 去跟每个词的 embed 行做点积。
void tied_lm_head_forward_device(void *stream, void *cublas_handle, const float *d_embed,
                                 const float *d_hidden, float *d_logits, int hidden_size,
                                 int vocab_size, int num_tokens);

// ======================== 仅供 Python 测试 ================================
// 单测用：host 上的 embed/hidden 拷进 GPU，调 tied 路径算 logits 再拷回来。
// 真跑推理走 transformer_model_lm_head_forward_device，它会帮你选 tied 还是 untied。
void tied_lm_head_forward_host(const float *embed_host, const float *hidden_host,
                               float *logits_host, int hidden_size, int vocab_size, int num_tokens);

#ifdef __cplusplus
}
#endif
