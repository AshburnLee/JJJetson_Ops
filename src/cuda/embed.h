#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 模型入口：你给它一串 token id，它就上 embed 词表里把每个 id 对应的那一行向量整行抄出来。
// 这是查表（gather），不是矩阵乘。词表 row-major 存，一行一个词；抄完按列塞进 hidden，
// col-major [hidden, num_tokens]，后面 layer、lm_head 都吃这个 layout。
//
// 比方说 hidden=3，这一步只有一个 token，id 是 2。词表第 2 行若是 [0.1, 0.2, 0.3]，
// 那出来的 hidden 第 0 列就是 [0.1, 0.2, 0.3]，原封不动。再来 token 就再抄一行填下一列。
// kernel 怎么并行见 embed.cu。
int embed_forward_device(void *stream, const float *d_embed, const int *d_token_ids,
                         float *d_hidden, int hidden_size, int num_tokens);

// ======================== 仅供 Python 测试 ================================
// 单测用：numpy 数组在 host 上，帮你 malloc、H2D、调上面的 device、再 D2H 回来。
// 真跑推理别走这条，Engine 用 transformer_model_embed_forward_device。
void embed_forward_host(const float *embed_host, const int *token_ids_host, float *hidden_host,
                        int hidden_size, int vocab_size, int num_tokens);

#ifdef __cplusplus
}
#endif
