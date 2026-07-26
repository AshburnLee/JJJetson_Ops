#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ElementwiseBinaryOp {
    ELEMENTWISE_ADD = 0,
    ELEMENTWISE_SUB = 1,
    ELEMENTWISE_MUL = 2,
    ELEMENTWISE_DIV = 3,
} ElementwiseBinaryOp;

// 通用二元逐元素：out[i] = op(a[i], b[i])。
// a/b/out 为连续 flat buffer；d_out 可与 d_a 或 d_b 相同（in-place）。
// n_elem 通常为 hidden_size * num_tokens（col-major hidden 展平）。
int elementwise_binary_forward_device(void *stream, ElementwiseBinaryOp op, const float *d_a,
                                      const float *d_b, float *d_out, int n_elem);

int elementwise_add_forward_device(void *stream, const float *d_a, const float *d_b, float *d_out,
                                   int n_elem);

int elementwise_sub_forward_device(void *stream, const float *d_a, const float *d_b, float *d_out,
                                   int n_elem);

int elementwise_mul_forward_device(void *stream, const float *d_a, const float *d_b, float *d_out,
                                   int n_elem);

int elementwise_div_forward_device(void *stream, const float *d_a, const float *d_b, float *d_out,
                                   int n_elem);

// 测试：host numpy H2D → elementwise_binary_forward_device → D2H
void elementwise_binary_forward_host(const float *a_host, const float *b_host, float *out_host,
                                     int n_elem, ElementwiseBinaryOp op);

#ifdef __cplusplus
}
#endif
