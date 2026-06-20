#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// MoE 的调度台，由模型决定，含有模型配置 & Graph、Eager 工具
typedef struct MoeRunner MoeRunner;

// 策略，用于调度
typedef enum MoeRunnerDispatch {
    MOE_RUNNER_DISPATCH_AUTO = 0, // 由 Ctx.is_decode 决定
    MOE_RUNNER_DISPATCH_EAGER = 1,
    MOE_RUNNER_DISPATCH_GRAPH = 2,
} MoeRunnerDispatch;

// 一次 forward 计算所需的数据包
typedef struct MoeRunnerForwardCtx {
    // 这一步的 Batch大小
    int num_tokens;
    int is_decode;
    void *stream;
    // 上一层 的输入 Device 指针
    const float *d_x;
    const float *d_logits;
    // 权重
    const float *d_w_gate;
    const float *d_w_up;
    const float *d_w_down;
    // 输出也写入 Device 指针
    float *d_y;
} MoeRunnerForwardCtx;

typedef struct MoeRunnerForwardResult {
    int used_graph;
} MoeRunnerForwardResult;

// --- Engine（session 级 + 每步 device forward）---
MoeRunner *moe_runner_create(int hidden_size, int intermediate_size, int num_experts, int top_k,
                             int enable_graph, void *stream);

void moe_runner_destroy(MoeRunner *runner);

void moe_runner_set_dispatch(MoeRunner *runner, MoeRunnerDispatch dispatch);

MoeRunnerDispatch moe_runner_get_dispatch(const MoeRunner *runner);

int moe_runner_capture_graph(MoeRunner *runner, int num_tokens);

int moe_runner_has_graph(const MoeRunner *runner, int num_tokens);

// 生产入口：ctx 中 d_* 已在 GPU，Runner 内 D2D + graph/eager launch
int moe_runner_forward(MoeRunner *runner, const MoeRunnerForwardCtx *ctx,
                       MoeRunnerForwardResult *result);

// --- 非 Engine 调用 ，仅测试用，host 指针 H2D -> forward -> D2H
int moe_runner_forward_host(MoeRunner *runner, const float *x_host, const float *logits_host,
                            const float *w_gate_host, const float *w_up_host,
                            const float *w_down_host, float *y_host, int num_tokens, int is_decode,
                            MoeRunnerForwardResult *result);

#ifdef __cplusplus
}
#endif
