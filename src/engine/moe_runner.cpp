#include "moe_runner.h"

#include <stdio.h>
#include <unordered_map>

#include "cuda_utils.h"
#include "moe_cuda_graph_instance_container.h"
#include "moe_pipeline_sota.h"

// MoE 的调度台，由模型决定的模型配置 & Graph、Eager 工具
struct MoeRunner {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_experts = 0;
    int top_k = 0;
    int enable_graph = 0;
    MoeRunnerDispatch dispatch = MOE_RUNNER_DISPATCH_AUTO;
    // 整条 MoE 工作队列
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
    // 1. Graph mode：按 num_tokens 管理 MoECudaGraphInstance
    MoECudaGraphInstanceContainer *graph_instance_container = nullptr;
    // 2. eager model 时用持久的 buffer
    // 每一个 Batch 大小对应一个buffer
    std::unordered_map<int, MoePipelineSotaBuffers *> eager_buffers;
};

static MoePipelineSotaBuffers *moe_runner_eager_buffers_get_or_create(MoeRunner *runner,
                                                                      int num_tokens) {
    const auto it = runner->eager_buffers.find(num_tokens);
    if (it != runner->eager_buffers.end()) {
        return it->second;
    }

    MoePipelineSotaBuffers *buffers =
        moe_pipeline_sota_buffers_create(num_tokens, runner->hidden_size, runner->intermediate_size,
                                         runner->num_experts, runner->top_k);
    runner->eager_buffers[num_tokens] = buffers;
    return buffers;
}

// 推理生产中，D2D 的 memcpy
static void moe_runner_copy_inputs(cudaStream_t stream, MoePipelineSotaBuffers *buffers,
                                   const MoeRunnerForwardCtx *ctx, int hidden_size, int num_experts,
                                   int intermediate_size) {
    const int num_tokens = ctx->num_tokens;
    const size_t x_bytes = static_cast<size_t>(num_tokens) * hidden_size * sizeof(float);
    const size_t logits_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
    const size_t w_gu_bytes =
        static_cast<size_t>(num_experts) * intermediate_size * hidden_size * sizeof(float);
    const size_t w_d_bytes =
        static_cast<size_t>(num_experts) * hidden_size * intermediate_size * sizeof(float);

    if (ctx->d_x != buffers->d_x) {
        CUDA_CHECK(
            cudaMemcpyAsync(buffers->d_x, ctx->d_x, x_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    if (ctx->d_logits != buffers->d_logits) {
        CUDA_CHECK(cudaMemcpyAsync(buffers->d_logits, ctx->d_logits, logits_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (ctx->d_w_gate != buffers->d_w_gate) {
        CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_gate, ctx->d_w_gate, w_gu_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (ctx->d_w_up != buffers->d_w_up) {
        CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_up, ctx->d_w_up, w_gu_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (ctx->d_w_down != buffers->d_w_down) {
        CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_down, ctx->d_w_down, w_d_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

// 推理生产中，D2D 的 memcpy
static void moe_runner_copy_output(cudaStream_t stream, float *d_y, MoePipelineSotaBuffers *buffers,
                                   int num_tokens, int hidden_size) {
    if (d_y == buffers->d_y) {
        return;
    }
    const size_t x_bytes = static_cast<size_t>(num_tokens) * hidden_size * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d_y, buffers->d_y, x_bytes, cudaMemcpyDeviceToDevice, stream));
}

// 计算路径的选择
static bool moe_runner_should_use_graph(const MoeRunner *runner, const MoeRunnerForwardCtx *ctx) {
    if (runner == nullptr || ctx == nullptr || !runner->enable_graph ||
        runner->graph_instance_container == nullptr) {
        return false;
    }

    switch (runner->dispatch) {
    case MOE_RUNNER_DISPATCH_EAGER:
        return false;
    case MOE_RUNNER_DISPATCH_GRAPH:
        return true;
    case MOE_RUNNER_DISPATCH_AUTO:
    default:
        return ctx->is_decode != 0; // 1 != 0 -> true -> use_graph
    }
}

static int moe_runner_forward_graph(MoeRunner *runner, const MoeRunnerForwardCtx *ctx,
                                    cudaStream_t stream) {
    const int num_tokens = ctx->num_tokens;

    if (!moe_cuda_graph_instance_container_has(runner->graph_instance_container, num_tokens)) {
        if (moe_cuda_graph_instance_container_capture(runner->graph_instance_container,
                                                      num_tokens) != 0) {
            return -1;
        }
    }

    MoePipelineSotaBuffers *buffers =
        moe_cuda_graph_instance_container_buffers_get(runner->graph_instance_container, num_tokens);
    if (buffers == nullptr) {
        return -1;
    }

    moe_runner_copy_inputs(stream, buffers, ctx, runner->hidden_size, runner->num_experts,
                           runner->intermediate_size);

    if (moe_cuda_graph_instance_container_replay(runner->graph_instance_container, num_tokens) !=
        0) {
        return -1;
    }

    moe_runner_copy_output(stream, ctx->d_y, buffers, num_tokens, runner->hidden_size);
    return 0;
}

static int moe_runner_forward_eager(MoeRunner *runner, const MoeRunnerForwardCtx *ctx,
                                    cudaStream_t stream) {
    MoePipelineSotaBuffers *buffers =
        moe_runner_eager_buffers_get_or_create(runner, ctx->num_tokens);
    if (buffers == nullptr) {
        return -1;
    }

    moe_runner_copy_inputs(stream, buffers, ctx, runner->hidden_size, runner->num_experts,
                           runner->intermediate_size);
    moe_pipeline_sota_forward_device(static_cast<void *>(stream), buffers);
    moe_runner_copy_output(stream, ctx->d_y, buffers, ctx->num_tokens, runner->hidden_size);
    return 0;
}

extern "C" MoeRunner *moe_runner_create(int hidden_size, int intermediate_size, int num_experts,
                                        int top_k, int enable_graph, void *stream) {
    auto *runner = new MoeRunner{};
    runner->hidden_size = hidden_size;
    runner->intermediate_size = intermediate_size;
    runner->num_experts = num_experts;
    runner->top_k = top_k;
    runner->enable_graph = enable_graph != 0;

    if (stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&runner->stream, cudaStreamNonBlocking));
        runner->owns_stream = true;
    } else {
        runner->stream = static_cast<cudaStream_t>(stream);
        runner->owns_stream = false;
    }

    if (runner->enable_graph) {
        runner->graph_instance_container =
            moe_cuda_graph_instance_container_create(hidden_size, intermediate_size, num_experts,
                                                     top_k, static_cast<void *>(runner->stream));
        if (runner->graph_instance_container == nullptr) {
            moe_runner_destroy(runner);
            return nullptr;
        }
    }

    return runner;
}

extern "C" void moe_runner_destroy(MoeRunner *runner) {
    if (runner == nullptr) {
        return;
    }

    for (auto &kv : runner->eager_buffers) {
        moe_pipeline_sota_buffers_destroy(kv.second);
    }
    runner->eager_buffers.clear();

    if (runner->graph_instance_container != nullptr) {
        moe_cuda_graph_instance_container_destroy(runner->graph_instance_container);
        runner->graph_instance_container = nullptr;
    }

    if (runner->owns_stream && runner->stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(runner->stream));
    }

    delete runner;
}

extern "C" void moe_runner_set_dispatch(MoeRunner *runner, MoeRunnerDispatch dispatch) {
    if (runner != nullptr) {
        runner->dispatch = dispatch;
    }
}

extern "C" MoeRunnerDispatch moe_runner_get_dispatch(const MoeRunner *runner) {
    if (runner == nullptr) {
        return MOE_RUNNER_DISPATCH_AUTO;
    }
    return runner->dispatch;
}

extern "C" int moe_runner_capture_graph(MoeRunner *runner, int num_tokens) {
    if (runner == nullptr || !runner->enable_graph || runner->graph_instance_container == nullptr) {
        return -1;
    }
    return moe_cuda_graph_instance_container_capture(runner->graph_instance_container, num_tokens);
}

extern "C" int moe_runner_has_graph(const MoeRunner *runner, int num_tokens) {
    if (runner == nullptr || !runner->enable_graph || runner->graph_instance_container == nullptr) {
        return 0;
    }
    return moe_cuda_graph_instance_container_has(runner->graph_instance_container, num_tokens);
}

// 生产推理的入口，D2D -> 选择 graph/eager -> D2D
extern "C" int moe_runner_forward(MoeRunner *runner, const MoeRunnerForwardCtx *ctx,
                                  MoeRunnerForwardResult *result) {
    if (runner == nullptr || ctx == nullptr || ctx->num_tokens <= 0 || ctx->d_x == nullptr ||
        ctx->d_logits == nullptr || ctx->d_w_gate == nullptr || ctx->d_w_up == nullptr ||
        ctx->d_w_down == nullptr || ctx->d_y == nullptr) {
        return -1;
    }

    cudaStream_t stream = runner->stream;
    if (ctx->stream != nullptr) {
        stream = static_cast<cudaStream_t>(ctx->stream);
    }

    const bool use_graph = moe_runner_should_use_graph(runner, ctx);
    int rc = -1;
    if (use_graph) {
        rc = moe_runner_forward_graph(runner, ctx, stream);
        if (rc != 0 && runner->dispatch == MOE_RUNNER_DISPATCH_AUTO) {
            fprintf(stderr, "moe_runner_forward: graph path failed, fallback to eager\n");
            rc = moe_runner_forward_eager(runner, ctx, stream);
            if (result != nullptr) {
                result->used_graph = 0;
            }
            return rc;
        }
    } else {
        rc = moe_runner_forward_eager(runner, ctx, stream);
    }

    if (result != nullptr) {
        result->used_graph = (rc == 0 && use_graph) ? 1 : 0;
    }
    return rc;
}

// 仅测试用
extern "C" int moe_runner_forward_host(MoeRunner *runner, const float *x_host,
                                       const float *logits_host, const float *w_gate_host,
                                       const float *w_up_host, const float *w_down_host,
                                       float *y_host, int num_tokens, int is_decode,
                                       MoeRunnerForwardResult *result) {
    if (runner == nullptr || num_tokens <= 0) {
        return -1;
    }

    MoePipelineSotaBuffers *staging = moe_runner_eager_buffers_get_or_create(runner, num_tokens);
    if (staging == nullptr) {
        return -1;
    }

    const size_t x_bytes = static_cast<size_t>(num_tokens) * runner->hidden_size * sizeof(float);
    const size_t logits_bytes =
        static_cast<size_t>(num_tokens) * runner->num_experts * sizeof(float);
    const size_t w_gu_bytes = static_cast<size_t>(runner->num_experts) * runner->intermediate_size *
                              runner->hidden_size * sizeof(float);
    const size_t w_d_bytes = static_cast<size_t>(runner->num_experts) * runner->hidden_size *
                             runner->intermediate_size * sizeof(float);

    CUDA_CHECK(
        cudaMemcpyAsync(staging->d_x, x_host, x_bytes, cudaMemcpyHostToDevice, runner->stream));
    CUDA_CHECK(cudaMemcpyAsync(staging->d_logits, logits_host, logits_bytes, cudaMemcpyHostToDevice,
                               runner->stream));
    CUDA_CHECK(cudaMemcpyAsync(staging->d_w_gate, w_gate_host, w_gu_bytes, cudaMemcpyHostToDevice,
                               runner->stream));
    CUDA_CHECK(cudaMemcpyAsync(staging->d_w_up, w_up_host, w_gu_bytes, cudaMemcpyHostToDevice,
                               runner->stream));
    CUDA_CHECK(cudaMemcpyAsync(staging->d_w_down, w_down_host, w_d_bytes, cudaMemcpyHostToDevice,
                               runner->stream));

    MoeRunnerForwardCtx ctx{};
    ctx.num_tokens = num_tokens;
    ctx.is_decode = is_decode;
    ctx.stream = static_cast<void *>(runner->stream);
    ctx.d_x = staging->d_x;
    ctx.d_logits = staging->d_logits;
    ctx.d_w_gate = staging->d_w_gate;
    ctx.d_w_up = staging->d_w_up;
    ctx.d_w_down = staging->d_w_down;
    ctx.d_y = staging->d_y;

    const int rc = moe_runner_forward(runner, &ctx, result);
    if (rc != 0) {
        return rc;
    }

    CUDA_CHECK(
        cudaMemcpyAsync(y_host, staging->d_y, x_bytes, cudaMemcpyDeviceToHost, runner->stream));
    CUDA_CHECK(cudaStreamSynchronize(runner->stream));
    return 0;
}
