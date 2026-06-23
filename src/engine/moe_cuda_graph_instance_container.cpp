#include "moe_cuda_graph_instance_container.h"

#include <stdio.h>
#include <unordered_map>

#include "cuda_utils.h"
#include "moe_pipeline_sota.h"

// 某一 num_tokens 对应的一张 CUDA Graph：固定 buffer + capture 结果 + 可 launch 的 exec
struct MoECudaGraphInstance {
    // buffer 必须和 Graph 绑定，地址不能变
    MoePipelineSotaBuffers *buffers = nullptr;
    // for capture，成功不表示可实例化
    // capture 只得到 DAG，表示计算蓝图，可编辑检查；改变cudaGraph_t 后可
    // cudaGraphExecUpdate，不必每次从零 capture 用完可释放
    cudaGraph_t graph = nullptr;
    // for 实例化，graph_exec 非空表示已经实例化，可launch，
    // 可执行体，包括依赖性分析，合法性检查，驱动测优化，生成高效 launch 实例
    // 长期持有
    cudaGraphExec_t graph_exec = nullptr;
};

// 按 num_tokens 管理多张 MoECudaGraphInstance
// num_tokens 一旦变化（即shape变化），buffer 大小和 kernel 规模都变，必须换一张 graph
// 对于 Decode：单 Request，num_tokens = 1, 同一个 GraphInstance 反复 Replay
//            多 Request，引擎同时计算 多个Request，num_tokens=4，此时选择相应的 GraphInstance
// 对于 Prefill：num_tokens 几乎总是不同，每一种值都 capture，导致Graph实例化爆炸，这正是 CUDA Graph
// 的缺点
//              此时考虑 eager 模式
// Padding Bucket, num_tokens pad到4，实际有效1~4
struct MoECudaGraphInstanceContainer {
    // int 属性由模型结构决定，容器创建时写入，中途不会被改变
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_experts = 0;
    int top_k = 0;
    // shared between all Graph instances
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
    // 本步 token 数 num_tokens 一旦变，buffer 和 launch 参数就变，必须单独 capture 一张图
    std::unordered_map<int, MoECudaGraphInstance> graph_instances;
};

static void moe_cuda_graph_instance_destroy(MoECudaGraphInstance *graph_instance) {
    if (graph_instance == nullptr) {
        return;
    }
    if (graph_instance->graph_exec != nullptr) {
        cudaGraphExecDestroy(graph_instance->graph_exec);
        graph_instance->graph_exec = nullptr;
    }
    if (graph_instance->graph != nullptr) {
        cudaGraphDestroy(graph_instance->graph);
        graph_instance->graph = nullptr;
    }
    if (graph_instance->buffers != nullptr) {
        moe_pipeline_sota_buffers_destroy(graph_instance->buffers);
        graph_instance->buffers = nullptr;
    }
}

extern "C" MoECudaGraphInstanceContainer *
moe_cuda_graph_instance_container_create(int hidden_size, int intermediate_size, int num_experts,
                                         int top_k, void *stream) {
    auto *container = new MoECudaGraphInstanceContainer{};
    container->hidden_size = hidden_size;
    container->intermediate_size = intermediate_size;
    container->num_experts = num_experts;
    container->top_k = top_k;

    if (stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&container->stream, cudaStreamNonBlocking));
        container->owns_stream = true;
    } else {
        container->stream = static_cast<cudaStream_t>(stream);
        container->owns_stream = false;
    }

    return container;
}

extern "C" void
moe_cuda_graph_instance_container_destroy(MoECudaGraphInstanceContainer *container) {
    if (container == nullptr) {
        return;
    }

    for (auto &kv : container->graph_instances) {
        moe_cuda_graph_instance_destroy(&kv.second);
    }
    container->graph_instances.clear();

    if (container->owns_stream && container->stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(container->stream));
    }

    delete container;
}

extern "C" int moe_cuda_graph_instance_container_has(const MoECudaGraphInstanceContainer *container,
                                                     int num_tokens) {
    if (container == nullptr) {
        return 0;
    }
    const auto it = container->graph_instances.find(num_tokens);
    return (it != container->graph_instances.end() && it->second.graph_exec != nullptr) ? 1 : 0;
}

extern "C" MoePipelineSotaBuffers *
moe_cuda_graph_instance_container_buffers_get(MoECudaGraphInstanceContainer *container,
                                              int num_tokens) {
    if (container == nullptr) {
        return nullptr;
    }
    const auto it = container->graph_instances.find(num_tokens);
    if (it == container->graph_instances.end()) {
        return nullptr;
    }
    return it->second.buffers;
}

extern "C" int moe_cuda_graph_instance_container_capture(MoECudaGraphInstanceContainer *container,
                                                         int num_tokens) {
    if (container == nullptr || num_tokens <= 0) {
        return -1;
    }

    if (moe_cuda_graph_instance_container_has(container, num_tokens)) {
        return 0;
    }

    MoECudaGraphInstance graph_instance{};
    graph_instance.buffers = moe_pipeline_sota_buffers_create(
        num_tokens, container->hidden_size, container->intermediate_size, container->num_experts,
        container->top_k);

    cudaGraph_t captured_graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(container->stream, cudaStreamCaptureModeGlobal));
    moe_pipeline_sota_forward_device(static_cast<void *>(container->stream),
                                     graph_instance.buffers);
    const cudaError_t end_err = cudaStreamEndCapture(container->stream, &captured_graph);
    if (end_err != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed: %s\n", cudaGetErrorString(end_err));
        moe_cuda_graph_instance_destroy(&graph_instance);
        return -1;
    }

    cudaGraphExec_t graph_exec = nullptr;
    const cudaError_t inst_err =
        cudaGraphInstantiate(&graph_exec, captured_graph, nullptr, nullptr, 0);
    if (inst_err != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed: %s\n", cudaGetErrorString(inst_err));
        cudaGraphDestroy(captured_graph);
        moe_cuda_graph_instance_destroy(&graph_instance);
        return -1;
    }

    graph_instance.graph = captured_graph;
    graph_instance.graph_exec = graph_exec;
    container->graph_instances[num_tokens] = graph_instance;
    return 0;
}

extern "C" int moe_cuda_graph_instance_container_replay(MoECudaGraphInstanceContainer *container,
                                                        int num_tokens) {
    if (container == nullptr) {
        return -1;
    }

    const auto it = container->graph_instances.find(num_tokens);
    if (it == container->graph_instances.end() || it->second.graph_exec == nullptr) {
        fprintf(stderr, "moe_cuda_graph_instance_container_replay: no graph for num_tokens=%d\n",
                num_tokens);
        return -1;
    }
    CUDA_CHECK(cudaGraphLaunch(it->second.graph_exec, container->stream));
    return 0;
}

extern "C" int moe_cuda_graph_instance_container_run(MoECudaGraphInstanceContainer *container,
                                                     int num_tokens, const float *x_host,
                                                     const float *logits_host,
                                                     const float *w_gate_host,
                                                     const float *w_up_host,
                                                     const float *w_down_host, float *y_host) {
    if (container == nullptr) {
        return -1;
    }

    if (!moe_cuda_graph_instance_container_has(container, num_tokens)) {
        if (moe_cuda_graph_instance_container_capture(container, num_tokens) != 0) {
            return -1;
        }
    }

    MoePipelineSotaBuffers *buffers =
        moe_cuda_graph_instance_container_buffers_get(container, num_tokens);
    if (buffers == nullptr) {
        return -1;
    }

    const size_t x_bytes = static_cast<size_t>(num_tokens) * container->hidden_size * sizeof(float);
    const size_t logits_bytes =
        static_cast<size_t>(num_tokens) * container->num_experts * sizeof(float);
    const size_t w_gu_bytes = static_cast<size_t>(container->num_experts) *
                              container->intermediate_size * container->hidden_size * sizeof(float);
    const size_t w_d_bytes = static_cast<size_t>(container->num_experts) * container->hidden_size *
                             container->intermediate_size * sizeof(float);

    CUDA_CHECK(
        cudaMemcpyAsync(buffers->d_x, x_host, x_bytes, cudaMemcpyHostToDevice, container->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_logits, logits_host, logits_bytes, cudaMemcpyHostToDevice,
                               container->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_gate, w_gate_host, w_gu_bytes, cudaMemcpyHostToDevice,
                               container->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_up, w_up_host, w_gu_bytes, cudaMemcpyHostToDevice,
                               container->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_down, w_down_host, w_d_bytes, cudaMemcpyHostToDevice,
                               container->stream));

    if (moe_cuda_graph_instance_container_replay(container, num_tokens) != 0) {
        return -1;
    }

    CUDA_CHECK(
        cudaMemcpyAsync(y_host, buffers->d_y, x_bytes, cudaMemcpyDeviceToHost, container->stream));
    CUDA_CHECK(cudaStreamSynchronize(container->stream));
    return 0;
}
