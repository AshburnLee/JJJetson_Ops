#include "moe_cuda_graph_cache.h"

#include <stdio.h>
#include <unordered_map>

#include "cuda_utils.h"
#include "moe_pipeline_sota.h"

// 一个实例化对象是一个CUDA Graph instance
struct MoECudaGraphEntry {
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

// CUDA Graphinstance 容器，装 decode 时每种 shape 录一个 Graph
// num_tokens 一旦变化（即shape变化），buffer 大小和 kernel 规模都变，必须换一张 graph
// 对于 Decode：单 Request，num_tokens = 1, 同一个 Entry 反复 Replay
//            多 Request，引擎同时计算 多个Request，num_tokens=4，此时选择相应的 entry
// 对于 Prefill：num_tokens 几乎总是不同，每一种值都 capture，导致Graph实例化爆炸，这正是 CUDA Graph
// 的缺点
//              此时考虑 eager 模式
// Padding Bucket, num_tokens pad到4，实际有效1~4
struct MoECudaGraphCache {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_experts = 0;
    int top_k = 0;
    // shared between all Graph intance
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
    // batch 大小一旦变，buffer 和 launch 参数就变，必须单独 capture 一张图
    std::unordered_map<int, MoECudaGraphEntry> entries;
};

static void moe_cuda_graph_entry_destroy(MoECudaGraphEntry *entry) {
    if (entry == nullptr) {
        return;
    }
    if (entry->graph_exec != nullptr) {
        cudaGraphExecDestroy(entry->graph_exec);
        entry->graph_exec = nullptr;
    }
    if (entry->graph != nullptr) {
        cudaGraphDestroy(entry->graph);
        entry->graph = nullptr;
    }
    if (entry->buffers != nullptr) {
        moe_pipeline_sota_buffers_destroy(entry->buffers);
        entry->buffers = nullptr;
    }
}

extern "C" MoECudaGraphCache *moe_cuda_graph_cache_create(int hidden_size, int intermediate_size,
                                                          int num_experts, int top_k,
                                                          void *stream) {
    auto *cache = new MoECudaGraphCache{};
    cache->hidden_size = hidden_size;
    cache->intermediate_size = intermediate_size;
    cache->num_experts = num_experts;
    cache->top_k = top_k;

    if (stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&cache->stream, cudaStreamNonBlocking));
        cache->owns_stream = true;
    } else {
        cache->stream = static_cast<cudaStream_t>(stream);
        cache->owns_stream = false;
    }

    return cache;
}

extern "C" void moe_cuda_graph_cache_destroy(MoECudaGraphCache *cache) {
    if (cache == nullptr) {
        return;
    }

    for (auto &kv : cache->entries) {
        moe_cuda_graph_entry_destroy(&kv.second);
    }
    cache->entries.clear();

    if (cache->owns_stream && cache->stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(cache->stream));
    }

    delete cache;
}

extern "C" int moe_cuda_graph_cache_has(const MoECudaGraphCache *cache, int num_tokens) {
    if (cache == nullptr) {
        return 0;
    }
    const auto it = cache->entries.find(num_tokens);
    return (it != cache->entries.end() && it->second.graph_exec != nullptr) ? 1 : 0;
}

// 返回指定 shape 的 graph buffer
extern "C" MoePipelineSotaBuffers *moe_cuda_graph_cache_buffers_get(MoECudaGraphCache *cache,
                                                                    int num_tokens) {
    if (cache == nullptr) {
        return nullptr;
    }
    const auto it = cache->entries.find(num_tokens);
    if (it == cache->entries.end()) {
        return nullptr;
    }
    return it->second.buffers;
}

// 对特定shape capture 并创建 cuda Graph instance
extern "C" int moe_cuda_graph_cache_capture(MoECudaGraphCache *cache, int num_tokens) {
    if (cache == nullptr || num_tokens <= 0) {
        return -1;
    }

    // 对于这个shape，有 cache 的 Graph，就不需要再capture了
    if (moe_cuda_graph_cache_has(cache, num_tokens)) {
        return 0;
    }

    MoECudaGraphEntry entry{};
    // create后，所有Device指针地址不变，这是CUDA Graph的基本要求
    entry.buffers = moe_pipeline_sota_buffers_create(
        num_tokens, cache->hidden_size, cache->intermediate_size, cache->num_experts, cache->top_k);

    cudaGraph_t captured_graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(cache->stream, cudaStreamCaptureModeGlobal));
    // 录制，第一次执行 moe pipline，CUDA Graph 机制上是必须的，其结果不作为MoE 结果
    moe_pipeline_sota_forward_device(static_cast<void *>(cache->stream), entry.buffers);
    const cudaError_t end_err = cudaStreamEndCapture(cache->stream, &captured_graph);
    if (end_err != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed: %s\n", cudaGetErrorString(end_err));
        moe_cuda_graph_entry_destroy(&entry);
        return -1;
    }

    cudaGraphExec_t graph_exec = nullptr;
    // 创建实例化对象并优化
    const cudaError_t inst_err =
        cudaGraphInstantiate(&graph_exec, captured_graph, nullptr, nullptr, 0);
    if (inst_err != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed: %s\n", cudaGetErrorString(inst_err));
        cudaGraphDestroy(captured_graph);
        moe_cuda_graph_entry_destroy(&entry);
        return -1;
    }

    entry.graph = captured_graph;
    entry.graph_exec = graph_exec;
    cache->entries[num_tokens] = entry;
    return 0;
}

extern "C" int moe_cuda_graph_cache_replay(MoECudaGraphCache *cache, int num_tokens) {
    if (cache == nullptr) {
        return -1;
    }

    const auto it = cache->entries.find(num_tokens);
    if (it == cache->entries.end() || it->second.graph_exec == nullptr) {
        fprintf(stderr, "moe_cuda_graph_cache_replay: no graph for num_tokens=%d\n", num_tokens);
        return -1;
    }
    // launch 的是 cudaGraphExec_t 对象
    CUDA_CHECK(cudaGraphLaunch(it->second.graph_exec, cache->stream));
    return 0;
}

// 禁止在推理时调用，仅作为 测试
extern "C" int moe_cuda_graph_cache_run(MoECudaGraphCache *cache, int num_tokens,
                                        const float *x_host, const float *logits_host,
                                        const float *w_gate_host, const float *w_up_host,
                                        const float *w_down_host, float *y_host) {
    if (cache == nullptr) {
        return -1;
    }

    if (!moe_cuda_graph_cache_has(cache, num_tokens)) {
        if (moe_cuda_graph_cache_capture(cache, num_tokens) != 0) {
            return -1;
        }
    }

    MoePipelineSotaBuffers *buffers = moe_cuda_graph_cache_buffers_get(cache, num_tokens);
    if (buffers == nullptr) {
        return -1;
    }

    const size_t x_bytes = static_cast<size_t>(num_tokens) * cache->hidden_size * sizeof(float);
    const size_t logits_bytes =
        static_cast<size_t>(num_tokens) * cache->num_experts * sizeof(float);
    const size_t w_gu_bytes = static_cast<size_t>(cache->num_experts) * cache->intermediate_size *
                              cache->hidden_size * sizeof(float);
    const size_t w_d_bytes = static_cast<size_t>(cache->num_experts) * cache->hidden_size *
                             cache->intermediate_size * sizeof(float);

    // 每一次的Replay 都需要 更新指针中真实值，
    // inference 中，memcpy 常是 GPU -> GPU，即上一层的输出已在Device
    CUDA_CHECK(
        cudaMemcpyAsync(buffers->d_x, x_host, x_bytes, cudaMemcpyHostToDevice, cache->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_logits, logits_host, logits_bytes, cudaMemcpyHostToDevice,
                               cache->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_gate, w_gate_host, w_gu_bytes, cudaMemcpyHostToDevice,
                               cache->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_up, w_up_host, w_gu_bytes, cudaMemcpyHostToDevice,
                               cache->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_down, w_down_host, w_d_bytes, cudaMemcpyHostToDevice,
                               cache->stream));

    if (moe_cuda_graph_cache_replay(cache, num_tokens) != 0) {
        return -1;
    }

    CUDA_CHECK(
        cudaMemcpyAsync(y_host, buffers->d_y, x_bytes, cudaMemcpyDeviceToHost, cache->stream));
    CUDA_CHECK(cudaStreamSynchronize(cache->stream));
    return 0;
}
