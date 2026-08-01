# Phase 2 顶层设计 — 实体模块（草案）

Phase 2 交付的是 单 request 完整生成链（embed → N×Pre-LN block → final norm → lm_head → 采样），对外暴露的是 InferenceEngine（session）与 TransformerModel（权重）。

**状态**：设计图纸（未实现）。Phase 1 对照：[`phase1_lifecycle.md`](phase1_lifecycle.md)。

**组织原则**：Phase 2 按 **实体模块** 划分（Loader | Model | Engine | Sampler）；lifecycle 切面（资源、时序、forward、API）作为 **各模块内实现细节**，不再作为顶层 checklist 维度。

**目标**：单 request **加载权重 → prefill → decode → 采样出 token**；多 layer。batching / Paged KV 见 roadmap 2.7。

---

## 0. 全局：模块关系与端到端时序

### 0.1 四模块 + 调用依赖

~~~
WeightLoader                TransformerModel           InferenceEngine          Sampler / GenerateLoop
  读模型文件/fixture  ──►    GPU 权重容器      ◄──引用──  session + 编排  ◄──调用──  logits → token
  输出 host tensor           immutable                 KVCache(N)               greedy / …
  name → tensor              embed / lm_head 权重       prefill/decode 调度
                             LayerWeights[N]

Phase 1 复用：transformer_layer_linears_forward_device（Engine 内按 layer_idx 调用）
~~~

**依赖方向**：Loader → Model → Engine ← Sampler（Sampler 不持有 Model/Engine 内部状态）

### 0.2 端到端时序（跨模块）

~~~
[Loader] load(path) → host tensors
[Model]  model_create + fill_weights → TransformerModel on GPU
[Engine] engine_create(model)
         prefill / decode×N  （engine_forward）
[Sampler] logits → next_token  （GenerateLoop 驱动 decode 循环）
[Engine] engine_destroy
[Model]  model_destroy
~~~

### 0.3 与 Phase 1 的边界

| Phase 1 | Phase 2 |
|---------|---------|
| `TransformerRunner` = session + 单层权重 | **Engine** = session；**Model** = 权重 |
| `KVCache(num_layers=1)` | `KVCache(num_layers=N)` |
| 手传 host 权重 | **Loader** 读模型文件 |
| 无 embed / lm_head / 采样 | Model 含 embed/lm_head；**Sampler** 出 token |

Phase 1 `TransformerRunner` 保留为单层测试基准；生产由 Engine 替代。

---

## 1. 模块 — WeightLoader

**职责**：模型文件 → host tensor + name 映射；**不**持有 GPU session，**不**做 forward。

**规划路径**：`src/model/weight_loader.{h,cpp}`

| 切面 | 内容 |
|------|------|
| 输入 | 文件路径（safetensors / gguf / fixture 目录） |
| 输出 | `ModelConfig` 校验过的 tensor 表；供 Model H2D |
| 生命周期 | 无持久 GPU 对象；单次 load 调用栈 |
| API（规划） | `weight_loader_load_safetensors(...)` 等 |

**实现细节**
- [ ] fixture 路径（Phase 2 先期，不依赖完整格式）
- [ ] safetensors 解析 + name→tensor
- [ ] gguf（若目标模型需要）
- [ ] 加载单测（小 fixture vs numpy/torch）

---

## 2. 模块 — TransformerModel

**职责**：**immutable** GPU 权重与 model 级只读资源；被 Engine **引用**，不含 session / KV 状态。

**规划路径**：`src/model/transformer_model.{h,cpp}`、`model_config.h`

### 2.1 资源

~~~
TransformerModel
  ├── ModelConfig
  ├── RopeCosSinCache
  ├── LayerWeights[N]     每层 d_w_q … d_w_post
  ├── d_embed             [vocab, hidden]
  ├── d_lm_head           [hidden, vocab]（可与 embed tied）
  └── d_w_final_norm
~~~

### 2.2 生命周期

| 操作 | 行为 |
|------|------|
| `model_create` + Loader 填充 | H2D 全部权重；构建 rope cache |
| `model_destroy` | cudaFree 权重；**须在引用它的 Engine 全部 destroy 之后** |
| reset | 无（immutable） |

### 2.3 数据流（被 Engine 调用）

- **embed**：`d_token_ids → d_hidden`
- **lm_head**：末 hidden → `d_logits`（decode 要 logits 时）
- **layer 权重**：Engine 按 `layer_idx` 取 `LayerWeights[i]` 传入 Phase 1 layer 链
- **final norm**：`d_w_final_norm` 作用于末层 hidden

### 2.4 API 与测试（规划）

- [ ] `ModelConfig` POD + 校验
- [ ] `transformer_model_create` / `destroy`
- [ ] embed / lm_head device 算子（或 gather GEMM）+ `forward_host` 单测
- [ ] 2-layer fixture 权重 layout 单测

---

## 3. 模块 — InferenceEngine

**职责**：**session 边界**；prefill/decode 调度；多 layer forward 编排；维护 KV 与 `cache_len` / `d_pos`。

**规划路径**：`src/engine/inference_engine.{h,cpp}`

### 3.1 资源

~~~
InferenceEngine
  ├── TransformerModel*      只读引用
  ├── KVCache(num_layers=N)
  ├── BufferPool             hidden、logits、d_pos、FA staging…
  ├── cudaStream / cublasHandle
  └── SessionState           cache_len；next_pos

每步 ephemeral:
  InferenceForwardCtx        num_tokens, d_token_ids|d_hidden, d_pos, …
~~~

### 3.2 生命周期

| 操作 | 行为 |
|------|------|
| `engine_create(model)` | KVCache(N)、pool、stream；`cache_len=0` |
| `engine_reset` | `kv_cache_reset`；`cache_len=0`；**不**动 Model / pool |
| `engine_destroy` | 释放 KV、pool、stream；**不** destroy model |
| forward 每步 | 栈上 `InferenceForwardCtx` |

### 3.3 单步 forward（模块内核心数据流）

~~~
inference_engine_forward(engine, &ctx)
  embed（Model）或跳过（测试传 hidden）
  for layer = 0..N-1:
      transformer_layer_linears_forward_device(
          model->layer[layer], kv_cache, layer_idx, …)
      // 每层 append KV；此处不 advance_len
  kv_cache_advance_len(T)          // 全 layer 完成后一次
  final_norm（Model）
  lm_head（Model，可选）
~~~

prefill：`T>1`，`pos=[0..T-1]`。decode：`T=1`，`pos=[cache_len]`，`num_kv_tokens=L+1`。

### 3.4 从 Phase 1 泛化（Engine 内细节）

- [ ] `kv_cache_create(..., num_layers=N)`；append/cast 按 layer
- [ ] per-layer 权重指针；`layer_idx` 传入 layer 链
- [ ] 超出 `max_seq` → 报错

### 3.5 API 与测试（规划）

- [ ] C：`engine_create` / `destroy` / `reset` / `forward`
- [ ] Python：`inference_engine_me`；`forward_host` 仅测试
- [ ] `doc/inference_engine_device_api.md`
- [ ] 2-layer prefill/decode e2e；N=1 退化 Phase 1

---

## 4. 模块 — Sampler / GenerateLoop

**职责**：Engine 产出 logits 之后，**token 出环**；不持有 KV / 权重。

**规划路径**：`src/engine/generate_loop.*` 或 Python 薄封装（先期）

| 切面 | 内容 |
|------|------|
| 输入 | 末 token `d_logits` [vocab] |
| 输出 | `next_token_id` |
| 生命周期 | 无 GPU session；可纯 host |
| 行为 | prefill 一次 → 循环 decode → sampler → 停止条件 |

**实现细节**
- [ ] 末 token logits slice
- [ ] greedy（先）；temperature / top-k / top-p
- [ ] 短序列 generate e2e + EOS/stop

---

## 5. 并行分支（不改四模块对象模型）

- **MoE FFN**：替换 Model 内 layer FFN 路径
- **性能**：BufferPool 完善、FP16/INT4、CUDA Graph、profiling
- **2.7**：batching、Paged KV、Radix cache

---

## 6. 实现顺序（模块骨架 → 细节）

1. **WeightLoader** — fixture（可并行起步）
2. **TransformerModel** — 容器 + embed/lm_head + fixture 灌入
3. **InferenceEngine** — session + N-layer forward + prefill/decode
4. **Sampler / GenerateLoop** — 闭合 token 环
5. **WeightLoader** — safetensors 生产格式
6. **并行**：MoE、Graph、量化

**API 契约**（收工前）：`doc/inference_engine_device_api.md`（Engine 为主；Loader/Model load API 可同文档分节）。
