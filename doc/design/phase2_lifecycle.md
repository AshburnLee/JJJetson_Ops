# Phase 2 顶层设计 — 实体模块（草案）

Phase 2 交付的是 单 request 完整生成链（embed → N×Pre-LN block → final norm → lm_head → 采样），对外暴露的是 InferenceEngine（session）与 TransformerModel（权重）。

**状态**：设计图纸（未实现）。Phase 1 对照：[`phase1_lifecycle.md`](phase1_lifecycle.md)（同目录）。

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

~~~
Phase 1                              Phase 2
─────────────────────────────────────────────────────────────────────────
TransformerRunner = session + 单层权重   Engine = session；Model = 权重
KVCache(num_layers=1)                 KVCache(num_layers=N)
手传 host 权重                         Loader 读模型文件
无 embed / lm_head / 采样              Model 含 embed/lm_head；Sampler 出 token
~~~

Phase 1 `TransformerRunner` 保留为单层测试基准；生产由 Engine 替代。

---

## 1. 模块 — WeightLoader

**职责**：模型文件 → host tensor + name 映射；**不**持有 GPU session，**不**做 forward。

**规划路径**：`src/model/weight_loader.{h,cpp}`、`model_config.h`

~~~
切面        内容
──────────────────────────────────────────────────────────────────
输入        文件路径（safetensors / gguf / fixture 目录）
输出        ModelConfig 校验过的 tensor 表；供 Model H2D
生命周期    无持久 GPU 对象；单次 load 调用栈
API         weight_loader_load_fixture / weight_loader_load_safetensors /
            weight_loader_load_safetensors_hf_llama；weight_load_result_*
~~~

**骨架（已实现）**
- [x] `ModelConfig` POD + `model_config_validate`
- [x] `HostTensor` / `WeightLoadResult` + `find` / `destroy`
- [x] load API 声明 + 桩实现（返回 -1）；Python `weight_loader_me`

**实现细节**
- [x] fixture 路径（`config.txt` + `manifest.txt` + `.f32`）；格式说明见 [`doc/guide/fixture_structure.md`](../guide/fixture_structure.md)
- [ ] **safetensors**（分步；见 roadmap 模块 1）
  - **定位**：safetensors 是真实模型权重的目标格式；当前完成了可读 safetensors + 可验证（步骤 1/2）。真实推理场景需从 HF safetensors + config 加载实际权重（步骤 3/4）。Orin 显存不够全量 TinyLlama F32，用 1~2 layer 切片
  - [x] 1 — Loader 只读格式（解析 `.safetensors` -> `HostTensor`；`safetensors_reader.cpp`；F32 only）
  - [x] 2 — fixture roundtrip（fixture 导出 `.safetensors` + `test_fixture_safetensors_roundtrip`）
  - [x] 3 — HF 名映射（`hf_llama_weight_map.cpp` + `load_safetensors_hf_llama`；见 [`doc/guide/hf_llama_weight_map.md`](../guide/hf_llama_weight_map.md)）
  - [ ] 4 — 真模型验证（TinyLlama / 1~2 layer 切片）
        路径（切片 load + Engine 短 prefill smoke）已完成，不能凭此勾项。
        步骤 4 已过。dump / 对比命令见 [`doc/guide/hf_llama_weight_map.md`](../guide/hf_llama_weight_map.md)。排查见 [`doc/debug-e2e.md`](../debug-e2e.md)。
        ~~~
        同一份切片 model.safetensors（21 个 tensor，F32）
        |
        +-- HuggingFace 加载 -> 前向 [1,2,3,4] -> 存成 npy（ref）
        |
        +-- Engine Loader 加载 -> 前向 [1,2,3,4] -> 得到一张 logits
                                              |
                                              v
                                    只比这两张 logits 表
        ~~~
- [ ] gguf（若目标模型需要）
- [x] fixture 加载单测（`tests/test_weight_loader.py`）

---

## 2. 模块 — TransformerModel

**职责**：**immutable** GPU 权重与 model 级只读资源；被 Engine **引用**，不含 session / KV 状态。immutable 具体含义见 §2.3。

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

### 2.1.1 tied 与 untied：embed 和 lm_head 要不要共用一块权重？

生成一条 token 时，模型一头一尾各碰一次 [词表]：

- **入口 embed**：你给一个 token id（比如 42），模型去查表，得到长度为 hidden 的向量。可以理解为 [这个词读进来时长什么样]。
- **出口 lm_head**：最后一层算完 hidden 之后，要对照整张词表打分，得到每个词可能是下一个 token 的 logits。可以理解为 [当前状态最像哪个词]。

于是自然有两个权重矩阵：

| 名字 | 形状（本项目） | 干什么 |
|------|----------------|--------|
| embed | `[vocab, hidden]` | id -> 向量，查表 |
| lm_head | `[hidden, vocab]`（untied 时） | 向量 -> 词表分数，投影 |

**untied（`tie_word_embeddings=0`）** —— 各用各的表，两本词典

checkpoint 里 **embed 和 lm_head 是分开存的两块权重**。认字用 embed，猜下一个词用 lm_head，互不共用。

在本项目里：

- GPU 上分配 `d_embed` 和 `d_lm_head` 两块显存；
- load 时要分别 H2D `embed` 和 `lm_head`；
- forward 时 lm_head 走 `untied_lm_head_forward_device`，读 `d_lm_head`。

参数量多一块 `vocab * hidden`，但两个矩阵可以各学各的，约束更少。

**tied（`tie_word_embeddings=1`）** —— 认字和猜词共用同一本词典

很多 LLM（含 LLaMA 系）会把 **输出层和输入 embed 绑成同一张表**：`d_lm_head` 不再单独占一块，而是 **直接指向 `d_embed`**。

直觉：同一个词，[读进来] 和 [被预测出来] 最好用同一套向量表示；input 和 output 关于词表是对称的。

在本项目里：

- create 时 `d_lm_head = d_embed`，只 alloc 一块 embed 大小的显存；
- load 时 **只 load `embed`**，fixture 里可以没有 `lm_head`；
- forward 时 lm_head 走 `tied_lm_head_forward_device`，用的仍是 `d_embed`，公式是 `logits[v,t] = sum_h embed[v,h] * hidden[h,t]`。

省参数量，也是 LLaMA 等模型的常见默认。

**和代码的对应关系**

配置在 `ModelConfig.tie_word_embeddings` 里，**create model 时** 就定好了（与 checkpoint / fixture 的 config.txt 一致）：

~~~
tie_word_embeddings=0   -> untied，embed + lm_head 各一块
tie_word_embeddings=1   -> tied，d_lm_head == d_embed
~~~

`transformer_model_lm_head_forward_device` 内部只看 `model->lm_head_tied`：为 1 调 tied 路径，为 0 调 untied 路径。Engine 不用每次再传这个开关。

**和 immutable 的关系**

无论 tied 还是 untied，权重都在 load 时一次性 H2D，之后 forward 只读。tied 只是 [占几块显存、load 哪些 tensor 名字] 不同，不改变 [Model 权重冻结、session 在 Engine] 的边界。详见 §2.3。

### 2.2 生命周期

~~~
操作                                              行为
────────────────────────────────────────────────────────────────────────────
model_create + transformer_model_load_weights   H2D 全部权重（仅一次，immutable）；
                                                  rope cache 在 create 时构建
model_destroy                                     cudaFree 权重；须在引用它的
                                                  Engine 全部 destroy 之后
reset                                             无（immutable）
~~~

### 2.3 immutable 含义与模块边界

此处 **immutable** 是 **Model 在推理链中的角色**：权重在 load 之后固定，session 演化发生在 Engine，不在 Model。

**（1）权重内容不变**

- `transformer_model_load_weights` **仅可成功调用一次**；`weights_loaded` 置位后再次 load 返回失败。
- 无 `model_reset`、无换 checkpoint、forward 路径不修改 `d_w_*` / `d_embed` 等。

**（2）不含 session 状态**

~~~
                          TransformerModel          InferenceEngine（§3）
─────────────────────────────────────────────────────────────────────────
KV cache                  无                        有
cache_len / d_pos         无                        有
reset                     无                        engine_reset（新对话）
权重                      load 后固定               只读引用 Model
~~~

Model **不随 prefill/decode 演化**；变的是 Engine 里的 KV 与 session 计数。

**（3）生命周期：Engine 只读借用**

~~~
WeightLoader  →  host tensors
       ↓  transformer_model_load_weights（一次 H2D）
TransformerModel  →  GPU 权重冻结（immutable）
       ↓  只读指针引用（Engine 不 copy 权重、不 own Model）
InferenceEngine   →  prefill/decode（改 KV，不改权重）
       ↓
engine_destroy  →  释放 KV / pool；**不** destroy Model
       ↓
model_destroy   →  释放 GPU 权重（须在 Engine 全部 destroy 之后）
~~~

**tied embed / lm_head**：概念与 load/forward 行为见 §2.1.1；此处仅强调 `tie_word_embeddings=1` 时 `d_lm_head == d_embed`，仍是 load 时定型的同一块显存，语义上同样 immutable。

对齐 vLLM / SGLang：**权重常驻 GPU（Model）**，**session / KV 单独管理（Engine）**。

### 2.4 数据流（被 Engine 调用）

- **embed**：`d_token_ids -> d_hidden`
- **lm_head**：末 hidden -> `d_logits`（decode 要 logits 时）
- **layer 权重**：Engine 按 `layer_idx` 取 `LayerWeights[i]` 传入 Phase 1 layer 链
- **final norm**：`d_w_final_norm` 作用于末层 hidden

**embed / lm_head 交付摘要**：Model 从 [权重容器 + H2D] 扩展为能独立跑 embed 与 lm_head 的 device 算子，并验证了 `token -> hidden -> logits` 数据流的两端；Engine 级完整 forward 待 §3。

### 2.4.1 算子 vs Model 薄封装：代码里其实是两层，别搞混

读代码时容易以为 embed / lm_head [住在 TransformerModel 里面]，像 struct 的成员函数一样。其实不是。它们和 `linear.cu`、`rms_norm.cu` 一样，先是 **`src/cuda/` 下的通用算子**；`transformer_model_*_forward` 只是在上面套了一层 [很薄的壳]，帮你把 Model 里的权重指针、tied 配置、load 状态接好。真正做乘加、查表的是 cuda 那一层。

可以想成三道工序，各干各的事：

~~~
谁                    在哪                          干什么（人话）
──────────────────────────────────────────────────────────────────────────────
cuda 算子             embed.cu / lm_head.cu         只认指针和 shape：给你 d_embed、
                                                    token_ids，我帮你 gather 出 hidden；
                                                    给你 d_hidden、权重，我帮你 GEMM
                                                    出 logits。不关心这些权重是谁
                                                    的、load 过没有。

Model 薄封装         transformer_model.h           认识 TransformerModel 这个对象：
                                                    从 model 里取出 d_embed / d_lm_head；
                                                    没 load 权重就报错；tied 时走
                                                    tied_lm_head_*，untied 时走
                                                    untied_lm_head_*。Engine 调这一层
                                                    最省事，不用自己拼。

Engine（规划）        inference_engine.*            管 session：这一步 prefill 还是
                                                    decode、跑几层 block、什么时候
                                                    final_norm、要不要 lm_head。权重
                                                    一律只读引用 Model，不自己 alloc
                                                    第二份 embed。
~~~

**权重在谁手里？**

`d_embed`、`d_lm_head`、各层 `LayerWeights[i]`、`d_w_final_norm` 全是 **Model 在 create + load 时** 搞定的，load 完就冻结。Engine 用时只拿指针，**不会** 再 copy 一份权重，也 **不会** 在 destroy Engine 时 free 掉 Model。

**那跟 layer 里的 Linear 有什么不一样？**

Phase 1 的 layer 链有个历史习惯：调用 `transformer_layer_linears_forward_device` 时，**9 个 `d_w_*` 都要调用方显式传进来**。所以 Model 用 `get_layer_weights(i)` 把指针暴露给 Engine，Engine 再传给 layer 链——算子 (`linear_forward_device`) 仍然通用，只是 weight 从 Engine 手里过一手。

embed / lm_head 没有这套 [传 9 个指针] 的历史接口。权重本来就是 Model 级的（见 §2.1），tied 时还要在 forward 里判断 [到底用 d_embed 还是 d_lm_head]。所以在 Model 上直接提供 `transformer_model_embed_forward_device` / `transformer_model_lm_head_forward_device`，把 [取指针 + 检查 + tied 分支] 收进去，Engine 一行调用就行。

对比如下（帮助你对照代码）：

~~~
                    layer 里的 Linear / RMSNorm              embed / lm_head
─────────────────────────────────────────────────────────────────────────────────
权重在哪            LayerWeights[i].d_w_*                    model->d_embed /
                                                             model->d_lm_head
底层算子            linear_forward_device(…, d_w, …)          embed_forward_device(…)
                    调用方必须传 weight                         lm_head_*_forward_device(…)
                                                             调用方传 d_embed / d_lm_head
Model 怎么接        get_layer_weights(i) 暴露指针            transformer_model_*_forward
                    Engine 再传给 Phase 1 layer 链            内部取 model 权重 + tied 判断
~~~

**Engine 能不能跳过 Model 壳，直接调 cuda 算子？**

可以，等价写法大致是：

~~~
embed_forward_device(stream, transformer_model_get_d_embed(model), d_token_ids, …);

// untied：
untied_lm_head_forward_device(stream, handle,
    transformer_model_get_d_lm_head(model), d_hidden, …);

// tied（tie_word_embeddings=1）：
tied_lm_head_forward_device(stream, handle,
    transformer_model_get_d_embed(model), d_hidden, …);
~~~

但要自己保证：权重已 load、tied 时别误用 `get_d_lm_head`（它和 `get_d_embed` 是同一地址，但 GEMM 接口不同）。默认推荐走 Model 薄封装，少踩坑。

**为什么拆成两层，而不是全塞进 Model？**

cuda 算子保持 dumb、好单测：给指针就能跑，不依赖 `TransformerModel` 这个类型。Model 壳表达 [这份 checkpoint 的权重 + tied 配置] 这一层业务边界，Engine 编排 session 时少写重复逻辑。final_norm 已同样处理：`rms_norm_forward_device` + `transformer_model_final_norm_forward_device`。

### 2.5 API 与测试（规划）

- [x] `ModelConfig` POD + 校验（`model_config.h`）
- [x] `transformer_model_create` / `destroy`（骨架：GPU 权重容器 + RopeCosSinCache）
- [x] `TransformerLayerWeights[N]` / `d_embed` / `d_lm_head` / `d_w_final_norm` 分配；tied 可选
- [x] Loader host 权重 H2D 拷贝（`transformer_model_load_weights`；内部名 `layer{i}.w_*` / `embed` / `final_norm`）
  - fixture：`load_weights_from_fixture`
  - HF Llama safetensors：`load_weights_from_safetensors_hf_llama`（步骤 4）
- [x] embed / lm_head device 算子（或 gather GEMM）+ `forward_host` 单测
- [x] final norm device（`rms_norm_forward_device` 薄封装）+ `forward_host` 单测
- [x] 2-layer fixture 权重 layout 单测（`tests/test_transformer_model_two_layer_fixture.py`）

---

## 3. 模块 — InferenceEngine

**职责**：**session 边界**；prefill/decode 调度；多 layer forward 编排；维护 KV 与 `cache_len` / `d_pos`。

**规划路径**：`src/engine/inference_engine.{h,cpp}`

### 3.1 资源

~~~
InferenceEngine
  ├── TransformerModel*      只读引用
  ├── KVCache(num_layers=N)
  ├── BufferPool             FA staging；session：d_token_ids / d_logits / d_hidden_out / d_out_token
                             （create 按 max_seq 一次分配，prefill/decode 复用）
                             layer workspace 按 T 分桶；create 预热 T=1
  ├── cudaStream / cublasHandle
  └── SessionState           cache_len；next_pos

每步 ephemeral:
  InferenceForwardCtx        num_tokens, d_token_ids|d_hidden, d_pos, …
~~~

### 3.2 生命周期

~~~
操作                  行为
──────────────────────────────────────────────────────────────────
engine_create(model)  KVCache(N)、pool、stream；cache_len=0；预热 T=1 workspace
engine_reset          kv_cache_reset；cache_len=0；不动 Model / pool
engine_destroy        释放 KV、pool、stream；不 destroy model
forward 每步          栈上 InferenceForwardCtx
~~~

**骨架（已实现）**：`src/engine/inference_engine.{h,cpp}` — `inference_engine_create` /
`destroy` / `reset`；`KVCache(num_layers)` + FA staging BufferPool + `SessionState.next_pos`；
Python `inference_engine_me`；`tests/test_inference_engine.py`。

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

**forward（已实现）**：`inference_engine_forward_device` / `forward_hidden_host`；数据流见 §3.3；
`tests/test_inference_engine_forward.py`（N=1/N=2 prefill、prefill+decode+reset）。

### 3.4 从 Phase 1 泛化（Engine 内细节）

- [x] `kv_cache_create(..., num_layers=N)`；append/cast 按 `layer_idx`
- [x] per-layer 权重指针；`layer_idx` 传入 layer 链
- [x] 超出 `max_seq` → 报错

### 3.5 API 与测试（规划）

- [x] C：`inference_engine_create` / `destroy` / `reset` / `forward_device` / `forward_hidden_host`
- [x] C：`inference_engine_forward_token_last_logits`（host token -> pool H2D + device forward；末列留在 GPU）
- [x] C：`inference_engine_forward_token_host`（H2D/D2H 测试包装，内部调 forward_token_device）
- [x] Python：`inference_engine_me` — create/destroy/reset/kv_cache_len/forward_hidden_host/forward_token_host
- [x] `../guide/inference_engine_device_api.md`
- [x] 2-layer prefill e2e；N=1 退化 Phase 1；prefill+decode+reset

---

## 4. 模块 — Sampler / GenerateLoop

**职责**：Engine 产出 logits 之后，**token 出环**；不持有 KV / 权重。

**路径**：`src/engine/generate_loop.{h,cpp}` + Python `generate_loop_me`

### 4.1 边界

GenerateLoop **借用** `InferenceEngine*`；不 create/destroy Engine、不 own KV/权重。

### 4.2 骨架（已实现）

- C：`sampler_top_k_device` / `sampler_top_k_host`（`src/cuda/sampler_top_k.*`）
- C：`generate_loop_run` — prefill + decode；GPU 末列 logits 上 top-k 采样
- Python：`generate_loop_me.generate(engine, prompt_ids, max_new_tokens, eos_token_id=-1)`
- 测试：`tests/test_generate_loop.py`

### 4.3 细节

**Sampler / 文档**

- [x] 末 token logits slice（GenerateLoop 内 `forward_token_step`）
- [x] greedy（`top_k==1` via `sampler_top_k_device`）
- [x] temperature（`sampler_top_k_device` 的 `temperature` 参数 + `generate` 透传）
- [x] top-k（`sampler_top_k_host` + `generate` 的 `top_k`/`seed`）
- [x] top-p（`sampler_top_p_host` + `generate` 的 `top_p`/`temperature`/`top_k`）
- [x] 短序列 generate e2e + EOS（`tests/test_generate_loop.py`）
- [x] `doc/guide/generate_loop_device_api.md`

**说明**：temperature / top-k / top-p 的函数签名与组合规则见 API doc [Sampler API] 节。

### 4.4 生产化（骨架后，roadmap 模块 4 / §2.6）

骨架期 `forward_token_step` 每步 cudaMalloc。本轮已做完 1+2：Engine create 按 `max_seq_len` 开辟 session buffer，GenerateLoop 复用，decode 不再每步 malloc。

~~~
create_engine 一次（按 cfg.max_seq_len / vocab / hidden）
  pool.d_token_ids    int[max_seq]
  pool.d_logits       float[vocab, max_seq]  col-major
  pool.d_hidden_out   float[hidden, max_seq]
  pool.d_out_token    int[1]

每步 last_logits(host token_ids, T, pos)
  H2D 前 T 个 id -> pool.d_token_ids
  forward_token_device(pool tokens, pool logits, T)   内部用 pool.d_hidden_out
  末列 = pool.d_logits + vocab*(T-1)

例：prompt T=3 再 decode 三次 T=1。create 预热 T=1 layer workspace + d_pos[1]；
    decode 三次只见 H2D 1 个 int + forward + 采样。prefill T=3 仍可能第一次懒分配。
~~~

尺寸核对（TinyLlama 2 层切片，max_seq=256）：logits 约 32MB，hidden 约 2MB。Orin 全局显存约 3.8GB，create 时一次分完可接受。

仍未做：

3. GenerateLoop：仅循环 + stop，无 CUDA glue（采样 / D2H token 仍在 loop）
4. Sampler：TODO(perf-topk) / TODO(perf-topp)

代码：`inference_engine_forward_token_last_logits`；pool 在 `inference_engine_create`。

---

## 5. 并行分支（不改四模块对象模型）

- **MoE FFN**：替换 Model 内 layer FFN 路径
- **性能**：BufferPool 完善、FP16/INT4、CUDA Graph、profiling
- **2.7**：batching、Paged KV、Radix cache

---

## 6. 实现顺序（模块骨架 → 细节）

1. **WeightLoader** — fixture（可并行起步）
2. **TransformerModel** — 容器 + embed/lm_head + fixture H2D 拷贝
3. **InferenceEngine** — session + N-layer forward + prefill/decode
4. **Sampler / GenerateLoop** — 闭合 token 环
5. **WeightLoader** — safetensors（1 只读格式 -> 2 fixture roundtrip -> 3 HF 名映射 -> 4 真模型验证）
6. **并行**：MoE、Graph、量化

**API 契约**（收工前）：[`../guide/inference_engine_device_api.md`](../guide/inference_engine_device_api.md)（Engine 为主；Loader/Model load API 可同文档分节）。
