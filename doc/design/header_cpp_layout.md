# Header / CPP 如何分工

实现 C/C++ 模块时：**`.h` 写对外契约，`.cpp` 写实现与可隐藏的内部结构**。与 **对象作用**、**生命周期**、**调用方是否需要触字段** 都有关，但不是「有生命周期就一定进 `.cpp`」这么简单。

**相关规范**：`jjjetson-ops-conventions.mdc` §4.6（头文件放哪）；本文属 **design/** 类文档，补充 **同一模块内 `.h` / `.cpp` 各写什么**。

---

## 1. 一句话原则

| 放 `.h` | 放 `.cpp` |
|---------|-----------|
| 调用方**必须知道**的类型 layout 或函数签名 | 只有 **create/destroy/forward 实现** 才需要知道的内部字段 |
| 稳定的 **C API**（`extern "C"`） | `struct` 完整定义（opaque 时） |
| 纯 **POD 视图**（如每层权重指针包） | 分配/释放、H2D、错误路径清理 |

---

## 2. 三种典型模式（本仓库）

### 2.1 Opaque 容器 — 完整 `struct` 只在 `.cpp`

**适用**：有 **create → 使用 → destroy** 生命周期；调用方只通过 **函数** 交互，不应直接 `obj->field`。

~~~
header:  typedef struct TransformerModel TransformerModel;
         TransformerModel *transformer_model_create(...);
         void transformer_model_destroy(TransformerModel *model);
         const float *transformer_model_get_d_embed(const TransformerModel *model);

cpp:     struct TransformerModel { ModelConfig config; float *d_embed; ... };
         // get_d_embed 实现里访问 model->d_embed
~~~

**本仓库例子**：`TransformerModel`、`TransformerRunner`、`KVCache`、`MoeRunner`。

**为何 header 里可以有 getter 声明却没有 struct 定义？**
Header 只有函数**声明**；**访问成员**的代码在 `.cpp` 的函数**定义**里。不完整类型 `TransformerModel*` 足够用于声明参数，不足以写 `model->d_embed`（那在 `.cpp` 里写）。

**与生命周期**：容器在堆上 `new`，`create` 把所有权交给调用方，`destroy` 回收 CPU 对象 + GPU 资源。内部布局变化不应迫使 Engine、binding 重编。

---

### 2.2 POD 视图 — 完整 `struct` 在 `.h`

**适用**：无独立 create/destroy；或 **设计意图就是** 让调用方 **直接读字段**（如把 9 个 `d_w_*` 传给 layer 链）。

~~~
header:  typedef struct TransformerLayerWeights {
             float *d_w_q;
             float *d_w_k;
             ...
         } TransformerLayerWeights;
         const TransformerLayerWeights *transformer_model_get_layer_weights(..., int layer_idx);

调用方:  const TransformerLayerWeights *lw = transformer_model_get_layer_weights(model, 0);
         transformer_layer_linears_forward_device(..., lw->d_w_q, lw->d_w_k, ...);
~~~

**本仓库例子**：`TransformerLayerWeights`、`TransformerRunnerForwardCtx`（ctx 字段 Engine 要填）。

**若把 LayerWeights 也藏进 `.cpp`**：Engine 无法写 `lw->d_w_q`，除非为每个指针各写一个 getter（API 碎、冗长）。

---

### 2.3 纯配置 POD — 全在 `.h`，无 `.cpp` 专属 struct

**适用**：只有 **数值/枚举**，无 GPU 指针、无 session；可拷贝、可写进 fixture `config.txt`。

~~~
model_config.h:  typedef struct ModelConfig { int hidden_size; ... } ModelConfig;
model_config.cpp: int model_config_validate(const ModelConfig *cfg);
~~~

**本仓库例子**：`ModelConfig`。

**与生命周期**：不拥有资源；随 Loader 输出、Model 内拷贝一份 `config`，没有单独的 destroy。

---

## 3. 决策表（写新模块时自问）

| 问题 | 是 → 倾向 |
|------|-----------|
| 调用方是否需要 `->field` 访问？ | 是 → struct 完整定义放 **`.h`** |
| 是否只有 create/destroy/getter 函数接触内部？ | 是 → struct 放 **`.cpp`**，`.h` 仅 forward declare |
| 是否纯 int/float 配置、无指针？ | 是 → POD 放 **`.h`** |
| 是否会加 C++ 成员（`std::map`、`vector`）？ | 是 → 几乎一定 **opaque + `.cpp`**（如 `TransformerRunner` 的 `buffers_by_tokens`） |
| 改内部字段是否应少牵连 include 方？ | 是 → **opaque** |

---

## 4. 与对象作用、生命周期的关系

~~~
Loader     host tensor，单次 load，无 GPU session     →  mostly 函数 API + HostTensor POD
Model      immutable GPU 权重容器，create/destroy      →  TransformerModel opaque；LayerWeights 公开视图
Engine     session + KV + 编排，create/reset/destroy   →  InferenceEngine opaque（规划）
Runner P1  session + 单层权重（Model 未拆前合一）       →  Runner opaque
~~~

- **生命周期越长、拥有的资源越多**（GPU buffer、KV、stream），越适合 **opaque 容器 + `.cpp` 隐藏布局**。
- **生命周期内的「视图」**（一层权重指针、一步 ForwardCtx）往往是 **短生命周期或只读借用**，用 **header 里的 POD** 传给下游算子更直接。
- **配置**（`ModelConfig`）穿越 Loader → Model → Engine，但 **不拥有** 任何地址，保持 **header POD** 即可。

---

## 5. 反例（避免）

- 在 `.h` 暴露 `TransformerModel` 完整 struct，Engine 写成 `model->d_embed` → 破坏封装，改字段全网重编。
- 把 `TransformerLayerWeights` 藏进 `.cpp` 却不提供字段 getter → Engine 无法接 layer 链。
- 在 `.h` 的 opaque 类型上实现需要完整类型的 inline 函数 → 要么移入 `.cpp`，要么把 struct 定义挪回 `.h`。

---

## 6. 本仓库速查

| 模块 | header | cpp 内 struct |
|------|--------|---------------|
| `transformer_model` | `TransformerModel` opaque；`TransformerLayerWeights` 全定义 | `struct TransformerModel` |
| `transformer_runner` | `TransformerRunner` opaque；`ForwardCtx`、buffers 名在 header | `struct TransformerRunner` |
| `kv_cache` | `KVCache` opaque | `struct KVCache` |
| `model_config` | `ModelConfig` 全定义 | 仅 validate 实现 |
| `weight_loader` | `HostTensor`、`WeightLoadResult` 全定义 | fixture 解析 helper |

---

## 7. 新增文件 checklist

1. 定 **对象角色**：容器 / 视图 / 配置 / 算子 API。
2. 定 **生命周期**：谁 create、谁 destroy、是否 immutable。
3. 定 **调用方是否触字段**：否 → opaque；是 → POD in header。
4. `.h` 只放 **稳定契约**；实现细节、parse helper、`static` 函数 → `.cpp`。
5. 与 **jjjetson-ops-conventions.mdc** §4.1–§4.6 对齐（C API / pybind / 头文件目录）。
