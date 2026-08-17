# Sampler：top-p（nucleus）参数说明

**归属**：Module 4 Sampler / GenerateLoop，**不是** InferenceEngine 行为。

Lifecycle 与 API 总览见 [`../design/phase2_lifecycle.md`](../design/phase2_lifecycle.md) 第 4 节、[`generate_loop_device_api.md`](generate_loop_device_api.md)。temperature 语义见 [`sampler-temperature.md`](sampler-temperature.md)。本文说明 top-p **在干什么**、**为何需要**，以及与 top_k / temperature 的组合关系。

---

## 在数据流中的位置

Engine 每步 forward 产出末 token 的 logits `[vocab_size]`；Sampler 把 logits 变成下一个 token id。Top-p 只作用在 **logits -> 概率 -> 裁 nucleus -> 采样** 这一步。

~~~
prompt token_ids
    -> Engine (prefill / decode, KV, forward)
    -> 末 token logits [vocab_size]
    -> Sampler (temperature, top_k, top_p, seed)
    -> next token id
    -> GenerateLoop 写回，再喂 Engine decode
~~~

`inference_engine_forward_token_*` **没有** top_p 参数；调 top_p 的是 `generate_loop_run` / `sampler_top_p_device`（当 `top_p < 1` 时）。

---

## top-p（nucleus）在干什么

Top-p 又叫 **nucleus sampling**：在概率分布上，保留**累计概率质量达到 `top_p` 的最小 token 集合**（nucleus，至少 1 个 token），只在这个集合里重新归一化并随机采样。

实现步骤（与 `src/cuda/sampler_top_p.h` 注释一致）：

1. **temperature** — 对 logits 除以 T，再 softmax 得到概率（见 [`sampler-temperature.md`](sampler-temperature.md)）
2. **按概率从高到低排序**
3. **从最高 prob 往下累加**，直到累计质量 **>= top_p**（至少保留 1 个 token）— 这就是 nucleus
4. **在 nucleus 内重新归一化**，按概率 + seed 抽一个 token id

~~~
top_p          效果（直观）
─────────────  ────────────────────────────────────────────────────────────────
top_p = 1      不裁 nucleus（JJJetson_Ops 默认走 top_k 路径，见下文）
top_p = 0.9    约 90% 概率质量落在候选集合内；长尾低 prob token 被丢掉
top_p 较小     候选更少、输出更稳；过大则接近全词表采样
~~~

**目的**：比 greedy 有随机性，又比 [整词表都参与采样] 稳 —— **裁掉长尾低概率 token**，避免采到离谱词，同时保留比固定 top_k 更自适应的候选规模（prob 尖时候选少，平时候选多）。

---

## 和 top_k、temperature 的关系

三者**可以一起用**，工业里常见组合是 `temperature` + `top_k` + `top_p`（OpenAI / vLLM 等同理）。在 JJJetson_Ops 里是**串联**，不是三选一：

~~~
参数          作用                          与另两者的关系
────────────  ────────────────────────────  ──────────────────────────────────────
temperature   调 softmax 锐度（logits / T）   与 top_k 排序无关；top_p 路径 softmax 前除 T
top_k         只保留 logit 最高的 k 个       可选预截断；再在其上做 top_p nucleus
top_p         按累计 prob 裁 nucleus          在（可能已 top_k 截断的）分布上裁；至少 1 token
~~~

**GenerateLoop 分流**（`src/engine/generate_loop.cpp` 中 `sample_token_device`）：

~~~
条件                         走哪条路径
───────────────────────────  ──────────────────────────────────────────────
top_p == 1（默认）           sampler_top_k_device（top_k + temperature）
top_p < 1                    sampler_top_p_device（temperature + top_p + 可选 top_k）
~~~

**组合示例**：

- `temperature=0.8, top_k=50, top_p=0.95` — 先取 top-50，再在这 50 个里按累计 0.95 裁 nucleus，再采样
- `temperature=1.0, top_k=0, top_p=0.9` — 全词表 softmax 后做 nucleus（`top_k==0` 表示不预截断）
- `top_k=1` — 只有 1 个候选，top_p / temperature **无法改变输出**（等同 greedy）

与 temperature 专文对照：greedy（`top_k==1`）时 T 无效；top-p 在仅 1 个候选时同样无效。要有 top-p 效果，需 **`top_p < 1` 且候选数 > 1**。

---

## 为什么需要它

1. **固定 top_k 不够灵活**：top_k=50 永远 50 个候选；top_p 随分布形状自动伸缩 nucleus 大小。
2. **裁长尾**：比全词表采样更安全，比 greedy 更多样。
3. **与 temperature 分工**：T 调整体锐度；top_p 按概率质量裁集合；top_k 可选硬上限（词表很大时常用）。

---

## JJJetson_Ops 实现（已完成）

- 独立算子：`src/cuda/sampler_top_p.{h,cu}`，`sampler_top_p_device` / `sampler_top_p_host`
- `top_p` 必须在 `(0, 1]`；`top_p == 1` 时 GenerateLoop 不调用本算子，改走 `sampler_top_k_device`
- 可选 `top_k` 预截断：`0 < top_k < vocab_size`（上限 128，与 top_k 算子一致）；`top_k==0` 全词表 nucleus
- Python：`generate_loop_me.sampler_top_p_host(...)`；`generate(..., top_p=1.0)` 透传
- 测试：`tests/test_generate_loop.py`（nucleus 合法性、可复现、e2e）
- **TODO(perf-topp)** — 性能债（与 top-k 同等级别，grep 同 tag）：
  - 现状：`<<<1,1>>>` 单 thread；O(n^2) 选择排序；`sampler_top_p_device` 每 call `cudaMallocAsync` probs/indices
  - 瓶颈：大词表 decode 时 SM 空闲 + 排序/分配开销
  - 目标：CUB/thrust 并行 sort + nucleus mask + sample；或与 top-k 合并 fused kernel（vLLM/SGLang）
  - 代码：`src/cuda/sampler_top_p.cu`（kernel / launch / temp alloc）；`src/engine/generate_loop.cpp`（调用点）

API 签名见 [`generate_loop_device_api.md`](generate_loop_device_api.md) [Sampler API] 节。

---

## 相关文件

~~~
doc/guide/sampler-top-p.md              # 本文
doc/guide/sampler-temperature.md      # temperature 语义
doc/guide/generate_loop_device_api.md   # GenerateLoop + Sampler API
doc/design/phase2_lifecycle.md          # 模块 4 lifecycle
src/cuda/sampler_top_p.{h,cu}           # top-p 算子
src/cuda/sampler_top_k.{h,cu}           # top_k 路径（top_p==1）
src/engine/generate_loop.{h,cpp}        # sample_token_device 分流
~~~
