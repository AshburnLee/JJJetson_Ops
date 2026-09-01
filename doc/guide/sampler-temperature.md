# Sampler：temperature 参数说明

**归属**：Module 4 Sampler / GenerateLoop，**不是** InferenceEngine 行为。

Lifecycle 与 API 总览见 [`../design/phase2_lifecycle.md`](../design/phase2_lifecycle.md) 第 4 节、[`generate_loop_device_api.md`](generate_loop_device_api.md)。本文说明 temperature **控制什么**、**为何需要**、以及在 JJJetson_Ops 里的落点（规划）。

---

## 在数据流中的位置

Engine 每步 forward 产出末 token 的 logits `[vocab_size]`；Sampler 把 logits 变成下一个 token id。Temperature 只作用在 **logits -> 概率 -> 采样** 这一步。

~~~
prompt token_ids
    -> Engine (prefill / decode, KV, forward)
    -> 末 token logits [vocab_size]
    -> Sampler (temperature, top_k, top_p, seed)
    -> next token id
    -> GenerateLoop 写回，再喂 Engine decode
~~~

`ie_forward_token_*` **没有** temperature 参数；调 temperature 的是 `generate_loop_run` / `sampler_top_k_device`。

---

## 它具体控制什么

对 logits 做 `logits / T`，再 softmax，然后按概率随机抽取 token（常与 top-k / top-p 组合）。

~~~
T              效果
─────────────  ────────────────────────────────────────────────────────────────
T = 1          使用模型原始分布（与未加 temperature 时相同）
T < 1 (如 0.7) 分布更 [尖]：高 logit 的 token 更容易被选中；输出更确定、重复感更强
T > 1 (如 1.2) 分布更 [平]：低分 token 也有机会；输出更多样、更发散
T -> 0         近似 greedy（argmax）；数值上 softmax 极尖；工程上常 clamp 最小 T 或走 greedy
~~~

**与 top_k 的关系**（正数 T 下，全体 logits 同除 T **不改变排序**）：

~~~
采样路径                                    temperature 是否改变输出
──────────────────────────────────────────  ──────────────────────────────────────────────
top_k == 1（greedy argmax）                 否 — argmax(z/T) == argmax(z)
top_k > 1（先 top-k，再在 k 个 token 上采样） 是 — 只改变 k 个候选内相对概率，不改变 top-k 集合
~~~

---

## 为什么需要它

1. **Greedy 太死板**：每步固定 argmax，易重复、循环；对话与创意生成通常需要可控随机性。
2. **与训练范式一致**：LLM 在整词表分布上训练；推理若永远 argmax，与训练时的采样行为不一致，质量可能下降。
3. **产品旋钮**：同一模型、同一 prompt，低 T 适合事实/代码（要稳），高 T 适合头脑风暴/故事（要活）。OpenAI `temperature`、vLLM `SamplingParams.temperature` 等同理。
4. **与 top-k / top-p 组合**：工业常见 `temperature` 调整体 [锐度]，top-k / top-p 再裁掉长尾，避免采到离谱 token。top-p 见 [`sampler-top-p.md`](sampler-top-p.md)。

---

## JJJetson_Ops 实现（已完成）

- **不**单独新增 Engine API；**`sampler_top_k_device(..., temperature)`** 已扩展（`temperature <= 0` 报错，默认 `1.f`）。
- 在 `src/cuda/sampler_top_k.cu` 的 top-k softmax 段对 logits 除以 T；greedy 路径忽略 T。
- GenerateLoop / `generate_loop_me.generate` 透传 `temperature`；测试见 `tests/test_generate_loop.py`。
- API 签名见 [`generate_loop_device_api.md`](generate_loop_device_api.md) [Sampler API] 节。

---

## 相关文件

~~~
doc/guide/sampler-temperature.md          # 本文
doc/guide/generate_loop_device_api.md     # GenerateLoop + Sampler API
doc/design/phase2_lifecycle.md            # 模块 4 lifecycle
src/cuda/sampler_top_k.{h,cu}             # 规划：temperature 参数落点
src/engine/generate_loop.{h,cpp}          # 透传至 Sampler
~~~
