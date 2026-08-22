# Weight Fixture 目录结构

C 头文件：`src/model/weight_loader.h`。Loader API：`weight_loader_load_fixture(path, out)`。

**生命周期总图**仍以 [`../design/phase2_lifecycle.md`](../design/phase2_lifecycle.md) §1 为准。本文只回答一件事：你传给 `path` 的 **fixture 目录里到底有什么**，以及 Loader / Model 按什么约定读它。

仓库里**没有**提交现成的 fixture 目录；测试和本地实验都是运行时写临时目录，或自己按下面格式拼。

---

## path 是什么

`weight_loader_load_fixture(path, ...)` 的 `path` 必须是一个**已存在的目录**，不是单个文件。

Loader 只认这个目录根下的固定文件名：

~~~
path/
  config.txt       # 必填
  manifest.txt     # 必填，至少一行 tensor
  *.f32            # manifest 里引用的权重文件
~~~

读完后得到 `WeightLoadResult`：`out->config`（`ModelConfig`）+ `out->tensors[]`（host 侧 float32 权重表）。之后 `transformer_model_load_weights_from_fixture` 再做 H2D。

Python 侧等价调用：

~~~python
loaded = weight_loader_me.load_fixture(fixture_dir)
transformer_model_me.load_weights_from_fixture(model, fixture_dir)
~~~

---

## 目录长什么样（1-layer 最小完整例）

假设 `hidden_size=128`，`vocab_size=512`，`num_layers=1`，`tie_word_embeddings=0`。目录可以长这样：

~~~
/tmp/jj_weight_fixture_abc/
  config.txt
  manifest.txt
  embed.f32
  final_norm.f32
  lm_head.f32
  layer0_w_q.f32
  layer0_w_k.f32
  layer0_w_v.f32
  layer0_w_o.f32
  layer0_w_gate.f32
  layer0_w_up.f32
  layer0_w_down.f32
  layer0_w_input_layernorm.f32
  layer0_w_post_attention_layernorm.f32
~~~

共 **12 张**权重 tensor（每层 9 张 + 全局 3 张）。2 层时再多一套 `layer1_*`，`config.txt` 里 `num_layers=2`。

---

## config.txt

位置：fixture 目录根下的 `config.txt`。

格式：每行一个 `key=value`；`#` 开头为注释。Loader 要求 **11 个 key 全部出现**，并通过 `model_config_validate`。

~~~
hidden_size=128
intermediate_size=256
num_layers=1
num_q_heads=4
num_kv_heads=2
head_dim=32
vocab_size=512
max_seq_len=256
freq_base=10000.0
rms_norm_epsilon=1e-06
tie_word_embeddings=0
~~~

字段含义见 `src/model/model_config.h`。其中 `num_layers` 必须和 manifest 里实际出现的 `layer{i}.*` 层数一致；`head_dim * num_q_heads` 应等于 `hidden_size`（KV 头同理由 validate 检查）。

**tied embed 时**：`tie_word_embeddings=1` 表示 input embed 与 lm_head 共用同一张表；fixture 里可以**不写** `lm_head` 这一行（Model load 只拷 `embed`）。详见 phase2_lifecycle §2.1.1。

---

## manifest.txt

位置：fixture 目录根下的 `manifest.txt`。

每行描述一张 tensor，格式固定为：

~~~
名字  ndim  dim0  dim1  ...  相对路径
~~~

- **名字**：Model 内部名，例如 `layer0.w_q`、`embed`（**不是** HuggingFace 的 `model.layers.0.self_attn.q_proj.weight`）
- **ndim**：维度个数
- **dim0, dim1, ...**：各维大小，与 `.f32` 文件字节数一致
- **相对路径**：相对 fixture 目录根；通常与 tensor 名对应，`.` 换成 `_`，后缀 `.f32`

示例（与上面 1-layer 目录对应，节选）：

~~~
embed 2 512 128 embed.f32
layer0.w_q 2 128 128 layer0_w_q.f32
layer0.w_k 2 128 64 layer0_w_k.f32
layer0.w_v 2 128 64 layer0_w_v.f32
layer0.w_o 2 128 128 layer0_w_o.f32
final_norm 1 128 final_norm.f32
lm_head 2 128 512 lm_head.f32
~~~

空行和以 `#` 开头的行会被跳过。Loader 解析实现：`weight_loader.cpp` 里 `parse_manifest_and_load_tensors`。

**文件名规则**（与 `tests/fixture_utils.py` 一致）：tensor 名里的 `.` 写成文件名的 `_`，例如 `layer0.w_q` -> `layer0_w_q.f32`。

---

## *.f32 权重文件

- 纯 **float32** 裸二进制，**无** header、无 shape 元数据
- **row-major**（C / numpy 默认 contiguous）顺序
- 字节数 = manifest 该行各维乘积 × 4

Loader 会校验：文件大小必须恰好等于 `numel * sizeof(float)`，否则 load 失败。

---

## tensor 名字与 shape（与 Model 对齐）

名字必须和 `TransformerModel` / `transformer_model_load_weights` 查找表一致。每层（`layer{i}`，`i` 从 0 起）固定 9 个后缀；全局 3 个。

以 `hidden_size=H`，`intermediate_size=I`，`num_q_heads=Qh`，`num_kv_heads=KVh`，`head_dim=D`，`vocab_size=V` 为例（且 `Q_DIM = Qh*D`，`KV_DIM = KVh*D`）：

~~~
名字                              shape
------------------------------------------------------------------
embed                             [V, H]
final_norm                        [H]
lm_head                           [H, V]          # tie=1 时可省略
layer{i}.w_q                      [H, Q_DIM]
layer{i}.w_k                      [H, KV_DIM]
layer{i}.w_v                      [H, KV_DIM]
layer{i}.w_o                      [Q_DIM, H]
layer{i}.w_gate                   [H, I]
layer{i}.w_up                     [H, I]
layer{i}.w_down                   [I, H]
layer{i}.w_input_layernorm        [H]
layer{i}.w_post_attention_layernorm [H]
~~~

测试里生成 1-layer 权重的参考代码：`tests/test_generate_loop.py` 的 `_fixture_tensors(1)`；2-layer layout 单测：`tests/test_transformer_model_two_layer_fixture.py`。

---

## fixture 与 safetensors roundtrip（步骤 2）

fixture 目录和 `.safetensors` 是两种存法，内部 tensor 名与 layout 相同（都是 Model 内部名 + f32 row-major）。

步骤 2 验证整条链：

~~~
write_weight_fixture(dir)     # fixture 落盘
export_fixture_dir_to_safetensors(dir, weights.safetensors)
load_fixture(dir)             # 基准
load_safetensors(weights.safetensors)   # 须与基准逐 tensor 一致
~~~

实现：`tests/fixture_utils.py` 的 `export_fixture_dir_to_safetensors`；单测 `tests/test_weight_loader.py::test_fixture_safetensors_roundtrip`（1-layer 完整 12 张 tensor）。

---

## 和 safetensors 路径的区别（交叉引用）

| 入口 | path 类型 | 权重在哪 | config 在哪 |
|------|-----------|----------|-------------|
| `load_fixture` | **目录** | manifest + 多个 `.f32` | 目录内 `config.txt`（必填） |
| `load_safetensors` | **单个 `.safetensors` 文件** | 文件内 JSON header + blob | 同目录可选 `config.txt` |

safetensors 步骤 1 只读 F32 blob，tensor 名仍是文件内 JSON key；HF 名映射是后续 roadmap 步骤。fixture 目录格式与 safetensors **无关**，是开发/单测用的明文 layout。

---

## 怎么快速生成一个 fixture

不要手搓二进制，用测试 helper：

~~~python
from fixture_utils import write_weight_fixture
import tempfile

fixture_dir = tempfile.mkdtemp(prefix="jj_weight_fixture_")
write_weight_fixture(fixture_dir, config_dict, tensors_dict)
# config_dict: 上面 11 个 key
# tensors_dict: {"embed": np.ndarray, "layer0.w_q": np.ndarray, ...}
~~~

`write_weight_fixture` 会按约定写出 `config.txt`、`manifest.txt` 和所有 `.f32`。

单测入口：`tests/test_weight_loader.py`（`test_load_fixture_roundtrip`）；Engine / GenerateLoop e2e 也在各测试里 `mkdtemp` + `write_weight_fixture` 后 `load_weights_from_fixture`。

---

## 常见踩坑

1. **path 传成文件路径** — `load_fixture` 只接受目录；传 `.safetensors` 请用 `load_safetensors`。
2. **config 缺字段** — 11 个 key 少任何一个，`parse_config_file` 直接 -1。
3. **manifest 维度和 .f32 字节数对不上** — 改 shape 后忘记重导 `.f32`。
4. **tensor 名写错** — 必须 `layer0.w_q` 这种内部名；HF checkpoint 名不会自动映射（真模型路径见 roadmap safetensors 步骤 3/4）。
5. **num_layers 与 manifest 不一致** — config 写 2 层但 manifest 只有 `layer0.*`，Model create 与 load 会对不齐。
