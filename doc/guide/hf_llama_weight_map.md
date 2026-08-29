# HF Llama 权重名映射（safetensors 步骤 3）

C 实现：`src/model/hf_llama_weight_map.cpp`。Loader 入口：`weight_loader_load_safetensors_hf_llama`（Python `weight_loader_me.load_safetensors_hf_llama`）。

**前提**：目标模型族为 **HF Llama 系**（Pre-LN、GQA、SwiGLU），与 Phase 1/2 layer 链一致。

---

## 做什么

将 HF 模型中的 weight 名映射为该推理引擎中的对应名称；读入 HF 风格 safetensors 后：

1. **改名**：JSON key -> 内部 `layer{i}.w_*` / `embed` / `final_norm` / `lm_head`
2. **2D 转置（仅 lm_head）**：HF `lm_head.weight` 是 `[vocab, hidden]`，内部 untied lm_head 是 `[hidden, vocab]`，要转置。q/k/v/o/gate/up/down **不要转**：HF 已经是 PyTorch `[out, in]`，跟 `linear_forward_device` 一样。以前把 Linear 也转成 `[in, out]`，Engine 实际在算 `W^T @ x`。
3. **1D RMSNorm**：shape 不变，只改名

`load_safetensors`（无 `_hf_llama`）仍保留 JSON key 原样，供步骤 1/2 单测。

---

## 映射表（Llama dense）

~~~
HF key                                              internal name
---------------------------------------------------------------------------
model.embed_tokens.weight                           embed
model.norm.weight                                   final_norm
lm_head.weight                                      lm_head              (+ transpose 2D)
model.layers.{i}.self_attn.q_proj.weight            layer{i}.w_q
model.layers.{i}.self_attn.k_proj.weight            layer{i}.w_k
model.layers.{i}.self_attn.v_proj.weight            layer{i}.w_v
model.layers.{i}.self_attn.o_proj.weight            layer{i}.w_o
model.layers.{i}.mlp.gate_proj.weight               layer{i}.w_gate
model.layers.{i}.mlp.up_proj.weight                 layer{i}.w_up
model.layers.{i}.mlp.down_proj.weight               layer{i}.w_down
model.layers.{i}.input_layernorm.weight             layer{i}.w_input_layernorm
model.layers.{i}.post_attention_layernorm.weight    layer{i}.w_post_attention_layernorm
~~~

未识别 key：stderr 警告并跳过（tied 时无 `lm_head.weight` 属正常）。

---

## 单测（无需下载真模型）

`tests/fixture_utils.py`：`internal_tensors_to_hf_llama_layout` 把内部 fixture 转成 HF key + layout。

`tests/test_weight_loader.py`：`test_hf_llama_safetensors_name_map` — 写 HF 风格 safetensors，`load_safetensors_hf_llama` 后与 `load_fixture` 逐 tensor 一致。

`tests/test_transformer_model_load_weights.py`：`test_load_weights_from_safetensors_hf_llama` — 同一套 HF 文件走 Model H2D。

`tests/test_inference_engine_forward_token.py`：`test_engine_forward_token_hf_llama_safetensors` — Engine 短 prefill，与内部 tensor ref 对比。

---

切片前先读本地 checkpoint：`doc/guide/understand_safetensors.md`。

## 步骤 4：真模型 / 切片

Orin 全局显存大约 4GB。TinyLlama-1.1B 若转成 F32 全量约 4GB+，**放不下**。开发期只切 **1~2 层**。

Loader 只读 **F32** safetensors。HF 仓库里常见 BF16/FP16，导出时必须先转成 F32。

### 目录（导出脚本写出）

~~~
slice_dir/
  model.safetensors     # HF key，F32；只含 layer0..N-1 + embed + norm + lm_head(若 untied)
  config.txt            # 本引擎 11 项 ModelConfig；num_layers = 切片层数
  config.json           # HF 字段名；num_hidden_layers = 切片层数（无 config.txt 时 Loader 读这个）
~~~

`config.txt` 优先于 `config.json`。

### 导出

先自己把完整 HF 目录下到本地（`config.json` + `model.safetensors`）。脚本只切片，不下载。在 `JJJetson_Ops` 目录：

~~~
python scripts/export_hf_llama_slice.py \
  --src models/hf_src/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
  --out-dir models/tinyllama_2layer \
  --num-layers 2 \
  --max-seq-len 256
~~~

`--src` 必须是本地目录。`models/` 已在 `.gitignore`，切片不要提交进 git。`--max-seq-len 256`：Orin 上短 prefill 够用，KV 更小。

一次真实运行（TinyLlama-1.1B-Chat，切 2 层）。日志里的 `dtype=BF16` 是**源文件**；写出的切片已转成 F32。21 个 tensor = layer0 的 9 块 + layer1 的 9 块 + embed + norm + lm_head：

~~~
  slice lm_head.weight shape=[32000, 2048] dtype=BF16
  slice model.embed_tokens.weight shape=[32000, 2048] dtype=BF16
  slice model.layers.0.input_layernorm.weight shape=[2048] dtype=BF16
  slice model.layers.0.mlp.down_proj.weight shape=[2048, 5632] dtype=BF16
  slice model.layers.0.mlp.gate_proj.weight shape=[5632, 2048] dtype=BF16
  slice model.layers.0.mlp.up_proj.weight shape=[5632, 2048] dtype=BF16
  slice model.layers.0.post_attention_layernorm.weight shape=[2048] dtype=BF16
  slice model.layers.0.self_attn.k_proj.weight shape=[256, 2048] dtype=BF16
  slice model.layers.0.self_attn.o_proj.weight shape=[2048, 2048] dtype=BF16
  slice model.layers.0.self_attn.q_proj.weight shape=[2048, 2048] dtype=BF16
  slice model.layers.0.self_attn.v_proj.weight shape=[256, 2048] dtype=BF16
  slice model.layers.1.input_layernorm.weight shape=[2048] dtype=BF16
  slice model.layers.1.mlp.down_proj.weight shape=[2048, 5632] dtype=BF16
  slice model.layers.1.mlp.gate_proj.weight shape=[5632, 2048] dtype=BF16
  slice model.layers.1.mlp.up_proj.weight shape=[5632, 2048] dtype=BF16
  slice model.layers.1.post_attention_layernorm.weight shape=[2048] dtype=BF16
  slice model.layers.1.self_attn.k_proj.weight shape=[256, 2048] dtype=BF16
  slice model.layers.1.self_attn.o_proj.weight shape=[2048, 2048] dtype=BF16
  slice model.layers.1.self_attn.q_proj.weight shape=[2048, 2048] dtype=BF16
  slice model.layers.1.self_attn.v_proj.weight shape=[256, 2048] dtype=BF16
  slice model.norm.weight shape=[2048] dtype=BF16
kept 21 tensors, wrote models/tinyllama_2layer/model.safetensors (876652798 bytes), num_layers=2
set JJ_HF_LLAMA_SLICE_DIR=.../models/tinyllama_2layer to run real-slice Engine test
~~~

有切片后，先确认路径能跑（smoke 只查 load / finite / KV 长度，不比 logits）：

~~~
export JJ_HF_LLAMA_SLICE_DIR=/home/junhui/workspace/moe/JJJetson_Ops/models/tinyllama_2layer
./run_tests.sh --suite test_hf_llama_real_slice_smoke.py
~~~

### dump HF logits（给 Engine 当数值 ref）

脚本：`scripts/export_hf_llama_slice_logits.py`。只读已经切好的 2 层目录，CPU + F32，不进 `run_tests.sh`。不要去加载 `models/hf_src/` 那份 22 层。Orin 大约 4GB，**不要**让 HuggingFace 占 GPU 的同时再跑 Engine。

在 `JJJetson_Ops`、cuda-ops 环境：

~~~
python scripts/export_hf_llama_slice_logits.py \
  --slice-dir models/tinyllama_2layer

python scripts/export_hf_llama_slice_logits.py \
  --slice-dir models/tinyllama_2layer --tokens 1
~~~

默认 token `[1, 2, 3, 4]`。ref 存成 `.npy`（numpy 的数组文件），在切片目录，不要提交：

~~~
ref_prefill_logits_t1234.npy    [vocab, 4] 列主序，last_argmax=6415
ref_prefill_logits_t1.npy       [vocab, 1] 列主序，last_argmax=2579
~~~

本机没有 `transformers` 就先装再 dump。dump 峰值只应是 2 层切片。

### 对比 suite（Engine vs 上面那张 npy）

同一 env，同一切片目录。测试里不 `import transformers`。无 env 或无 npy 会 skip。

~~~
export JJ_HF_LLAMA_SLICE_DIR=/home/junhui/workspace/moe/JJJetson_Ops/models/tinyllama_2layer
./run_tests.sh --suite test_hf_llama_real_slice_logits.py
~~~

只跑 T=4 那条：

~~~
./run_tests.sh --suite test_hf_llama_real_slice_logits.py \
  --case test_hf_llama_real_slice_logits
~~~

T=4 应对上 last-token argmax=6415。FA 的 QKV 是 fp16，和 HF 全 F32 不会 bitwise 相同；实测 `max_abs` 大约 `3.6e-3`，测试 `atol=rtol=1e-2`。差到十几说明结构又错了（以前 Linear 多转一次就是那种量级），不要把 tol 放到 20。
