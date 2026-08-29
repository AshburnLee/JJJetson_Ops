# 读懂本地 HF checkpoint（TinyLlama 为例）

切片之前，先看磁盘上已经下好的两样东西：`config.json`（说明书）和 `model.safetensors`（权重本体）。`config.json` 已读完；本节起讲 safetensors：先解释名字里的 safe，二进制布局下一步再拆。

本机示例目录：

~~~
JJJetson_Ops/tmp/hf_src/TinyLlama__TinyLlama-1.1B-Chat-v1.0/
  config.json
  model.safetensors
~~~

`tmp/` 在 `.gitignore` 里，不进 git。映射与切片流程见 `hf_llama_weight_map.md`。

---

## 1. 这是谁：architectures / model_type / torch_dtype

打开 `config.json`，先对这三行：

~~~
"architectures": ["LlamaForCausalLM"]
"model_type": "llama"
"torch_dtype": "bfloat16"
~~~

怎么读：

- `LlamaForCausalLM`：HuggingFace 里的 Llama 因果语言模型，读前面的 token、预测下一个。Phase 1/2 的 layer 链就是按这个族做的（Pre-LN、RoPE、SwiGLU）。
- `model_type: llama`：告诉 Loader 按 Llama 的权重名去映射。不是 Mistral MoE，也不是 GPT-2。
- `torch_dtype: bfloat16`：磁盘上的权重是 BF16。引擎 Loader 目前只吃 F32，切片时必须把每个 tensor 转成 F32。这不是可选项。

这三行对不上，就不要往下切，也不要往引擎里塞。

---

## 2. 有多大：hidden_size / intermediate_size / num_hidden_layers

同一文件里再对这三行（TinyLlama-1.1B 的值）：

~~~
"hidden_size": 2048
"intermediate_size": 5632
"num_hidden_layers": 22
~~~

怎么读：

- `hidden_size: 2048`：每个 token 的隐藏向量长度。embed 之后、每一层进出，都是 2048 维。引擎 `ModelConfig` 里同名。
- `intermediate_size: 5632`：FFN（SwiGLU）中间变胖的那一维。gate / up 是 `2048 -> 5632`，down 是 `5632 -> 2048`。
- `num_hidden_layers: 22`：完整模型 22 层，磁盘上 `model.layers.0` 一直到 `model.layers.21`。Orin 显存扛不住 22 层 F32，开发期只切前 2 层；切完之后引擎看到的 `num_layers` 是 2，不是 22。

HF 字段和引擎字段：

~~~
HF config.json              引擎 ModelConfig
--------------------------------------------
hidden_size            ->   hidden_size
intermediate_size      ->   intermediate_size
num_hidden_layers      ->   num_layers   （切片时改成 N，例如 2）
~~~

切片脚本 `scripts/export_hf_llama_slice.py` 读的就是这些字段，再写出切片目录里的 `config.txt`。

---

## 3. Attention 头：num_attention_heads / num_key_value_heads

同一文件里对这两行。TinyLlama 的 `config.json` **没有** `head_dim` 字段，要自己除出来。

~~~
"num_attention_heads": 32
"num_key_value_heads": 4
~~~

怎么读：

- `num_attention_heads: 32`：Query 的头数。引擎里叫 `num_q_heads`。
- `num_key_value_heads: 4`：Key / Value 的头数。引擎里叫 `num_kv_heads`。比 Q 少，这就是 GQA（Grouped Query Attention）。
- `head_dim`：每个头的宽度。HF 没写时，用 `hidden_size / num_attention_heads`。这里是 `2048 / 32 = 64`。Loader 也是这么算的。

Q 头多、KV 头少是为了省 KV cache：32 个 Q 头共用 4 组 K/V，每一组 K/V 被 `32 / 4 = 8` 个 Q 头复用。cache 体积大约是 [Q/K/V 头数相同] 时的 `4/32 = 1/8`。

小例子（一个 token、一层）：

~~~
hidden = 2048，head_dim = 64

Q: 32 头 * 64 = 2048 个数   ->  q_proj 权重 HF shape [2048, 2048]
K:  4 头 * 64 =  256 个数   ->  k_proj 权重 HF shape [ 256, 2048]
V:  4 头 * 64 =  256 个数   ->  v_proj 权重 HF shape [ 256, 2048]
O: 拼回 2048               ->  o_proj 权重 HF shape [2048, 2048]
~~~

（HF Linear 是 `[out, in]`，引擎 Linear 也按这个读，见 `hf_llama_weight_map.md`。只有 lm_head 会转成 `[hidden, vocab]`。）

对应关系：

~~~
HF config.json                 引擎 ModelConfig
-----------------------------------------------
num_attention_heads       ->   num_q_heads        (32)
num_key_value_heads       ->   num_kv_heads       (4)
(hidden / q_heads)        ->   head_dim           (64，config 里没有这个键)
~~~

如果 `num_key_value_heads` 等于 `num_attention_heads`，那就是普通 MHA，不是 GQA。TinyLlama 明确是 GQA。

---

## 4. 序列长度与 RoPE：max_position_embeddings / rope_theta

同一文件里对这几行：

~~~
"max_position_embeddings": 2048
"rope_theta": 10000.0
"rope_scaling": null
~~~

怎么读：

- `max_position_embeddings: 2048`：这份权重按最长 2048 个位置训练。引擎里对应 `max_seq_len`：KV cache 按这个长度预分配。Orin 上开发期不需要 2048，切片时常用 `--max-seq-len 256`，短 prefill 够用，KV 更小。
- `rope_theta: 10000.0`：RoPE（旋转位置编码）的基频。引擎里叫 `freq_base`。位置 `0, 1, 2, ...` 不是另学一套 embedding 表，而是用这个基频把角度编进 Q/K。
- `rope_scaling: null`：没有 YaRN / linear 这类外推缩放。位置就按普通 RoPE 算，别把 scaling 公式套上去。

小例子（只看基频怎么进 cos/sin，不是完整 kernel）：

~~~
freq_base = 10000，head_dim = 64
第 0 对频率：10000^(-0/64) = 1
第 1 对频率：10000^(-2/64) ≈ 0.749
位置 pos=1 时，这对的角度 = pos * 频率，再变成 cos/sin 乘到 Q/K 的偶数/奇数维上。
~~~

对应关系：

~~~
HF config.json                 引擎 ModelConfig
-----------------------------------------------
max_position_embeddings   ->   max_seq_len     （切片时可改成 256）
rope_theta                ->   freq_base       (10000.0)
rope_scaling: null        ->   （引擎当前不读这一项）
~~~

`max_seq_len` 改小不会改权重，只改 KV / RoPE cache 分配上限。权重本身还是按 2048 训出来的；你喂超过 `max_seq_len` 的序列，引擎会报错，不是静默截断。

---

## 5. 词表：vocab_size / bos_token_id / eos_token_id

同一文件里对这三行：

~~~
"vocab_size": 32000
"bos_token_id": 1
"eos_token_id": 2
~~~

怎么读：

- `vocab_size: 32000`：词表有 32000 个 token。embed 一行对应一个 id，lm_head 输出 32000 维 logits。引擎 `ModelConfig.vocab_size` 就是这个数。你喂给引擎的 `token_ids` 必须落在 `[0, 32000)`。
- `bos_token_id: 1`：句子开头常用的 id。这份目录里没有 tokenizer 文件，id 已经写在 config 里了。
- `eos_token_id: 2`：结束符。GenerateLoop 用它判断该停了（`eos_token_id` 参数）。`ModelConfig` **不存** bos/eos，只存 `vocab_size`。

小例子：

~~~
token_id = 1
embed 权重 HF shape [32000, 2048]   -> 取出第 1 行，得到 2048 维 hidden
... 跑完 N 层 ...
lm_head 把 2048 维投到 32000 维     -> logits[2] 是 [产出 eos] 的分数
~~~

对应关系：

~~~
HF config.json                 引擎
-----------------------------------------------
vocab_size                ->   ModelConfig.vocab_size     (32000)
bos_token_id              ->   （不进 ModelConfig；tokenizer / 调用方使用）
eos_token_id              ->   GenerateLoop 的 eos_token_id 参数
~~~

这份 `tmp/hf_src/...` 只有 `config.json` + `model.safetensors`，没有 `tokenizer.json`。真权重冒烟测试（`test_hf_llama_real_slice_smoke.py`）用的是手写 id `[1, 2, 3, 4]`，不走分词器。

---

## 6. lm_head 是否共用、RMSNorm eps：tie_word_embeddings / rms_norm_eps

同一文件里对这两行：

~~~
"tie_word_embeddings": false
"rms_norm_eps": 1e-05
~~~

怎么读：

- `tie_word_embeddings: false`：embed 和 lm_head **各有一块权重**。safetensors 里会有 `lm_head.weight`。引擎 `ModelConfig.tie_word_embeddings = 0`（untied）。若是 `true`，lm_head 不单独存，投影时复用 `embed`（转置），切片脚本也会跳过 `lm_head.weight`。TinyLlama-1.1B-Chat 是 false，所以切片必须把 `lm_head.weight` 带上。
- `rms_norm_eps: 1e-05`：RMSNorm 分母里的小常数。引擎叫 `rms_norm_epsilon`。每层的 `input_layernorm` / `post_attention_layernorm`，以及最后的 `model.norm`，都用同一个 eps。

小例子（一个数、一维 RMSNorm 的精神，不是完整 kernel）：

~~~
x = [3, 4]，weight = [1, 1]，eps = 1e-5
mean(x^2) = (9+16)/2 = 12.5
rms = sqrt(12.5 + 1e-5) ≈ 3.5355
y = x / rms * weight ≈ [0.849, 1.131]
~~~

`eps` 是为了 x 全接近 0 时分母不为 0。它不是可训练参数，改 config 等于改计算，不要随便改。

对应关系：

~~~
HF config.json                 引擎 ModelConfig
-----------------------------------------------
tie_word_embeddings       ->   tie_word_embeddings   (false -> 0，true -> 1)
rms_norm_eps              ->   rms_norm_epsilon      (1e-5)
~~~

---

## 7. 激活与 bias：hidden_act / attention_bias

同一文件里对这两行，`config.json` 到这里就读完了。

~~~
"hidden_act": "silu"
"attention_bias": false
~~~

怎么读：

- `hidden_act: silu`：FFN 里 gate 那路的激活。SiLU 就是 `x * sigmoid(x)`。Llama 的 FFN 是 SwiGLU：`down( silu(gate(x)) * up(x) )`。引擎 FFN 按这条做，**不读** `hidden_act` 字段；若哪天遇到 `gelu` 模型，对不上，不能硬加载。
- `attention_bias: false`：Q/K/V/O 的 Linear **没有 bias**，只有 weight。引擎的 `linear_forward_device` 也是只有权重。safetensors 里你不会看到 `q_proj.bias` 这类键。

小例子（SwiGLU 两个数）：

~~~
gate = 1.0，up = 2.0
silu(1.0) = 1.0 * sigmoid(1.0) ≈ 0.731
silu(gate) * up ≈ 1.462
再乘 down 权重（这里省略）得到 FFN 输出
~~~

引擎不读、可以忽略的字段：

~~~
initializer_range       训练初始化标准差，推理不用
pretraining_tp          训练期 tensor parallel 切分，单卡推理不用
use_cache               HF generate 是否缓存 KV；你们引擎自己管 KVCache
transformers_version    导出时的 transformers 版本号
~~~

对应关系：

~~~
HF config.json                 引擎
-----------------------------------------------
hidden_act: silu          ->   写死 SwiGLU（不读该字段，但必须是 silu）
attention_bias: false     ->   Linear 无 bias（不读该字段，权重里也没有 bias）
~~~

---

## 8. 为什么叫 safetensors：safe 指什么

文件名是 `model.safetensors`，不是 `pytorch_model.bin` / `model.pt`。**safe 指的是加载时不能执行代码**，不是指数值更准、也不是加密。

旧格式（PyTorch `.pt` / `.bin`）底层常常是 Python pickle：反序列化时可以跑任意 Python。一张伪装成模型的文件，`torch.load` 的，这很危险，不安全。这是格式能力，不是实现写错。

safetensors 故意只能装两样东西：

~~~
1. 一段 JSON：tensor 名字、dtype、shape、文件内偏移
2. 一段纯数字：BF16 / F32 / F16 的原始字节
~~~

没有 class、没有 `eval`、没有自定义 layer 代码。你们 Loader 读它，只是按偏移把字节拷进数组。哪怕文件来自不信任的人，最坏是权重是垃圾数，不会变成一次远程代码执行。

对比（只谈加载安全性）：

~~~
格式                  加载时会不会跑代码
-----------------------------------------
.py / pickle .pt      会（这就是 unsafe）
safetensors           不会（这就是 safe）
config.json           不会（纯文本说明书）
~~~

顺带还有：可以只读 header 再按需 mmap 某一层，不必把 2GB 一次性 pickle 进 RAM。这是性能，不是 safe 这个词的本义。

Hugging Face 后来把 safetensors 做成默认权重格式，就是因为这点。你们引擎步骤 1 只认 safetensors、不认 `.pt`，同一原因。

---

## 9. 2GB 文件怎么看：先读 header，不要用 IDE 打开

`model.safetensors` 大约 2.1G，编辑器打不开是正常的。不要 `cat`、不要拖进 IDE。它不是文本。

拆法：只读文件开头那一小段目录，不碰后面的权重字节。

~~~
[8 字节，小端 uint64]     header 有多长
[接着 N 字节 JSON]        每个 tensor 的名字 / dtype / shape / 偏移
[再往后直到文件末尾]      真正的 BF16 数字（这一步暂时不读）
~~~

这份 TinyLlama 实测：header 长度 **23088 字节（约 22.5KB）**，登记了 **201** 个 tensor。22KB 用 Python 读 JSON 就行，和打开 2GB 完全是两件事。

切片脚本 `scripts/export_hf_llama_slice.py` 也是这套：先扫 header 决定留哪些名字，再按偏移去文件里抠那几块，转成 F32 写出去。峰值内存大约一张 embed，不是整模。

自己看 header（在 `JJJetson_Ops` 下）：

~~~py
python3 -c "
import json, struct
p='tmp/hf_src/TinyLlama__TinyLlama-1.1B-Chat-v1.0/model.safetensors'
with open(p,'rb') as f:
    n=struct.unpack('<Q', f.read(8))[0]
    h=json.loads(f.read(n))
print('header_bytes', n, 'n_tensors', len([k for k in h if k!='__metadata__']))
print(h['model.layers.0.input_layernorm.weight'])
"
~~~
结果：
~~~sh
header_bytes 23088 n_tensors 201
{'dtype': 'BF16', 'shape': [2048], 'data_offsets': [262144000, 262148096]}
~~~

下一步会拿这 8 字节和 JSON 里的一项（dtype / shape / data_offsets）对着讲。

---

## 10. 宏观长什么样：三截拼接，不是文本模型

不要想象成 [一篇能滚动的文章]。它是 **8 字节整数 + 一段 JSON 目录 + 一大段密密的数字**。下面全是你这份 TinyLlama 的真实切片。

### 整文件比例

~~~
总长 2200119864 字节 ≈ 2.05 GiB

 0          8                    23096                         文件末尾
 |-- 8 B --|-------- 22.5 KB --------|------------ ~2.05 GiB ------------|
 | 长度 N  |   JSON 目录（201 项）    |   BF16 权重数据（按 offsets 紧挨）  |
~~~

前 8 字节的十六进制：

~~~
30 5a 00 00 00 00 00 00
~~~

小端 uint64：`0x5a30 = 23088`。意思是：从第 9 个字节起，往后 23088 字节是 JSON。数据区从文件偏移 `8+23088=23096` 开始。

### JSON 目录开头（原文切片）

IDE 打不开 2GB，但这 22KB 是普通 UTF-8。开头长这样（截断）：

~~~
{"__metadata__":{"format":"pt"},
 "lm_head.weight":{"dtype":"BF16","shape":[32000,2048],"data_offsets":[0,131072000]},
 "model.embed_tokens.weight":{"dtype":"BF16","shape":[32000,2048],"data_offsets":[131072000,262144000]},
 "model.layers.0.input_layernorm.weight":{"dtype":"BF16","shape":[2048],"data_offsets":[262144000,262148096]},
 ...}
~~~

`__metadata__.format=pt` 只是标注 [从 PyTorch 导出]，不是再跑 pickle。

键大致按字母序：`lm_head` -> `embed` -> `layers.0` ... `layers.9`（字符串序，所以 9 在 10 后面出现）-> `model.norm`。

### 数据区按目录排好的积木

`data_offsets` 是 **相对数据区起点** 的 `[start, end)`，单位字节。BF16 每个数 2 字节。

~~~sh
相对数据区          名字                         shape            字节数
---------------------------------------------------------------------------
[0          , 131072000 )  lm_head.weight              [32000, 2048]   125.0 MiB
[131072000  , 262144000 )  model.embed_tokens.weight   [32000, 2048]   125.0 MiB
[262144000  , 262148096 )  layers.0.input_layernorm    [2048]            4.0 KiB
[262148096  , 285216768 )  layers.0.mlp.down_proj      [2048, 5632]    22.0 MiB
... 中间 layer0 其余 6 块，再 layer1 .. layer21 ...
[2200092672 , 2200096768)  model.norm.weight           [2048]            4.0 KiB
---------------------------------------------------------------------------
数据区合计 2200096768 字节
~~~

每一层固定 9 块（22 层 × 9 = 198），加上 embed / final_norm / lm_head，正好 **201**。`x` 是层号 0..21：

~~~sh
model.layers.x.input_layernorm.weight
model.layers.x.self_attn.q_proj.weight
model.layers.x.self_attn.k_proj.weight
model.layers.x.self_attn.v_proj.weight
model.layers.x.self_attn.o_proj.weight
model.layers.x.post_attention_layernorm.weight
model.layers.x.mlp.gate_proj.weight
model.layers.x.mlp.up_proj.weight
model.layers.x.mlp.down_proj.weight
~~~

整份 model.safetensors 还要加上 3 块层外的：

~~~sh
embed          model.embed_tokens.weight
lm_head        lm_head.weight          （这份 TinyLlama 没 tie，所以单独有）
final_norm     model.norm.weight
~~~

回忆 engine中的名称的map：

~~~cpp
    static const Rule kRules[] = {
        {"self_attn.q_proj.weight", "w_q", true},
        {"self_attn.k_proj.weight", "w_k", true},
        {"self_attn.v_proj.weight", "w_v", true},
        {"self_attn.o_proj.weight", "w_o", true},
        {"mlp.gate_proj.weight", "w_gate", true},
        {"mlp.up_proj.weight", "w_up", true},
        {"mlp.down_proj.weight", "w_down", true},
        {"input_layernorm.weight", "w_input_layernorm", false},
        {"post_attention_layernorm.weight", "w_post_attention_layernorm", false},
    };
~~~

### 真正的数字长什么样（只抠 8 个数）

`layers.0.input_layernorm` 声明 `[2048]` BF16，应占 `2048*2=4096` 字节。从数据区偏移 262144000 读出前 16 字节：

~~~
hex:  89 bb  cf 3b  8f 3d  f1 bc  c9 bb  79 bc  96 3b  88 3b
转F32: -0.00418, 0.00632, 0.0698, -0.0294, -0.00613, -0.0152, 0.00458, 0.00415
~~~

这才是权重。header 里的 `"dtype":"BF16"` 是说明书；这些 hex 才是数。其余 2GB 同理，只是块更大，不必打印。

切片脚本的宏观动作：读完这份 22KB 目录 -> 只拷 layer0/1 那 9+9 块，加上 embed/norm/lm_head -> 每块 BF16 转 F32 -> 写成新的、小很多的 safetensors。
