# Debug e2e：Engine logits 对不上 HuggingFace

我们拿 2 层 TinyLlama 切片、token `[1, 2, 3, 4]` 跑了一遍。Engine 最后一个位置 greedy 出来是 12323，HuggingFace 是 6415，整张 logits 表最大绝对差大概 12.3。你要是把 `atol` 放到 20 让测试变绿，那叫混过去，不是在查。正确做法是把整条链切开，看差是从哪一段开始冒出来的。

切片就在 `models/tinyllama_2layer/`，gitignore 了，别提交。HuggingFace 这边用 CPU dump，Engine 走 GPU。Orin 大概 4GB 统一内存，两边一起占会挤爆：先把 HF 结果存成 npy，再单独跑 Engine。

## 三种 [对得上]，别混在一块

你以后看到 [测过了]，先问自己是跟谁对：

~~~
A. Engine 自己跟自己
   smoke 和 4.2 都得到 argmax=12323
   这只说明你再跑一次还是这个数，不说明这个数是对的

B. Engine 跟自己写的公式
   FA kernel 去对 tests/fa_test_common.py 里的 fa_ref
   fixture、chain_linear_me_ref 也走同一套 FA
   公式和 kernel 一起写错的时候，内部单测照样全绿

C. Engine 跟 HuggingFace
   这才是步骤 4 要的：真模型数值对不对
~~~

这次炸的是 C。B 在 `tok_kv=32` 的时候是绿的，你换成真实 prefill 那种 `tok_kv=4`，B 自己也撑不住了。下面会讲到。

还有一件事容易绕晕：要比的不是那 21 个权重 tensor。两边都加载同一份切片，权重在 load 的时候就已经用上了。你对比的是算完之后那一张 logits 表，shape 是 `[vocab, T]`，列主序。比如 T=4、vocab=32000，就是 4 列词表分数，`logits[v, t]` 是第 t 个输入位置、词表第 v 维。


## 怎么切：一次只动一刀，看完数再决定下一刀

整条链是这样的：

~~~
token_ids -> embed -> N x layer(里面有 FA) -> final_norm -> lm_head -> logits
~~~

你从外往里切。每切一刀，看 max_abs 是 [几乎 0] 还是 [还是十几]，再决定下一刀切哪。别三处一起改。


### 第 1 刀：只用 T=1，先问问是不是缺 causal

当时有个很像样的怀疑：生产 FA 和 `fa_ref` 只把超出 `num_kv_tokens` 的 pad 打成 `-inf`，没有 `kv_col > q_row` 那种下三角。T=4 的时候 HuggingFace 是因果的：第 0 个 token 只能看自己，第 3 个才能看 0..3。Engine 这边四个 Q 都能看见全部 4 个 KV，等于双向。内部 fixture 绿，是因为 ref 也没做 causal，两边一起错。

怎么证伪？喂一个 token。T=1 没有 [未来位置]，因果和双向几乎是同一件事。要是 T=1 突然对上了、T=4 还差 12，那才该去补 mask。

dump 只跑 token `1`：

~~~
python scripts/export_hf_llama_slice_logits.py \
  --slice-dir models/tinyllama_2layer --tokens 1
~~~

会写出 `ref_prefill_logits_t1.npy`，shape `[32000, 1]`。Engine 也只喂 `[1]`。

跑出来是：

~~~
T=1  engine_argmax=21167  hf=2579  max_abs≈13.6
~~~

你看，T=4 差大约 12，T=1 差大约 13，一个量级。缺 causal **不是**这次的主因，先别去改 mask。


### 第 2 刀：只比 embed，问 Loader 和查表有没有先歪

dump 脚本会顺手写一份 `ref_embed_t1.npy`。HuggingFace 的 embedding 是 `[T, hidden]`，Engine 要 `[hidden, T]` 列主序，转置一下就对齐了。

举个最小的例子。T=1、hidden=3，HF 那一行是 `[0.1, 0.2, 0.3]`，Engine 的第 0 列就应该原样是这三个数。差哪怕 1e-5，都说明 gather 或权重 layout 有问题。

Engine 这边用现成的 `transformer_model_me.embed_forward_host`，别把 layer 建起来，否则你分不清是 embed 错还是后面错。

跑出来是 `embed_t1 max_abs=0`。bitwise 一样。Loader 和查表没问题，差在 layer 里面。


### 第 3 刀：FA 不加载 TinyLlama，只跟自己的公式比

你仓库里那条 `gqa8` 单测把 `tok_kv` 设成 32。KV tile 正好是 32，末 tile 是满的，所以它躲开了 [长度不是 32 倍数] 这条边界。真实 prefill 是 T=4，4 除以 32 余 4，末 tile 是瘪的。

所以加一条不碰真权重的测试：g=8、head_dim=64、`tok_q=tok_kv=4`，随机 QKV，`fa_me` 对 `fa_ref`。

跑出来是：

~~~
nan_heads = [24, 25, 26, 27, 28, 29, 30, 31]
前 24 个头 vs fa_ref  max_abs ≈ 6e-4
~~~

6e-4 是 fp16 staging 那种毛刺，前 24 头没问题。NaN 的那 8 个才是故事。


#### 病因：是指针读过了这个 KV 头该停的地方

先把两个词掰开，别混。文档开头那句 [两边一起占会挤爆]，说的是 **OOM**：Orin 大概 4GB，HuggingFace 和 Engine 同时占，`cudaMalloc` 会失败。那是流程约束，跟这一刀无关。你现在看到的是 **OOB**：buffer 已经分好了，kernel 还去读这块后面的地址。显存够不够是一回事，读没读过界是另一回事。

FA 搬 K/V 的时候，一次搬一个 tile，tile 写死是 32 行。意思是：shared memory 里每次都给 32 个 token 的位置。合法写法是：这 32 个格子里，有几个真 token 就从 global memory 搬几个，剩下的格子自己填 0。旧代码偷懒了：不管这回真正有几个 token，一律按 32 行去 global memory 读。

拿 TinyLlama 这次真实 prefill 举例。`num_kv_tokens=4`，head_dim=64。4 除以 32 余 4，全程就一个 tile，而且是瘪的：有效就 4 行。kernel 还是去读 32 行。那多出来的 28 行从哪来？

K/V 在设备上是按头摊开的：每个 KV 头自己一块平面，这块平面里只有 `4 * 64` 个数。头和头是紧挨着排的。所以：

~~~
KV 头 0  有效就 token 0..3
         你还去读 token 4..31  -> 其实读进了 KV 头 1 的内存
KV 头 1  同理，尾巴伸进 KV 头 2
KV 头 2  尾巴伸进 KV 头 3
KV 头 3  后面没有 [下一个头] 了
         token 4..31 已经出了这块分配  -> 垃圾，经常是 NaN
~~~

TinyLlama 是 `g=8`：4 个 KV 头，每个 KV 头管 8 个 Q。KV 头 3 对应的就是 Q 头 24 到 31。所以你看见 NaN 正好落在这 8 个 Q 上，不是巧合。

你可能会问：softmax 不是已经把 `kv_col >= num_kv_tokens` 打成 `-inf` 了吗？按理说第 4 列往后不该进输出啊。问题在更早一步。WMMA 做 QK 的时候是 16 列捆在一起乘的。那捆里既有合法的 4 列，也有你读出界读进来的 NaN。乘法先把 NaN 混进合法列的 score 里了，后面再 mask 也救不回来。这就叫 [药抹在伤口外面]：mask 管的是 [哪些列算进 softmax]，管不了 [乘法的时候有没有把垃圾乘进去]。

那前面 24 个 Q 头为什么看起来还行？因为 KV 头 0、1、2 多读的那 28 行，地址还落在整块 K/V buffer 里面，只是读成了 [下一个头] 的数。数是错的，但不是 NaN。softmax 又把这 28 列打成 `-inf`，错数被盖住了，`fa_ref` 对前 24 头还能 allclose。只有最后一个头，连 [下一个头] 都没有，才把 NaN 露出来。

仓库里原来那条 `gqa8` 单测把 `tok_kv` 设成 32。32 正好一整个满 tile，根本不会走到 [末 tile 是瘪的] 这条路，所以它一直绿，你一直没看见这病。换成真实长度 4，自己的 `fa_ref` 都对不上。

所以这一刀的病，就一句话：末 tile 不满 32 的时候，kernel 还按 32 行去 global memory 读；最后一个 KV 头没有 [下一块平面] 可借，读出界，Q 24–31 变成 NaN。不是 RoPE 先转错，也不是权重 2D 转置，更不是 4GB 不够。

注意范围：这是 **B 炸了** 的病因，也就是 FA 跟自己的公式对不上。把它堵住之后，B 绿了，C 还是差十几。Engine 对 HuggingFace 还有下一层的事，别把这一刀说成 [e2e 全修好了]。


### 第 4 刀：药

药是这样下的，末 tile 只从 global memory 读这个 KV 头还活着的那 n 行，T=4 就是 4 行；多出来的 32-n 行在 shared memory 里写 0。softmax 照旧把 `kv_col >= num_kv_tokens` 打成 `-inf`。清零是给 WMMA 看的：乘法那一捆里就算还有 pad 列，乘进去的也是 0，不会再把 NaN 混进合法列。

`test_fa_double_buffer_gqa8_tok4` 现在对 `tok_kv=1/4/13` 都能和 `fa_ref` allclose，Q 头 24–31 不再 NaN。


### 第 5 刀：embed 已经对齐了，再把两层跑完

你已经有一份和 HF 一模一样的 embed。把它喂给 Engine 的 `forward_hidden_host`：跳过查表，只跑 N 层再加 final RMSNorm。出来的 hidden 再去对 HF。

这里有个小坑：HuggingFace 的 `hidden_states` 最后一块，不一定已经过了 final norm。dump 的时候要用 `model.model.norm(hs[-1])` 存成 `ref_hidden_final_t1.npy`。这次 `hs[-1]` 和再 norm 一次差大约 5，你要对的是过完 norm 的那份，否则你会误以为 final_norm 算错了。

跑出来 `hidden_final_t1 max_abs≈20`，数还是有限的，没有整表变 NaN。说明层内已经歪了，和后面 logits 差 12 是对得上的：hidden 偏了，lm_head 再乘一次，argmax 当然飞。


## 这几刀的数字放在一起看

~~~
你切的位置                   看到的数              告诉你什么
---------------------------  --------------------  --------------------------------
Engine vs 上次 Engine        argmax 都 12323       稳定，不是测试把 Engine 读乱了
T=1 logits vs HF             max_abs≈13.6          先别只补 causal；修末 tile 后还是这个数
embed vs HF                  max_abs=0             Loader / gather 没问题
FA g=8 tok=1/4/13 vs fa_ref  allclose，无 NaN      末 tile 出界已经堵住
hidden 两层+norm vs HF       max_abs 仍约 20       差还在 layer 里（RoPE / 和 HF attention）
~~~

B 末 tile 已经堵住了。C 还没完，但差十几不是 fp16 毛刺：T=1 时 FA 跟自己的公式已经是 0 误差，embed 也是 bitwise 对齐。剩下是合同问题——同一份 TinyLlama 权重，Engine 按自己的 Transformer 跑，HF 按 Llama 前向跑。下一步不是继续猜 RoPE，而是先做 roadmap **4.1b 二选一**：贴 Llama 合同去改 Engine，或改口不把 HF logits 当黄金标准。未选之前 4.2 先别当绿；**不要**重切模型。


## 你要回头翻的文件

~~~
scripts/export_hf_llama_slice_logits.py     在 CPU 上让 HF dump logits / embed / hidden
tests/test_hf_llama_real_slice_logits.py    Engine 探针：T=1 logits、embed、hidden、T=4
tests/test_fa_double_buffer_shapes.py       gqa8 满 tile 回归；tok4 末 tile 探针
tests/fa_test_common.py                     跟 kernel 同一套公式的 numpy FA（没有 causal）
src/cuda/fa/fa_double_buffer_kernel.cuh     生产 FA，tile=32；末 tile 只读有效行
doc/guide/fa_device_api.md                  偶数 g 时 block 怎么领 Q 头
~~~
