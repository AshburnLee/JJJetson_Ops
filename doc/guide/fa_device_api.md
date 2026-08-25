# FA Device API

Flash Attention **生产路径**：`fa_double_buffer_forward_device`（`src/cuda/fa/fa.h`，实现 `src/cuda/fa/fa_double_buffer.cu`）。

`fa/` 下其余 kernel 为中间优化实验，**engine 勿用**。

## Layout (col-major)

~~~
Q:   fp16  [head_dim, num_q_tokens, num_q_heads, 1]
K/V: fp16  [head_dim, num_kv_tokens, num_kv_heads, 1]
dst: fp32  [head_dim, num_q_tokens, num_q_heads, 1]
~~~

## 约束（Phase 1）

- `head_dim` ∈ {32, 64, 128}（16 的倍数，WMMA）
- `num_q_heads / num_kv_heads = g`，`g` 为偶数且 `g >= 2`（每 block 仍 2 个 Q head）
  - `g=2`：与旧 kernel 相同，`grid = num_kv_heads`
  - `g=8`：TinyLlama，每个 KV 头 4 个 block
- `1 <= num_q_tokens <= 16`
- `num_kv_tokens >= 1`（末 tile 自动 mask）
- Q/K/V fp16；scale 通常为 `1/sqrt(head_dim)`


## Trick：偶数 g 不必改 WMMA

每个 CUDA block **从来都只做 [2 个 Q 对上 1 组 K/V]**。倍数变了，变的不是这块计算，而是开多少个这样的 block、每个 block 去领哪两个 Q。

GQA 的本质是：很多 Q 头共用同一组 K/V。Attention 本身按 Q 头独立，只是 K/V 指针相同。所以不必让一个 block 一次吃 8 个 Q；把 8 个 Q 拆成 4 对，用 4 个同样的 block 就行。

小例子（TinyLlama：`g=8`，4 个 KV 头，32 个 Q 头）：

~~~
KV 头 0 被 Q 0..7 共用。拆成 4 个 block，每个仍是 2 个 Q：

block   kv_h   pair   q0  q1    读哪组 K/V
-----------------------------------------
  0       0      0     0   1     K/V 头 0
  1       0      1     2   3     K/V 头 0   <- 同一份 KV
  2       0      2     4   5     K/V 头 0
  3       0      3     6   7     K/V 头 0
  4       1      0     8   9     K/V 头 1
  ...
 15       3      3    30  31     K/V 头 3
~~~

`g=2` 时 `pair` 只能是 0，于是 `q0 = kv_h * 2`，grid 也还是 `num_kv_heads` 个 block，和改之前一模一样。所以旧测试不用动。

公式就这一组（input：`blockIdx.x`、`g`；output：这个 block 的 KV 头和两个 Q 头）：

~~~
pairs = g / 2                    # 每个 KV 要开几对（几 block）
kv_h  = blockIdx.x / pairs
pair  = blockIdx.x % pairs
q0    = kv_h * g + pair * 2
q1    = q0 + 1
grid  = num_q_heads / 2          # 对 g=2 等于 num_kv_heads
~~~

WMMA、softmax、双缓冲这些 **完全不知道 g**。它们只看见：shared 里两行 Q、一份 K/V tile。这就是为什么 kernel 结构几乎没变。

限制也还在：每 block 固定 2 个 Q，所以 `g` 必须是偶数。`g=1`（纯 MHA，Q 和 KV 头数相同）这条路还是走不通，那才需要改 block 内部。

## 数据流

~~~
Linear(Q/K/V)  ──► fp16 device Q/K/V
    │
    ▼
fa_double_buffer_forward_device(stream, &shape, d_q, d_k, d_v, d_dst, scale)
    │  WMMA + K/V 双缓冲 cp.async；(num_q_heads/2) blocks × 每 block 2 Q-heads
    ▼
d_dst (fp32) ──► Linear(O)
~~~

## API

~~~
FaDoubleBufferShape
  head_dim / tokens / heads 结构体

fa_double_buffer_validate_shape
  host 校验

fa_double_buffer_forward_device
  生产 device 入口

fa_double_buffer_forward_host
fa_double_buffer_forward_host_legacy
  测试 H2D/D2H

fa_me.forward_host_shape
  Python: 从 Q/K/V shape 推断

fa_me.forward_host
  Python: legacy 固定 shape
~~~

Legacy 固定 128×13×16 / KV256 仍可用 `fa_me.launch_fa` / `forward_host`。

## 待办（Runner 接入）

- fp32 Q/K/V 与 Linear 输出对接（cast 或 fp32 路径）
- KV cache 读写与 decode 变长 seq
