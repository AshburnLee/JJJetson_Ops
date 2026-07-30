# Elementwise Device API

通用二元逐元素算子（add / sub / mul / div），单 kernel 模板 + device functor 实现；Residual add 使用 `elementwise_add_forward_device`（`src/cuda/elementwise.h`）。

## Residual add 在 Pre-LN block 中的位置

`elementwise_add_forward_device` 用于 **子块出口**：把 subblock 输出加回 residual stream（plain add，无 Norm）。

### ① Post-Attn residual add（可选独立调用）

Attn 子块（Q/K/V → RoPE → FA → O）算完后，与 **Attn 前保存的 residual** 相加：

~~~
hidden_mid[device]  （Attn 前 fused add 写入的 residual = z_attn）
    │
    ├──────────────────────────────────────────────────────────────┐
    │                              skip（residual stream）         │
    ▼                                                              │
 Q/K/V ──► RoPE ──► FA ──► O Linear ──► attn_out[d_subblock]       │
    │                                                              │
    └──────────────► elementwise_add_forward_device ◄──────────────┘
                     (d_a=residual, d_b=attn_out, d_out=hidden_mid)
                              │
                              ▼
                        hidden_mid  ──► 下一子块 Pre-FFN rms_norm_fused_add ...
~~~

**Runner 可选优化**：不单独调 add，直接把 `attn_out` 作为 input、`hidden_mid` 作为 residual 传入下一次 `rms_norm_fused_add_forward_device`（语义等价于先 add 再 norm）。

### ② Post-FFN residual add（必须 plain add）

FFN 子块算完后，与 **FFN 前保存的 residual** 相加，得到层输出：

~~~
hidden_out_buf[device]  （FFN 前 fused add 写入的 residual = z_ffn）
    │
    ├──────────────────────────────────────────────────────────────────────────┐
    │                              skip（residual stream）                     │
    ▼                                                                          │
 rms_norm_fused_add ──► gate/up ──► SwiGLU ──► down Linear ──► ffn_out[d_subblock] │
    │                                                                          │
    └──────────────► elementwise_add_forward_device ◄──────────────────────────┘
                     (d_a=residual, d_b=ffn_out, d_out=hidden_out)
                              │
                              ▼
                        hidden_out  ──► 下一层 / 输出
~~~

### 调用约定

~~~
d_a    residual stream（子块入口前保存的 hidden）
d_b    subblock 输出（attn_out 或 ffn_out）
d_out  相加结果；可与 d_a in-place
n_elem hidden_size * num_tokens * batch（col-major 展平）
~~~

~~~
elementwise_add_forward_device(stream, d_residual, d_subblock, d_out, n_elem);
// 等价于 d_out[i] = d_residual[i] + d_subblock[i]
~~~

## 实现

- 单 kernel：`elementwise_binary_kernel<Op>`，`Op` 为 device functor（`ElementwiseAddOp` 等）
- Host 侧 `switch(op)` 实例化不同模板，避免为每种 op 复制 kernel 体
- 支持 in-place：`d_out` 可与 `d_a` 或 `d_b` 相同

## API

~~~
elementwise_binary_forward_device
  生产: 通用二元 op（ElementwiseBinaryOp 枚举）

elementwise_add_forward_device
  生产: add（residual + subblock）

elementwise_sub_forward_device / mul / div
  生产: sub / mul / div

elementwise_binary_forward_host
  测试: host 包装
  Python: elementwise_me.forward_host(op, a, b, out)
~~~

Layout：连续 flat buffer；hidden tensor 为 col-major `[hidden_size, num_tokens, 1, batch]` 时 `n_elem = hidden_size * num_tokens * batch`。

## Python 测试

~~~python
import elementwise_me

elementwise_me.forward_host("add", residual_np, subblock_np, out_np)
elementwise_me.forward_host("sub", a_np, b_np, out_np)
~~~

`op` 字符串：`add` / `sub` / `mul` / `div`。
