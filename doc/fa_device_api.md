# FA Device API

Flash Attention **生产路径**：`fa_double_buffer_forward_device`（`include/fa.h`，实现 `src/cuda/fa/fa_double_buffer.cu`）。

`fa/` 下其余 kernel（`fa_one_pass_parallel`、`fa_one_pass_parallel_tc`、`fa_one_pass_parallel_tc_true` 等）为中间优化实验，**engine 勿用**。

## 固定形状（Phase 1 kernel 常量）

与 `tests/fa_test_common.py` 一致：

| Tensor | dtype | layout (col-major) |
|--------|-------|-------------------|
| Q | fp16 | [128, 13, 16, 1] |
| K | fp16 | [128, 256, 8, 1] |
| V | fp16 | [128, 256, 8, 1] |
| dst | fp32 | [128, 13, 16, 1] |

## 数据流

~~~
Linear(Q/K/V)  ──► fp16 device Q/K/V
    │
    ▼
fa_double_buffer_forward_device(stream, d_q, d_k, d_v, d_dst, scale)
    │  WMMA + K/V 双缓冲 cp.async；8 blocks × 2 Q-heads
    ▼
d_dst (fp32) ──► Linear(O)
~~~

## 接入 TransformerRunner（待办）

~~~
RoPE(Q,K) in-place
    │
    ├─ 当前：D2D 占位 d_q ──► d_attn_out
    │
    └─ 目标：fa_double_buffer_forward_device + KV cache 读写
              │
              ▼
         Linear(O)
~~~

**阻塞点**：现有 FA kernel 固定 head_dim=128、fp16 QKV；`test_transformer_runner` 使用 head_dim=32、fp32。接入 Runner 前需泛化 FA 或增加 FA 对齐的集成 profile。

## API

| 函数 | 用途 |
|------|------|
| `fa_double_buffer_forward_device` | 生产：device 上 Q/K/V in，dst out |
| `fa_one_pass_parallel_double_buffer` | 测试：host H2D/D2H 包装 |
| `fa()` / `fa_me.launch_fa` | 默认生产 alias |
| `fa_me.forward_device` | Python 测试入口 |

scale 通常为 `1/sqrt(head_dim)`。
