# 体现高新能的技术点

## 1. cpy_contnue

- 通过字节级 stride 地址计算，直接支持非连续内存。具体地，通过 nb0_0 ~ nb0_3 / nb1_0 ~ nb1_3 做真实地址偏移，不需要先做额外的重排/pack 再 copy。这点在处理 permute/padding 张量时很关键，避免了额外 kernel 或中间缓冲。
- 异步stream，便于与其他任务重叠执行。


## 2. cpy_transpose

- 使用 32*32 的 shared tile 将全局内存的转置转化为 读连续喝写连续的 模式。
- 32*（32+1）缓解 bank 冲突。
- 3D grid 处理 batched transpose，`blockIdx.z` 配合 `CUDA_CPY_BLOCK_NM=8`，一次 kernel 处理多张矩阵（nmat），提升并行度与吞吐
- 异步stream，便于与其他任务重叠执行。


## 3. flash_attention

- 分块并限制块的大小，充分利用 shared memory 硬件资源，同时避免重复读写，如 `s_tile`。保证把搞重复使用的数据留在片上。
- 合理的并行划分，此处是一个 block 处理2个 q_head。
- 半精度输入 + 单精度累加。计算 streaming softmax 时尽可能确保数值稳定。
- 向量化的加载与计算，通过 half2 类型。
- warp level 的reduce
- 将输出暂存 shared memory，最后一次性写入 Global
- 异步 stream，便于与其他任务重叠执行。


## 4. gated_unary

- 同 cpy_contnue，通过字节级 stride 地址计算，直接支持非连续内存。计算更直接。
- 通过 将函数指针作为非类型模版参数 让各个激活函数在 compile time 绑定。
- 本身是个融合 kernel ，融合了 unary 和 逐元素相乘。
- 异步 stream，便于与其他任务重叠执行。


## 5. q8_1

- warp level reduce
- 结构化存储，对应着向量化加载。每一个线程一次处理 4 个float
- 异步 stream，便于与其他任务重叠执行。


## 6. roll

- `__forceinline__ device` 避免函数调用开销，属于编译器优化。
- 异步stream，便于与其他任务重叠执行。


## 7. rope_neox

- 异步 stream，便于与其他任务重叠执行。

## 8. top_k_moe











## `top_k_moe.cu`

- 高度模板化以换取性能：
  - `template<int n_experts, bool with_norm, bool delayed_softmax>`
  - `softmax_warp_inplace` 也使用模板并配合 `#pragma unroll`
- 关键内核标注：
  - `__launch_bounds__(WARP_SIZE * 4, 1)` 帮助编译器进行寄存器/占用权衡
- warp 级并行策略：
  - 每个 warp 内对“候选 expert”做 max 选择，并使用 `__shfl_xor_sync` 并行归约最大值与其 index
- 通过局部数组分散存储避免 race：
  - `wt[experts_per_thread]` / `out_wt[experts_per_thread]` 作为每线程私有寄存器/局部空间，避免跨线程写冲突
- top-k 迭代选择（在寄存器中更新）：
  - 每次找到一个 top 值后，将其对应位置写为 `-INFINITY`，进入下一轮找 top-k
- 可选归一化/可选延迟 softmax：
  - `with_norm` 时使用 warp 级 sum（`warp_reduce_xor_sum`）并归一化
  - `delayed_softmax` 通过 `softmax_warp_inplace` 对已选 top-k 做 exp/sum/归一


