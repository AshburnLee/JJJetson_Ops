# 体现高性能的技术点

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

# Perf 记录

平台 Jetson Orin Nano，SM 频率 305.99 Mhz，确保 Release 编译

|kernel | gridxblock |shape |duration|
|---|---|---|---|
|flash_attn_tile_kernel | (8,1,1),(32,4,1) | .. |  3.39 ms|
|quantize_q8_1_kernel   | (13,4,1),(128,1,1) | .. | 26.78 us|
|rope_neox_kernel | (208,1,1),(1,256,1) | .. | 48.54 us|
|top_k_moe_kernel | (1024,1,1),(32,4,1) | .. | 1.02 ms|
|cpy_transpose    | (8,4,1),(32,8,1) | (256,128,4) | 145.73 us -> 52.22 us |
|cpy_continue| (64,1,1),(416,1,1) | (128,16,13,1) | 105.31 us|
|roll        | (256,1,1),(2048,1,1) | (2048,256,1,1) | 1.13 ms|
