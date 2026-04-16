
# Correctness

|op name| status |
|:---|:---|
|fa               |done|
|RoPE             |done|
|q8_1             |done|
|top_k_moe        |done|
|roll             |done|
|cpy_transpose    |done|
|cpy_continue     |done|
|gated_unary      |done|

# fa 正确性

结论：m 和 s 计算正确，l 计算不对，说明和ref相比的softmax(Q*K) 计算是正确的。

max abs diff m: 1.9073486e-05
max abs diff l: 11.665453
max abs diff S: 2.2888184e-05
max abs diff dst: 3.4609783

更多信息：
max abs diff m: 1.9073486e-05
max abs diff l: 11.665453
max abs diff S: 2.2888184e-05
max abs diff row_sum: 9.777383
max abs diff scale_old: 1.41859055e-05
max abs diff scale_new: 1.5318394e-05
max abs diff exp_val: 1.0
max abs diff dst: 3.4609783


## 1.可能原因

浮点加法不是严格结合的，cu是树形reducemax的，而ref是线性reducemax的。

原因是 reduce down 应该是reduce xor

max abs diff m: 1.9073486e-05
max abs diff l: 2.0503998e-05
max abs diff S: 2.2888184e-05
max abs diff row_sum: 1.5974045e-05
max abs diff scale_old: 1.41859055e-05
max abs diff scale_new: 1.5318394e-05
max abs diff exp_val: 1.5377998e-05
max abs diff dst: 1.446344e-05


### TODO：仔细分析这个原因

xor 归约这版的关键优势就是：每个 lane 最终都拿到同一个全局结果，因此后续在所有 lane 上做： `exp_val = expf(s - row_max)` 都保持一致，不会出现 lane 级不一致导致的偏差。

而 down 版本如果不额外广播 lane0 的结果，会出现： lane#0 是全局 max/sum，但是其他 lane 是部分归约结果。

当前case的关键在：
~~~cpp
float row_max = warp_reduce_down_max(s);
float exp_val = expf(s - row_max); // 这里每个thread计算自己的，故 row_max 需要是全局的（广播）非局部的
float row_sum = warp_reduce_down_sum(exp_val);
~~~

如果没广播 row_max：

lane0 的 row_max 是全局 max；其他 lane 的 row_max 不是全局 max（是部分归约值）。 所以这些 lane 的 exp_val 就错了。 然后 lane0 在做 row_sum 归约时，把这些“错的 exp_val”也加进来了，导致最终 lane0 的 row_sum 当然也会错。

已修复！

# RoPE 正确性

max abs diff: 1.2293458e-07
max abs diff: 2.682209e-07
max abs diff: 2.30968e-07
max abs diff: 4.3027103e-07
max abs diff: 2.6077032e-07
max abs diff: 3.5762787e-07
max abs diff: 9.536743e-07
max abs diff: 0.08268285     <- 错误

已修复！


# 类型对应

c++/cuda 中的4个类型：

|c++/cuda|torch|numpy|
|:---|:---|:---|
|float| torch.float32 |np.float32|    
|nv_bfloat16|torch.bfloat32|x(np.float32)|
|half|torch.float32|np.float32|
|int32_t|torch.int32|np.int32|


