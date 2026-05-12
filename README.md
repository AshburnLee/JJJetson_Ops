# JJJetson_Ops

面向 Jetson Orin，把 推理场景下经过业界验证的 SOTA 推理算子落成端侧、AI 边缘计算性能优异的 CUDA 实现；正确性依托 可对拍的参考与小例，持续用测试补强覆盖。


## 0. 真实推理场景下 Op 的性能对比

[WIP]

## 1. Prerequisites
### GPU 规格

| 项 | 值 |
|:---|:---|
| 设备名 | Orin |
| 计算能力 | 8.7 |
| SM 数量 | 4 |
| SM 频率（报告值） | 1.02 GHz |
| 显存频率（报告值） | 1.02 GHz |
| 显存位宽 | 128 bit |
| 总显存 | 3.52 GiB |
| L2 Cache | 2 MiB |
| 每 block 共享内存上限 | 48 KiB |
| 每 SM 共享内存容量 | 164 KiB |
| 每 block 最大线程数 | 1024（32 warps） |
| 每 SM 最大驻留线程数 | 1536（48 warps） |
| Warp 宽度 | 32 |
| 理论峰值 DRAM 带宽（估算） | 30.4 GiB/s |
| 理论峰值 FP32（估算） | 1.04 TFLOP/s |
| 理论峰值 FP16 Tensor Core dense（估算） | 4.18 TFLOP/s |


### 软件版本

| 项 | 值 |
|:---|:---|
| 操作系统 | 5.15.148-tegra aarch64 GNU/Linux |
| CMake | 3.22.1 |
| GNU Make | 4.3 |
| nvcc | release 12.6, V12.6.68 |
| CUB | 随 CUDA |
| cuBLAS | 随 CUDA |
| pybind11 | 2.13.6 |
| GCC 编译器 | GCC 11.4.0 |
| C++ / CUDA 标准 | C++17；CUDA C++17 |
| Python | 3.12.10 |
| Conda | 25.3.0 |
| NumPy | 2.1.2 |
| PyTorch | 2.6.0+cu126 |



## 2. Build & run tests

~~~sh
# 创建 conda 环境（首次）
conda env create --file environment.yml
conda activate cuda-ops

# 构建所有 op，默认是 release，
./build_all.sh
# debug mode：
./build_all.sh --debug
~~~

~~~sh
# 执行正确性检测
./run_tests.sh

# 测试单个 fa kernel（共用参考实现见 tests/fa_test_common.py）
export PYTHONPATH="${PWD}/python:${PYTHONPATH}"
export DEBUG_MY_OPS=1   # 可选：test_fa_one_pass 会跑 launch_fa_debug_ml debug用
python ./tests/test_fa_two_pass.py
python ./tests/test_fa_one_pass.py
python ./tests/test_fa_one_pass_parallel.py
python ./tests/test_fa_tc.py

# kernel 基本性能, （ncu需要 sudo 权限）
export PYTHONPATH="$PWD/python"
export PROFILE_KERNEL_FROM_PYTHON=1
sudo -E env PYTHONPATH="$PYTHONPATH" $(which ncu) --section SpeedOfLight $(which python) tests/test_fa_one_pass_parallel.py

sudo -E env PYTHONPATH="$PYTHONPATH" $(which ncu) \
  --target-processes all \
  --kernel-name-base demangled \
  -k "regex:fa_kernel" \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  $(which python) tests/test_fa_one_pass_parallel.py > log
~~~

可用环境变量：

- python 端的debug 模式，显示更多信息：`DEBUG_MY_OPS=1`
- python 端开启制定Kernel的 profile：`PROFILE_KERNEL_FROM_PYTHON=1`

另：资源有限的机器上，`./build_all.sh --debug` 可能会导致机器卡死，只构建制定目标： `cd build && make -j3 xxx_me `
