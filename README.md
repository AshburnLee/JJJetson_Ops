
# 1. Build & run tests

~~~sh
conda activate cuda-ops
# 构建所有 op，默认是 release，
./build_all.sh
# debug mode：
./build_all.sh --debug

# 复制conda 环境
conda env create --file environment.yml
~~~

~~~sh
# 执行正确性检测
./run_tests.sh

# 测试单个op
export PYTHONPATH="${PWD}/python:${PYTHONPATH}"
# 如果需要python端的debug信息：
export DEBUG_MY_OPS=1
python ./tests/test_flash_attention.py

# kernel 基本性能, （ncu需要 sudo 权限）
export PYTHONPATH="$PWD/python"
export PROFILE_KERNEL_FROM_PYTHON=1
sudo -E env PYTHONPATH="$PYTHONPATH" $(which ncu) --section SpeedOfLight $(which python) tests/test_flash_attention.py

sudo -E env PYTHONPATH="$PYTHONPATH" $(which ncu) \
  --target-processes all \
  --kernel-name-base demangled \
  -k "regex:flash_attn" \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  $(which python) tests/test_flash_attention.py > log
~~~

可用环境变量：

- python 端的debug 模式，显示更多信息：`DEBUG_MY_OPS=1`
- python 端开启制定Kernel的 profile：`PROFILE_KERNEL_FROM_PYTHON=1`

另：资源有限的机器上，`./build_all.sh --debug` 可能会导致机器卡死，只构建制定目标： `cd build && make -j3 xxx_me `

---

# 2. CUDA Backend OPs

改项目目标：围绕 LLM 推理阶段常见算子，逐步实现可落地的 CUDA backend，并实现打磨每个算子的实现细节、正确性验证方式与优化。

## 项目目的

- 实现 LLM 推理中会用到的核心 OP（如 copy/transpose/attention/quantization/rope/moe gating 等），后端统一使用 CUDA。

- 逐步把当前分散的 OP 实现为可复用、可组合的 CUDA backend 算子部分。


## 当前内容

1. 确保各个 OP 的正确性  
   - 为每个 OP 构造 Python 测试/参考实现（ref），与 CUDA 输出逐项对齐。  

2. 记录 CUDA 实现中的优化点，作为优化 baseline
   - 对每个 OP 记录关键性能设计（并行映射、访存模式、warp/shared 使用、向量化、融合等）。
   - 形成“实现 + 解释 + 可复盘”的文档，便于后续调优和对比。

3. 持续迭代完善  
   - 新增 OP 时同步补测试与文档。  
   - 对已有 OP 持续补齐边界处理、类型支持和性能 profiling 结果。


## TODO

- 添加更多 LLM 推理相关 OP，扩展 dtype、shape 与布局覆盖范围。
- 补充当前实现中的缺口（正确性、接口一致性、性能基线、异常处理等）。
- 逐步把各 OP 串联为一个可用的 CUDA backend（可测试、可维护、可集成）。
