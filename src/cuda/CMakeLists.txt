#
# src/cuda/CMakeLists.txt
# - 目标: 为每个 CUDA 算子 (.cu) 定义一个 OBJECT 库
# - 用途: 只负责把 CUDA 源代码编译成目标文件, 由 bindings 层的 MODULE 目标复用
# - 关键逻辑:
#   * 统一设置 CUDA 标准、架构、-fPIC
#   * 为每个算子 (gated_unary/rope/q8_1/flash_attention/...) 建立 OBJECT target
#   * 在 foreach 中为所有 CUDA target 打开 POSITION_INDEPENDENT_CODE 并加入 include 目录
#
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC -lineinfo")
set(CMAKE_CUDA_ARCHITECTURES 87)

add_library(gated_unary_cuda OBJECT
  gated_unary.cu
)

add_library(rope_cuda OBJECT
  rope_neox.cu
)

add_library(q8_1_cuda OBJECT
  q8_1.cu
)

add_library(flash_attention_cuda OBJECT
  flash_attention.cu
)

add_library(cpy_continue_cuda OBJECT
  cpy_continue.cu
)

add_library(cpy_transpose_cuda OBJECT
  cpy_transpose.cu
)

add_library(top_k_moe_cuda OBJECT
  top_k_moe.cu
)

add_library(roll_cuda OBJECT
  roll.cu
)

foreach(t gated_unary_cuda 
          rope_cuda q8_1_cuda
          flash_attention_cuda
          cpy_continue_cuda
          cpy_transpose_cuda
          top_k_moe_cuda
          roll_cuda)
  set_target_properties(${t} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(${t} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )

  # Debug 构建时 为每一个 xxx_cuda 对象编译debug相关路径
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(${t} PRIVATE MY_OPS_DEBUG=1)
  endif()
endforeach()

