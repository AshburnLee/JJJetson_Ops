#
# src/cuda/CMakeLists.txt
# - 目标: 为每个 CUDA 算子 (.cu) 定义一个 OBJECT 库
# - 用途: 只负责把 CUDA 源代码编译成目标文件, 由 bindings 层的 MODULE 目标复用
# - 关键逻辑:
#   * 统一设置 CUDA 标准、架构、-fPIC
#   * 为每个算子 (gated_unary/rope/q8_1/fa_*.cu/...) 建立 OBJECT target
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

add_library(rope_search_table_cuda OBJECT
  rope_neox_search_table.cu
)

add_library(rope_global_cache_cuda OBJECT
  rope_neox_global_cache.cu
)

add_library(q8_1_cuda OBJECT
  q8_1.cu
)

add_library(mm_q4_0_q8_1_cuda OBJECT
  mm_q4_0_q8_1.cu
)

add_subdirectory(fa)
add_subdirectory(moe)

add_library(cpy_continue_cuda OBJECT
  cpy_continue.cu
)

add_library(cpy_transpose_cuda OBJECT
  cpy_transpose.cu
)

add_library(roll_cuda OBJECT
  roll.cu
)

add_library(rms_norm_cuda OBJECT
  rms_norm.cu
)

add_library(layer_norm_cuda OBJECT
  layer_norm.cu
)

add_library(rms_norm_fused_add_cuda OBJECT
  rms_norm_fused_add.cu
)

add_library(linear_cuda OBJECT
  linear.cu
)

add_library(kv_cache_cuda OBJECT
  kv_cache.cu
)

add_library(swiglu_cuda OBJECT
  swiglu.cu
)

foreach(t gated_unary_cuda
          rope_cuda rope_search_table_cuda rope_global_cache_cuda q8_1_cuda mm_q4_0_q8_1_cuda
          cpy_continue_cuda
          cpy_transpose_cuda
          roll_cuda
          rms_norm_cuda
          layer_norm_cuda
          rms_norm_fused_add_cuda
          linear_cuda
          kv_cache_cuda
          swiglu_cuda)
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

target_link_libraries(linear_cuda PRIVATE CUDA::cublas)
