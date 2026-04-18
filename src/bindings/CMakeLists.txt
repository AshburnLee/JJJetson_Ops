#
# src/bindings/CMakeLists.txt
# - 目标: 为每个 Python 扩展 ( *_me ) 定义 MODULE 库
# - 用途: 将对应的 binding.cpp 与 CUDA OBJECT 库链接, 生成可被 Python import 的 .so
# - 关键逻辑:
#   * include_directories 引入 Python3 与公共 include/
#   * set_python_module_props 统一去掉 lib 前缀, 并把 .so 输出到顶层定义的 python/ 目录
#   * 对每个算子 (gated_unary/rope/q8_1/fa/cpy_*/top_k_moe/roll) 建立 MODULE target,
#     链接到各自的 CUDA OBJECT target 与 pybind11/Python/CUDA 运行库
#
include_directories(
  ${Python3_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
)

function(set_python_module_props target_name)
  set_target_properties(${target_name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MY_OPS_PYTHON_OUTPUT_DIR}"
    PREFIX ""
  )
endfunction()

add_library(gated_unary_me MODULE
  gated_unary_binding.cpp
  $<TARGET_OBJECTS:gated_unary_cuda>
)

target_link_libraries(gated_unary_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(gated_unary_me)

add_library(rope_me MODULE
  rope_neox_binding.cpp
  $<TARGET_OBJECTS:rope_cuda>
)

target_link_libraries(rope_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(rope_me)

add_library(q8_1_me MODULE
  q8_1_binding.cpp
  $<TARGET_OBJECTS:q8_1_cuda>
)

target_link_libraries(q8_1_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(q8_1_me)

add_library(fa_me MODULE
  fa_binding.cpp
  $<TARGET_OBJECTS:fa_two_pass_cuda>
  $<TARGET_OBJECTS:fa_one_pass_cuda>
  $<TARGET_OBJECTS:fa_one_pass_parallel_cuda>
)

target_link_libraries(fa_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(fa_me)

add_library(fa_tc_me MODULE
  fa_tc_binding.cpp
  $<TARGET_OBJECTS:fa_one_pass_parallel_tc_cuda>
  $<TARGET_OBJECTS:fa_one_pass_parallel_tc_true_cuda>
  $<TARGET_OBJECTS:fa_one_pass_parallel_tc_true_64_cuda>
)

target_link_libraries(fa_tc_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(fa_tc_me)

add_library(cpy_continue_me MODULE
  cpy_continue_binding.cpp
  $<TARGET_OBJECTS:cpy_continue_cuda>
)

target_link_libraries(cpy_continue_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(cpy_continue_me)

add_library(cpy_transpose_me MODULE
  cpy_transpose_binding.cpp
  $<TARGET_OBJECTS:cpy_transpose_cuda>
)

target_link_libraries(cpy_transpose_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(cpy_transpose_me)

add_library(top_k_moe_me MODULE
  top_k_moe_binding.cpp
  $<TARGET_OBJECTS:top_k_moe_cuda>
)

target_link_libraries(top_k_moe_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(top_k_moe_me)

add_library(roll_me MODULE
  roll_binding.cpp
  $<TARGET_OBJECTS:roll_cuda>
)

target_link_libraries(roll_me PRIVATE
  pybind11::module
  ${Python3_LIBRARIES}
  cuda
  cudart
)

set_python_module_props(roll_me)

# Debug 构建时：对所有 *_me 模块统一开启可选调试/可观测性代码路径
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  foreach(t gated_unary_me
            rope_me q8_1_me fa_me fa_tc_me
            cpy_continue_me cpy_transpose_me
            top_k_moe_me
            roll_me)
    target_compile_definitions(${t} PRIVATE MY_OPS_DEBUG=1)
  endforeach()
endif()
