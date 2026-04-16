#
# Top-level CMake for my-ops
# - 目标: 统一管理所有 CUDA kernel 和 Python 扩展模块的构建
# - 用途: 配置 C++/CUDA 环境, 找到 Python3/pybind11, 并把构建入口集中到 src/cuda 与 src/bindings
# - 关键逻辑:
#   * 启用 CXX/CUDA 17, 设置默认构建类型为 Release
#   * 定义 MY_OPS_PYTHON_OUTPUT_DIR, 统一所有 .so 输出到 python/ 目录
#   * add_subdirectory(src/cuda) 让 CMake 进入子目录，读取它们的 CMakeLists.txt 并生成对应目标/规则。生成各个 OBJECT CUDA target
#   * add_subdirectory(src/bindings) 同上，生成对应的 Python MODULE (.so)
#
cmake_minimum_required(VERSION 3.22)
project(my_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

# 显示 GPU 信息的工具
add_executable(show_device_info tools/show_device_info.cu)
target_include_directories(show_device_info PRIVATE "${CMAKE_SOURCE_DIR}/include")
set_property(TARGET show_device_info PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# 统一 Python 扩展输出目录
set(MY_OPS_PYTHON_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python")
file(MAKE_DIRECTORY "${MY_OPS_PYTHON_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${MY_OPS_PYTHON_OUTPUT_DIR}")

add_subdirectory(src/cuda)
add_subdirectory(src/bindings)

