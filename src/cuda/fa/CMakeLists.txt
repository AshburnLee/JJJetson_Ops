#
# src/cuda/fa/CMakeLists.txt
# - Flash-attention style kernels (fa_*.cu)
#
add_library(fa_two_pass_cuda OBJECT
  fa_two_pass.cu
)

add_library(fa_one_pass_cuda OBJECT
  fa_one_pass.cu
)

add_library(fa_one_pass_parallel_cuda OBJECT
  fa_one_pass_parallel.cu
)

add_library(fa_one_pass_parallel_tc_cuda OBJECT
  fa_one_pass_parallel_tc.cu
)

add_library(fa_one_pass_parallel_tc_true_cuda OBJECT
  fa_one_pass_parallel_tc_true.cu
)

add_library(fa_double_buffer_cuda OBJECT
  fa_double_buffer.cu
)

foreach(t fa_two_pass_cuda
          fa_one_pass_cuda
          fa_one_pass_parallel_cuda
          fa_one_pass_parallel_tc_cuda
          fa_one_pass_parallel_tc_true_cuda
          fa_double_buffer_cuda)
  set_target_properties(${t} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(${t} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )

  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(${t} PRIVATE MY_OPS_DEBUG=1)
  endif()
endforeach()
