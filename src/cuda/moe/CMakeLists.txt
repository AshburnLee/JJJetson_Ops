add_library(moe_top_k_cuda OBJECT
  moe_top_k.cu
)

add_library(moe_dispatch_cuda OBJECT
  moe_dispatch.cu
)

add_library(moe_dispatch_sota_cuda OBJECT
  moe_dispatch_sota.cu
)

add_library(moe_combine_cuda OBJECT
  moe_combine.cu
)

add_library(moe_combine_sota_cuda OBJECT
  moe_combine_sota.cu
)

add_library(moe_pipeline_sota_cuda OBJECT
  moe_pipeline_sota.cu
)

add_library(moe_pipeline_cuda OBJECT
  moe_pipline.cu
)
target_link_libraries(moe_pipeline_cuda PRIVATE CUDA::cublas)

foreach(t moe_top_k_cuda
          moe_dispatch_cuda
          moe_dispatch_sota_cuda
          moe_combine_cuda
          moe_combine_sota_cuda
          moe_pipeline_sota_cuda
          moe_pipeline_cuda)
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
