add_library(moe_engine OBJECT
  moe_cuda_graph_instance_container.cpp
  moe_runner.cpp
)

target_link_libraries(moe_engine PRIVATE CUDA::cudart)

add_library(transformer_engine OBJECT
  transformer_runner.cpp
)

target_link_libraries(transformer_engine PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(moe_engine transformer_engine PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(moe_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(transformer_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  target_compile_definitions(moe_engine PRIVATE MY_OPS_DEBUG=1)
  target_compile_definitions(transformer_engine PRIVATE MY_OPS_DEBUG=1)
endif()
