add_library(rope_model OBJECT
  rope_cossin_cache.cpp
)

target_link_libraries(rope_model PRIVATE CUDA::cudart)

set_target_properties(rope_model PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(rope_model PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  target_compile_definitions(rope_model PRIVATE MY_OPS_DEBUG=1)
endif()
