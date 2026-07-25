add_library(rope_model OBJECT
  rope_cossin_cache.cpp
)

add_library(kv_cache_model OBJECT
  kv_cache.cpp
)

foreach(t rope_model kv_cache_model)
  target_link_libraries(${t} PRIVATE CUDA::cudart)
  set_target_properties(${t} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(${t} PRIVATE MY_OPS_DEBUG=1)
  endif()
endforeach()
