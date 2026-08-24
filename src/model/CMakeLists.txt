add_library(rope_model OBJECT
  rope_cossin_cache.cpp
)

add_library(kv_cache_model OBJECT
  kv_cache.cpp
)

add_library(weight_loader_model OBJECT
  model_config.cpp
  hf_llama_weight_map.cpp
  safetensors_reader.cpp
  weight_loader.cpp
)

add_library(transformer_model OBJECT
  transformer_model.cpp
)

foreach(t rope_model kv_cache_model weight_loader_model transformer_model)
  target_link_libraries(${t} PRIVATE CUDA::cudart)
  set_target_properties(${t} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${t} PRIVATE ${MY_OPS_INCLUDE_DIRS})
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(${t} PRIVATE MY_OPS_DEBUG=1)
  endif()
endforeach()

target_link_libraries(transformer_model PRIVATE rope_model)
