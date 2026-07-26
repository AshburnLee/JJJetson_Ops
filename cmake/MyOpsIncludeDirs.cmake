# Shared (include/) + module headers colocated under src/{cuda,engine,model,...}
set(MY_OPS_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/src/cuda"
  "${CMAKE_SOURCE_DIR}/src/cuda/fa"
  "${CMAKE_SOURCE_DIR}/src/cuda/moe"
  "${CMAKE_SOURCE_DIR}/src/engine"
  "${CMAKE_SOURCE_DIR}/src/model"
)
