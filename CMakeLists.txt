cmake_minimum_required(VERSION 3.5)
project(quac_video LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_definitions(JETSON_PLATFORM)

# Jetson Orin = sm_87, Xavier = sm_72
set(CMAKE_CUDA_ARCHITECTURES "72;87")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)


if(NOT DEFINED TENSORRT_DIR)
    # Try common TensorRT locations
    set(TENSORRT_SEARCH_PATHS
        "/usr/lib/x86_64-linux-gnu"          # Ubuntu x86 system install
        "/usr/lib/aarch64-linux-gnu"          # Jetson / aarch64 system
        "/usr/local/tensorrt"                 # Manual install
        "$ENV{TENSORRT_DIR}"                  # Environment variable
        "${CMAKE_SOURCE_DIR}/TensorRT"        # Local copy
    )
    foreach(path ${TENSORRT_SEARCH_PATHS})
        if(EXISTS "${path}/libnvinfer.so" OR EXISTS "${path}/lib/libnvinfer.so")
            set(TENSORRT_DIR "${path}")
            break()
        endif()
    endforeach()
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR}/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include
)

# Find TensorRT libraries
find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_DIR}/lib ${TENSORRT_DIR} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
)
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS ${TENSORRT_DIR}/lib ${TENSORRT_DIR} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
)

# CUDA Preprocessing Library

add_library(cuda_preprocessing STATIC
  YOLOs-CPP-TensorRT_clone/include/yolos/core/cuda_preprocessing.cu
)
target_include_directories(cuda_preprocessing PRIVATE
  YOLOs-CPP-TensorRT_clone/include
  ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(cuda_preprocessing PRIVATE
  CUDA::cudart
)
set_target_properties(cuda_preprocessing PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

#-----------------------------------------------
#----------------- application -----------------
#-----------------------------------------------

include_directories(
  ${GST_INCLUDE_DIRS}
  ${realsense2_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  include/
  YOLOs-CPP-TensorRT_clone/include
)

add_executable(app
  main.cpp
)

target_link_libraries(app
  ${GST_LIBRARIES}
  ${GST_APP_LIBRARIES}
  ${realsense2_LIBRARY}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}
  nvinfer
  CUDA::cudart
  cuda_preprocessing
)

if(NVONNXPARSER_LIB)
  target_link_libraries(app ${NVONNXPARSER_LIB})
endif()

#target_link_options(app PRIVATE
#  -Wl,--allow-shlib-undefined
#)
