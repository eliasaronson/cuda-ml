cmake_minimum_required(VERSION 3.3)

option(BUILD_EXAMPLES "Build examples" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Line numbers and device debug info for debugging
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
endif()

project(cuda_ml LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(public_headers src/online_regression.h src/utils.cuh)
add_library(cuda_ml src/online_regression.cu ${public_headers})
# TODO: How to handle atomicAdd without building each arch?
set_target_properties(cuda_ml PROPERTIES CUDA_ARCHITECTURES "75")

target_include_directories(cuda_ml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(cuda_ml PRIVATE CUDA::cublas CUDA::cusolver)

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
