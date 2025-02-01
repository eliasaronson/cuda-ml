cmake_minimum_required(VERSION 3.3)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cuda_ml LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

add_executable(main src/main.cc src/online_regression.cu)
# TODO: How to handle atomicAdd without building each arch?
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "75")

include(FetchContent)
# Also requires one of: libbfd (gnu binutils), libdwarf, libdw (elfutils)
FetchContent_Declare(backward
    GIT_REPOSITORY https://github.com/bombela/backward-cpp
    GIT_TAG master  # or a version tag, such as v1.6
    SYSTEM          # optional, the Backward include directory will be treated as system directory
)
FetchContent_MakeAvailable(backward)

add_compile_options(-g -G)
target_link_libraries(main PRIVATE CUDA::cublas CUDA::cusolver Backward::Interface -lbfd -ldl)
