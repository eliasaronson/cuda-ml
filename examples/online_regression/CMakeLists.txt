include(FetchContent)
# Also requires one of: libbfd (gnu binutils), libdwarf, libdw (elfutils)
FetchContent_Declare(backward
    GIT_REPOSITORY https://github.com/bombela/backward-cpp
    GIT_TAG master  # or a version tag, such as v1.6
    SYSTEM          # optional, the Backward include directory will be treated as system directory
)

FetchContent_MakeAvailable(backward)

add_executable(host_online_regression host_online_regression.cc)
target_link_libraries(host_online_regression cuda_ml Backward::Interface -lbfd -ldl)

# add_executable(device_online_regression device_online_regression.cu)
# target_link_libraries(device_online_regression cuda_ml Backward::Interface -lbfd -ldl)
