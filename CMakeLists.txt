cmake_minimum_required(VERSION 3.16)
project(mini_infer LANGUAGES CXX)

# TODO: 接入 CUDA 时增加可选构建开关，启用 CUDA 语言、链接 Runtime 并添加 .cu 源文件。

add_library(mini_infer STATIC
    tensor/tensor.cpp
    utils/logger.cpp
    utils/memory.cpp
)

target_include_directories(mini_infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(mini_infer PUBLIC cxx_std_17)

include(CTest)
if(BUILD_TESTING)
    add_executable(dtype_encode_test tests/dtype_encode_test.cpp)
    target_link_libraries(dtype_encode_test PRIVATE mini_infer)
    add_test(NAME dtype_encode_test COMMAND dtype_encode_test)

    add_executable(tensor_cpu_test tests/tensor_cpu_test.cpp)
    target_link_libraries(tensor_cpu_test PRIVATE mini_infer)
    add_test(NAME tensor_cpu_test COMMAND tensor_cpu_test)
endif()
