cmake_minimum_required(VERSION 3.17)
project(mini_infer LANGUAGES CXX)

option(MINI_INFER_ENABLE_CUDA "Enable CUDA support" OFF)
if(MINI_INFER_ENABLE_CUDA)
    find_package(CUDAToolkit 12.4 REQUIRED)
    enable_language(CUDA)
endif()

add_library(mini_infer STATIC
    tensor/tensor.cpp
    utils/logger.cpp
    utils/memory.cpp
)

target_include_directories(mini_infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(MINI_INFER_ENABLE_CUDA)
    target_sources(mini_infer PRIVATE tensor/tensor_cuda.cu)
    set_target_properties(mini_infer PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(mini_infer PUBLIC CUDA::cudart)
    target_compile_definitions(mini_infer PUBLIC MINI_INFER_ENABLE_CUDA=1)
endif()

target_compile_features(mini_infer PUBLIC cxx_std_17)

include(CTest)
if(BUILD_TESTING)
    add_executable(logger_test tests/logger_test.cpp)
    target_link_libraries(logger_test PRIVATE mini_infer)
    add_test(NAME logger_test COMMAND logger_test)

    add_executable(dtype_encode_test tests/dtype_encode_test.cpp)
    target_link_libraries(dtype_encode_test PRIVATE mini_infer)
    add_test(NAME dtype_encode_test COMMAND dtype_encode_test)

    add_executable(tensor_cpu_test tests/tensor_cpu_test.cpp)
    target_link_libraries(tensor_cpu_test PRIVATE mini_infer)
    add_test(NAME tensor_cpu_test COMMAND tensor_cpu_test)

    if(MINI_INFER_ENABLE_CUDA)
        add_executable(tensor_cuda_test tests/tensor_cuda_test.cpp)
        target_link_libraries(tensor_cuda_test PRIVATE mini_infer)
        add_test(NAME tensor_cuda_test COMMAND tensor_cuda_test)
    endif()
endif()
