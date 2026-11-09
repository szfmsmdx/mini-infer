cmake_minimum_required(VERSION 3.18)
if(CMAKE_HOST_WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows-cuda.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows-cuda.cmake")
endif()
project(mini_infer LANGUAGES CXX)

set(ENABLE_CUDA OFF)
find_package(CUDAToolkit 12.4 QUIET)
if(CUDAToolkit_FOUND)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 80)
        endif()
        enable_language(CUDA)
        set(ENABLE_CUDA ON)
    endif()
endif()
if(ENABLE_CUDA)
    message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
else()
    message(STATUS "CUDA unavailable (requires Toolkit >= 12.4 and a working CUDA compiler); building CPU only")
endif()

add_library(mini_infer STATIC
    ops/elementwise.cpp
    ops/elementwise_cpu.cpp
    tensor/tensor.cpp
    tensor/tensor_cpu.cpp
    utils/logger.cpp
    utils/memory.cpp
)

target_include_directories(mini_infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_CUDA)
    target_sources(mini_infer PRIVATE
        ops/elementwise_cuda.cu
        tensor/tensor_cuda.cu
    )
    set_target_properties(mini_infer PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(mini_infer PUBLIC CUDA::cudart)
    target_compile_definitions(mini_infer PUBLIC ENABLE_CUDA=1)
endif()

target_compile_features(mini_infer PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(mini_infer PUBLIC
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
    )
endif()

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

    add_executable(elementwise_cpu_test tests/elementwise_cpu_test.cpp)
    target_link_libraries(elementwise_cpu_test PRIVATE mini_infer)
    add_test(NAME elementwise_cpu_test COMMAND elementwise_cpu_test)

    if(ENABLE_CUDA)
        add_executable(tensor_cuda_metadata_test tests/tensor_cuda_metadata_test.cpp)
        target_link_libraries(tensor_cuda_metadata_test PRIVATE mini_infer)
        add_test(NAME tensor_cuda_metadata_test COMMAND tensor_cuda_metadata_test)
        set_tests_properties(tensor_cuda_metadata_test PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(tensor_cuda_test tests/tensor_cuda_test.cpp)
        target_link_libraries(tensor_cuda_test PRIVATE mini_infer)
        add_test(NAME tensor_cuda_test COMMAND tensor_cuda_test)
        set_tests_properties(tensor_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(elementwise_cuda_test tests/elementwise_cuda_test.cpp)
        target_link_libraries(elementwise_cuda_test PRIVATE mini_infer)
        add_test(NAME elementwise_cuda_test COMMAND elementwise_cuda_test)
        set_tests_properties(elementwise_cuda_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
