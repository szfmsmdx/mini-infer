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
    src/autograd/autograd.cpp
    src/autograd/cpu/rsqrt.cpp
    src/autograd/backward_nodes/elementwise.cpp
    src/autograd/backward_nodes/matmul.cpp
    src/autograd/backward_nodes/reduction.cpp
    src/autograd/backward_nodes/tensor_ops.cpp
    src/modules/linear.cpp
    src/modules/rmsnorm.cpp
    src/ops/activation.cpp
    src/ops/norm.cpp
    src/ops/layout.cpp
    src/ops/fill.cpp
    src/ops/copy.cpp
    src/ops/elementwise.cpp
    src/ops/matmul.cpp
    src/ops/reduction.cpp
    src/ops/cpu/copy.cpp
    src/ops/cpu/elementwise.cpp
    src/ops/cpu/sqrt.cpp
    src/ops/cpu/rsqrt.cpp
    src/ops/cpu/fill.cpp
    src/ops/cpu/matmul.cpp
    src/ops/cpu/reduction.cpp
    src/tensor/tensor.cpp
    src/tensor/check.cpp
    src/tensor/print.cpp
    src/utils/logger.cpp
    src/utils/memory.cpp
)

target_include_directories(mini_infer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(ENABLE_CUDA)
    target_sources(mini_infer PRIVATE
        src/autograd/cuda/rsqrt.cu
        src/ops/cuda/copy.cu
        src/ops/cuda/elementwise.cu
        src/ops/cuda/sqrt.cu
        src/ops/cuda/rsqrt.cu
        src/ops/cuda/fill.cu
        src/ops/cuda/matmul.cu
        src/ops/cuda/reduction.cu
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
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(BUILD_CUDA_BENCHMARKS "Build CUDA performance benchmarks" OFF)
if(BUILD_CUDA_BENCHMARKS AND NOT ENABLE_CUDA)
    message(FATAL_ERROR "BUILD_CUDA_BENCHMARKS requires a working CUDA compiler and Toolkit >= 12.4")
endif()
if(BUILD_BENCHMARKS OR BUILD_CUDA_BENCHMARKS)
    add_subdirectory(3rdparty)
    add_subdirectory(tests/benchmarks)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
