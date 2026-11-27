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
    src/backend/dispatch.cpp
    src/backend/cpu/register.cpp
    src/backend/cpu/embedding.cpp
    src/backend/cpu/rope.cpp
    src/backend/cpu/cross_entropy.cpp
    src/backend/cpu/causal_attention.cpp
    src/backend/cpu/norm.cpp
    src/autograd/autograd.cpp
    src/autograd/tensor_autograd.cpp
    src/ops/activation/activation.cpp
    src/backend/cpu/activation.cpp
    src/ops/activation/softmax.cpp
    src/ops/elementwise/elementwise.cpp
    src/ops/matmul/matmul.cpp
    src/ops/norm/norm.cpp
    src/ops/reduction/reduction.cpp
    src/ops/layout/layout.cpp
    src/init/weight_init.cpp
    src/modules/linear.cpp
    src/modules/rmsnorm.cpp
    src/modules/embedding.cpp
    src/models/gpt.cpp
    src/models/gpt_checkpoint.cpp
    src/models/layers/simple_decoder_layer.cpp
    src/ops/attention/causal_attention.cpp
    src/ops/embedding/embedding.cpp
    src/ops/loss_fn/cross_entropy.cpp
    src/ops/position_embedding/rope.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/optim/optimizer.cpp
    src/optim/sgd.cpp
    src/optim/adam.cpp
    src/ops/fill/fill.cpp
    src/ops/copy/copy.cpp
    src/backend/cpu/copy.cpp
    src/backend/cpu/elementwise.cpp
    src/backend/cpu/fill.cpp
    src/backend/cpu/matmul.cpp
    src/backend/cpu/reduction.cpp
    src/tensor/tensor.cpp
    src/tensor/check.cpp
    src/tensor/print.cpp
    src/utils/logger.cpp
    src/runtime/memory.cpp
    src/runtime/device_context.cpp
)

target_include_directories(mini_infer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(ENABLE_CUDA)
    target_sources(mini_infer PRIVATE
        src/backend/cuda/register.cu
        src/backend/common/cuda_layout.cpp
        src/backend/cuda/activation.cu
        src/backend/cuda/elementwise.cu
        src/backend/cuda/norm.cu
        src/backend/cuda/copy.cu
        src/backend/cuda/fill.cu
        src/backend/cuda/matmul.cu
        src/backend/cuda/reduction.cu
        src/backend/cuda/rope.cu
        src/backend/cuda/causal_attention.cu
        src/backend/cuda/cross_entropy.cu
        src/backend/cuda/embedding.cu
    )
    set_target_properties(mini_infer PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(mini_infer PUBLIC CUDA::cudart)
    target_compile_definitions(mini_infer PUBLIC ENABLE_CUDA=1)
endif()

target_compile_features(mini_infer PUBLIC cxx_std_17)
add_executable(train_gpt EXCLUDE_FROM_ALL examples/nanogpt/train_gpt.cpp)
target_include_directories(train_gpt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(train_gpt PRIVATE mini_infer)
add_executable(chat_gpt EXCLUDE_FROM_ALL examples/nanogpt/chat_gpt.cpp)
target_link_libraries(chat_gpt PRIVATE mini_infer)
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
