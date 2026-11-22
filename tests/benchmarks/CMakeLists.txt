if(BUILD_BENCHMARKS)
    add_executable(matmul_cpu_bench matmul_cpu_bench.cpp)
    target_include_directories(matmul_cpu_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(matmul_cpu_bench PRIVATE mini_infer ggml)
    add_executable(ops_cpu_bench ops_bench.cpp)
    target_link_libraries(ops_cpu_bench PRIVATE mini_infer ggml)
endif()

if(BUILD_CUDA_BENCHMARKS)
    add_executable(matmul_cuda_bench matmul_cuda_bench.cpp)
    target_include_directories(matmul_cuda_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(matmul_cuda_bench PRIVATE mini_infer ggml)
    add_executable(ops_cuda_bench ops_bench.cpp)
    target_include_directories(ops_cuda_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_definitions(ops_cuda_bench PRIVATE OPS_BENCH_CUDA)
    target_link_libraries(ops_cuda_bench PRIVATE mini_infer ggml)
endif()

execute_process(COMMAND git describe --always --dirty
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE BENCH_MINI_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT BENCH_MINI_REVISION)
    set(BENCH_MINI_REVISION unknown)
endif()
foreach(target ops_cpu_bench ops_cuda_bench)
    if(TARGET ${target})
        target_compile_definitions(${target} PRIVATE
            BENCH_GGML_REVISION="${LLAMA_CPP_REVISION}"
            BENCH_MINI_REVISION="${BENCH_MINI_REVISION}"
            BENCH_BUILD_CONFIG="$<CONFIG>")
    endif()
endforeach()
