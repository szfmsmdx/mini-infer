if(BUILD_BENCHMARKS)
    add_executable(matmul_cpu_bench matmul_cpu_bench.cpp)
    target_link_libraries(matmul_cpu_bench PRIVATE mini_infer ggml)
endif()

if(BUILD_CUDA_BENCHMARKS)
    add_executable(matmul_cuda_bench matmul_cuda_bench.cpp)
    target_link_libraries(matmul_cuda_bench PRIVATE mini_infer ggml)
endif()
