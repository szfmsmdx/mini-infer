add_executable(matmul_cpu_test matmul_test.cpp)
target_link_libraries(matmul_cpu_test PRIVATE mini_infer)
add_test(NAME matmul_cpu_test COMMAND matmul_cpu_test)

add_executable(reduction_cpu_test reduction_cpu_test.cpp)
target_link_libraries(reduction_cpu_test PRIVATE mini_infer)
add_test(NAME reduction_cpu_test COMMAND reduction_cpu_test)

add_executable(layout_cpu_test layout_cpu_test.cpp)
target_link_libraries(layout_cpu_test PRIVATE mini_infer)
add_test(NAME layout_cpu_test COMMAND layout_cpu_test)

add_executable(scalar_test scalar_test.cpp)
target_link_libraries(scalar_test PRIVATE mini_infer)
target_include_directories(scalar_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME scalar_test COMMAND scalar_test)

add_executable(autograd_cpu_test autograd_cpu_test.cpp)
target_link_libraries(autograd_cpu_test PRIVATE mini_infer)
add_test(NAME autograd_cpu_test COMMAND autograd_cpu_test)

add_executable(logger_test logger_test.cpp)
target_link_libraries(logger_test PRIVATE mini_infer)
target_include_directories(logger_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME logger_test COMMAND logger_test)

add_executable(dtype_encode_test dtype_encode_test.cpp)
target_link_libraries(dtype_encode_test PRIVATE mini_infer)
add_test(NAME dtype_encode_test COMMAND dtype_encode_test)

add_executable(tensor_cpu_test tensor_cpu_test.cpp)
target_link_libraries(tensor_cpu_test PRIVATE mini_infer)
add_test(NAME tensor_cpu_test COMMAND tensor_cpu_test)

add_executable(elementwise_cpu_test elementwise_cpu_test.cpp)
target_link_libraries(elementwise_cpu_test PRIVATE mini_infer)
add_test(NAME elementwise_cpu_test COMMAND elementwise_cpu_test)

if(ENABLE_CUDA)
    add_executable(matmul_cuda_test matmul_test.cpp)
    target_compile_definitions(matmul_cuda_test PRIVATE MATMUL_TEST_CUDA)
    target_link_libraries(matmul_cuda_test PRIVATE mini_infer)
    add_test(NAME matmul_cuda_test COMMAND matmul_cuda_test)
    set_tests_properties(matmul_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(reduction_cuda_test reduction_cuda_test.cpp)
    target_link_libraries(reduction_cuda_test PRIVATE mini_infer)
    add_test(NAME reduction_cuda_test COMMAND reduction_cuda_test)
    set_tests_properties(reduction_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(autograd_cuda_test autograd_cuda_test.cpp)
    target_link_libraries(autograd_cuda_test PRIVATE mini_infer)
    target_include_directories(autograd_cuda_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME autograd_cuda_test COMMAND autograd_cuda_test)
    set_tests_properties(autograd_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(tensor_cuda_metadata_test tensor_cuda_metadata_test.cpp)
    target_link_libraries(tensor_cuda_metadata_test PRIVATE mini_infer)
    add_test(NAME tensor_cuda_metadata_test COMMAND tensor_cuda_metadata_test)
    set_tests_properties(tensor_cuda_metadata_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(tensor_cuda_test tensor_cuda_test.cpp)
    target_link_libraries(tensor_cuda_test PRIVATE mini_infer)
    target_include_directories(tensor_cuda_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME tensor_cuda_test COMMAND tensor_cuda_test)
    set_tests_properties(tensor_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(elementwise_cuda_test elementwise_cuda_test.cpp)
    target_link_libraries(elementwise_cuda_test PRIVATE mini_infer)
    add_test(NAME elementwise_cuda_test COMMAND elementwise_cuda_test)
    set_tests_properties(elementwise_cuda_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
