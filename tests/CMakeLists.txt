add_executable(tokenizer_test tokenizer/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE mini_infer)
add_test(NAME tokenizer_test COMMAND tokenizer_test)

add_executable(bpe_tokenizer_test tokenizer/bpe_tokenizer_test.cpp)
target_link_libraries(bpe_tokenizer_test PRIVATE mini_infer)
add_test(NAME bpe_tokenizer_test COMMAND bpe_tokenizer_test)

add_executable(gpt_checkpoint_test models/gpt_checkpoint_test.cpp)
target_link_libraries(gpt_checkpoint_test PRIVATE mini_infer)
add_test(NAME gpt_checkpoint_test COMMAND gpt_checkpoint_test)

add_executable(transformer_cpu_test ops/transformer_test.cpp)
target_link_libraries(transformer_cpu_test PRIVATE mini_infer)
add_test(NAME transformer_cpu_test COMMAND transformer_cpu_test)

add_executable(attention_cpu_test ops/attention_test.cpp)
target_link_libraries(attention_cpu_test PRIVATE mini_infer)
add_test(NAME attention_cpu_test COMMAND attention_cpu_test)

add_executable(matmul_cpu_test ops/matmul_test.cpp)
target_link_libraries(matmul_cpu_test PRIVATE mini_infer)
add_test(NAME matmul_cpu_test COMMAND matmul_cpu_test)

add_executable(module_test modules/module_test.cpp)
target_link_libraries(module_test PRIVATE mini_infer)
add_test(NAME module_test COMMAND module_test)

add_executable(linear_cpu_test modules/linear_test.cpp)
target_link_libraries(linear_cpu_test PRIVATE mini_infer)
add_test(NAME linear_cpu_test COMMAND linear_cpu_test)

add_executable(weight_init_test init/weight_init_test.cpp)
target_link_libraries(weight_init_test PRIVATE mini_infer)
add_test(NAME weight_init_test COMMAND weight_init_test)

add_executable(optimizer_cpu_test optim/optimizer_test.cpp)
target_link_libraries(optimizer_cpu_test PRIVATE mini_infer)
add_test(NAME optimizer_cpu_test COMMAND optimizer_cpu_test)

add_executable(rmsnorm_cpu_test modules/rmsnorm_test.cpp)
target_link_libraries(rmsnorm_cpu_test PRIVATE mini_infer)
add_test(NAME rmsnorm_cpu_test COMMAND rmsnorm_cpu_test)

add_executable(reduction_cpu_test ops/reduction_cpu_test.cpp)
target_link_libraries(reduction_cpu_test PRIVATE mini_infer)
add_test(NAME reduction_cpu_test COMMAND reduction_cpu_test)

add_executable(layout_cpu_test tensor/layout_cpu_test.cpp)
target_link_libraries(layout_cpu_test PRIVATE mini_infer)
add_test(NAME layout_cpu_test COMMAND layout_cpu_test)

add_executable(scalar_test tensor/scalar_test.cpp)
target_link_libraries(scalar_test PRIVATE mini_infer)
target_include_directories(scalar_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME scalar_test COMMAND scalar_test)

add_executable(autograd_cpu_test autograd/autograd_cpu_test.cpp)
target_link_libraries(autograd_cpu_test PRIVATE mini_infer)
add_test(NAME autograd_cpu_test COMMAND autograd_cpu_test)

add_executable(logger_test utils/logger_test.cpp)
target_link_libraries(logger_test PRIVATE mini_infer)
target_include_directories(logger_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME logger_test COMMAND logger_test)

add_executable(dtype_encode_test tensor/dtype_encode_test.cpp)
target_link_libraries(dtype_encode_test PRIVATE mini_infer)
add_test(NAME dtype_encode_test COMMAND dtype_encode_test)

add_executable(tensor_cpu_test tensor/tensor_cpu_test.cpp)
target_link_libraries(tensor_cpu_test PRIVATE mini_infer)
add_test(NAME tensor_cpu_test COMMAND tensor_cpu_test)

add_executable(elementwise_cpu_test ops/elementwise_cpu_test.cpp)
target_link_libraries(elementwise_cpu_test PRIVATE mini_infer)
add_test(NAME elementwise_cpu_test COMMAND elementwise_cpu_test)

add_executable(sqrt_cpu_test ops/sqrt_test.cpp)
target_link_libraries(sqrt_cpu_test PRIVATE mini_infer)
add_test(NAME sqrt_cpu_test COMMAND sqrt_cpu_test)

add_executable(exp_cpu_test ops/exp_test.cpp)
target_link_libraries(exp_cpu_test PRIVATE mini_infer)
add_test(NAME exp_cpu_test COMMAND exp_cpu_test)

add_executable(log_cpu_test ops/log_test.cpp)
target_link_libraries(log_cpu_test PRIVATE mini_infer)
add_test(NAME log_cpu_test COMMAND log_cpu_test)

add_executable(activation_cpu_test ops/activation_test.cpp)
target_link_libraries(activation_cpu_test PRIVATE mini_infer)
add_test(NAME activation_cpu_test COMMAND activation_cpu_test)

add_executable(precision_cpu_test precision/precision_test.cpp)
target_link_libraries(precision_cpu_test PRIVATE mini_infer)

# Python is a test-only dependency. Prefer the documented local environment.
if(NOT Python3_EXECUTABLE)
    if(WIN32 AND EXISTS "${PROJECT_SOURCE_DIR}/build/torch-env/Scripts/python.exe")
        set(Python3_EXECUTABLE "${PROJECT_SOURCE_DIR}/build/torch-env/Scripts/python.exe")
    elseif(EXISTS "${PROJECT_SOURCE_DIR}/build/torch-env/bin/python")
        set(Python3_EXECUTABLE "${PROJECT_SOURCE_DIR}/build/torch-env/bin/python")
    endif()
endif()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME optimizer_precision_cpu_test
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/precision/compare_optimizer.py"
        --executable $<TARGET_FILE:optimizer_cpu_test> --device cpu
        --report-dir "${CMAKE_CURRENT_BINARY_DIR}/optimizer-cpu")
    add_test(NAME precision_cpu_test
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/precision/compare_torch.py"
        --executable $<TARGET_FILE:precision_cpu_test> --device cpu
        --report-dir "${CMAKE_CURRENT_BINARY_DIR}/torch-cpu")
    add_test(NAME torch_comparator_test
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/precision/test_compare_torch.py")
    add_test(NAME rmsnorm_precision_cpu_test
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/precision/compare_rmsnorm.py"
        --executable $<TARGET_FILE:rmsnorm_cpu_test> --device cpu
        --report-dir "${CMAKE_CURRENT_BINARY_DIR}/rmsnorm-cpu")
    set_tests_properties(rmsnorm_precision_cpu_test PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME rmsnorm_reference_test
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/precision/test_compare_rmsnorm.py")
else()
    message(WARNING "Python unavailable: PyTorch precision tests will be SKIPPED. See tests/README.md.")
    add_test(NAME precision_cpu_test COMMAND precision_cpu_test)
endif()
set_tests_properties(precision_cpu_test PROPERTIES SKIP_RETURN_CODE 77)

if(ENABLE_CUDA)
    add_executable(transformer_cuda_test ops/transformer_test.cpp)
    target_compile_definitions(transformer_cuda_test PRIVATE TRANSFORMER_TEST_CUDA)
    target_link_libraries(transformer_cuda_test PRIVATE mini_infer)
    add_test(NAME transformer_cuda_test COMMAND transformer_cuda_test)
    set_tests_properties(transformer_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(attention_cuda_test ops/attention_test.cpp)
    target_compile_definitions(attention_cuda_test PRIVATE ATTENTION_TEST_CUDA)
    target_link_libraries(attention_cuda_test PRIVATE mini_infer)
    add_test(NAME attention_cuda_test COMMAND attention_cuda_test)
    set_tests_properties(attention_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(optimizer_cuda_test optim/optimizer_test.cpp)
    target_compile_definitions(optimizer_cuda_test PRIVATE OPTIMIZER_TEST_CUDA)
    target_link_libraries(optimizer_cuda_test PRIVATE mini_infer)
    add_test(NAME optimizer_cuda_test COMMAND optimizer_cuda_test)
    set_tests_properties(optimizer_cuda_test PROPERTIES SKIP_RETURN_CODE 77)
    if(Python3_Interpreter_FOUND)
        add_test(NAME optimizer_precision_cuda_test
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/precision/compare_optimizer.py"
            --executable $<TARGET_FILE:optimizer_cuda_test> --device cuda
            --report-dir "${CMAKE_CURRENT_BINARY_DIR}/optimizer-cuda")
        set_tests_properties(optimizer_precision_cuda_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    add_executable(rmsnorm_cuda_test modules/rmsnorm_test.cpp)
    target_compile_definitions(rmsnorm_cuda_test PRIVATE RMSNORM_TEST_CUDA)
    target_link_libraries(rmsnorm_cuda_test PRIVATE mini_infer)
    add_test(NAME rmsnorm_cuda_test COMMAND rmsnorm_cuda_test)
    set_tests_properties(rmsnorm_cuda_test PROPERTIES SKIP_RETURN_CODE 77)
    if(Python3_Interpreter_FOUND)
        add_test(NAME rmsnorm_precision_cuda_test
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/precision/compare_rmsnorm.py"
            --executable $<TARGET_FILE:rmsnorm_cuda_test> --device cuda
            --report-dir "${CMAKE_CURRENT_BINARY_DIR}/rmsnorm-cuda")
        set_tests_properties(rmsnorm_precision_cuda_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(precision_cuda_test precision/precision_test.cpp)
    target_compile_definitions(precision_cuda_test PRIVATE PRECISION_TEST_CUDA)
    target_link_libraries(precision_cuda_test PRIVATE mini_infer)
    if(Python3_Interpreter_FOUND)
        add_test(NAME precision_cuda_test
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/precision/compare_torch.py"
            --executable $<TARGET_FILE:precision_cuda_test> --device cuda
            --report-dir "${CMAKE_CURRENT_BINARY_DIR}/torch-cuda")
    else()
        add_test(NAME precision_cuda_test COMMAND precision_cuda_test)
    endif()
    set_tests_properties(precision_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(activation_cuda_test ops/activation_test.cpp)
    target_compile_definitions(activation_cuda_test PRIVATE ACTIVATION_TEST_CUDA)
    target_link_libraries(activation_cuda_test PRIVATE mini_infer)
    add_test(NAME activation_cuda_test COMMAND activation_cuda_test)
    set_tests_properties(activation_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(exp_cuda_test ops/exp_test.cpp)
    target_compile_definitions(exp_cuda_test PRIVATE EXP_TEST_CUDA)
    target_link_libraries(exp_cuda_test PRIVATE mini_infer)
    add_test(NAME exp_cuda_test COMMAND exp_cuda_test)
    set_tests_properties(exp_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(log_cuda_test ops/log_test.cpp)
    target_compile_definitions(log_cuda_test PRIVATE LOG_TEST_CUDA)
    target_link_libraries(log_cuda_test PRIVATE mini_infer)
    add_test(NAME log_cuda_test COMMAND log_cuda_test)
    set_tests_properties(log_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(sqrt_cuda_test ops/sqrt_test.cpp)
    target_compile_definitions(sqrt_cuda_test PRIVATE SQRT_TEST_CUDA)
    target_link_libraries(sqrt_cuda_test PRIVATE mini_infer)
    add_test(NAME sqrt_cuda_test COMMAND sqrt_cuda_test)
    set_tests_properties(sqrt_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(linear_cuda_test modules/linear_test.cpp)
    target_compile_definitions(linear_cuda_test PRIVATE LINEAR_TEST_CUDA)
    target_link_libraries(linear_cuda_test PRIVATE mini_infer)
    add_test(NAME linear_cuda_test COMMAND linear_cuda_test)
    set_tests_properties(linear_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(matmul_cuda_test ops/matmul_test.cpp)
    target_compile_definitions(matmul_cuda_test PRIVATE MATMUL_TEST_CUDA)
    target_link_libraries(matmul_cuda_test PRIVATE mini_infer)
    add_test(NAME matmul_cuda_test COMMAND matmul_cuda_test)
    set_tests_properties(matmul_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(reduction_cuda_test ops/reduction_cuda_test.cpp)
    target_link_libraries(reduction_cuda_test PRIVATE mini_infer)
    add_test(NAME reduction_cuda_test COMMAND reduction_cuda_test)
    set_tests_properties(reduction_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(autograd_cuda_test autograd/autograd_cuda_test.cpp)
    target_link_libraries(autograd_cuda_test PRIVATE mini_infer)
    target_include_directories(autograd_cuda_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME autograd_cuda_test COMMAND autograd_cuda_test)
    set_tests_properties(autograd_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(tensor_cuda_metadata_test tensor/tensor_cuda_metadata_test.cpp)
    target_link_libraries(tensor_cuda_metadata_test PRIVATE mini_infer)
    add_test(NAME tensor_cuda_metadata_test COMMAND tensor_cuda_metadata_test)
    set_tests_properties(tensor_cuda_metadata_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(tensor_cuda_test tensor/tensor_cuda_test.cpp)
    target_link_libraries(tensor_cuda_test PRIVATE mini_infer)
    target_include_directories(tensor_cuda_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME tensor_cuda_test COMMAND tensor_cuda_test)
    set_tests_properties(tensor_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(elementwise_cuda_test ops/elementwise_cuda_test.cpp)
    target_link_libraries(elementwise_cuda_test PRIVATE mini_infer)
    add_test(NAME elementwise_cuda_test COMMAND elementwise_cuda_test)
    set_tests_properties(elementwise_cuda_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
