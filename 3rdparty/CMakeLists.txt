include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(BUILD_SHARED_LIBS OFF)
set(GGML_BUILD_TESTS OFF)
set(GGML_BUILD_EXAMPLES OFF)
set(GGML_METAL OFF)
if(BUILD_CUDA_BENCHMARKS)
    set(GGML_CUDA ON)
    # Match the default set by llama.cpp when adding its ggml subdirectory.
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()
set(LLAMA_CPP_REVISION 401563f10ec60754e688cd3165c1343cf4be9642)
set(LLAMA_CPP_REVISION ${LLAMA_CPP_REVISION} PARENT_SCOPE)
if(NOT FETCHCONTENT_SOURCE_DIR_LLAMA_CPP)
    # The archive has no .git directory from which ggml can read its revision.
    set(GIT_EXE "")
    set(GGML_BUILD_COMMIT ${LLAMA_CPP_REVISION})
endif()

FetchContent_Declare(llama_cpp
    URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${LLAMA_CPP_REVISION}
    SOURCE_SUBDIR ggml
)
FetchContent_MakeAvailable(llama_cpp)
