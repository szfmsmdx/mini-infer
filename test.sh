#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cuda_arch="${CUDA_ARCH:-89}"

case "$(uname -s)" in
    MINGW*|MSYS*)
        repo_dir="$(cygpath -m "$repo_dir")"
        generator="${CUDA_GENERATOR:-Visual Studio 17 2022}"
        generator_args=(-G "$generator")
        if [[ "$generator" == Visual\ Studio* ]]; then
            generator_args+=(-A x64)
        fi
        ;;
    *) generator_args=() ;;
esac

for tool in cmake ctest nvcc; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$tool" >&2
        exit 1
    fi
done

build_dir="$repo_dir/build/cuda-tests"
cmake -S "$repo_dir" -B "$build_dir" \
    "${generator_args[@]}" \
    -DMINI_INFER_ENABLE_CUDA=ON \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"

cmake --build "$build_dir" --config Release --parallel 2
printf '\nRunning CTest\n'
ctest --test-dir "$build_dir" -C Release --output-on-failure
