#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cuda_arch="${CUDA_ARCH:-80}"

case "$(uname -s)" in
    MINGW*|MSYS*)
        repo_dir="$(cygpath -m "$repo_dir")"
        ;;
esac

for tool in cmake ctest; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$tool" >&2
        exit 1
    fi
done

build_dir="$repo_dir/build/tests"
cmake -S "$repo_dir" -B "$build_dir" \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"

cmake --build "$build_dir" --config Release --parallel 2
printf '\nRunning CTest\n'
ctest --test-dir "$build_dir" -C Release --output-on-failure
