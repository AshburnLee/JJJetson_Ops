#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

BUILD_TYPE="Release"
if [[ "${1:-}" == "--debug" ]]; then
  BUILD_TYPE="Debug"
fi
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" ..
make -j"$(( $(nproc) - 2 ))"

echo "Done building. Python extensions are in: ${ROOT_DIR}/python"
