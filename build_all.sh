#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

BUILD_TYPE="Release"
ENABLE_NVTX="OFF"
for arg in "$@"; do
  case "${arg}" in
    --debug) BUILD_TYPE="Debug" ;;
    --nvtx) ENABLE_NVTX="ON" ;;
    *)
      echo "ERROR: unknown option: ${arg}" >&2
      echo "Usage: $0 [--debug] [--nvtx]" >&2
      exit 1
      ;;
  esac
done
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DENABLE_NVTX="${ENABLE_NVTX}" ..
make -j"$(( $(nproc) - 2 ))"

echo "Done building. Python extensions are in: ${ROOT_DIR}/python"
