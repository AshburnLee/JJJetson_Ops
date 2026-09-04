#!/usr/bin/env bash
#
# run_tests.sh — 跑 JJJetson_Ops/tests 下的 Python 单测
#
# 前置：已 build（python/*_me.so）；建议 conda activate cuda-ops
# 脚本会自动设置 PYTHONPATH=python:py:tests
#   python/ — CMake 输出的 *_me.so（gitignore）
#   py/     — 可提交的纯 Python 正式入口（如 hf_tokenizer）
#   tests/  — 单测
#
# -----------------------------------------------------------------------
# 用法
# -----------------------------------------------------------------------
#
#   ./run_tests.sh
#       默认：跑 tests/ 下所有 test_*.py（每个文件走其 if __name__ == "__main__"）
#
#   ./run_tests.sh --suite FILE
#       只跑一个测试文件里的全部 case
#       FILE 可以是 test_weight_loader.py，或 tests/test_weight_loader.py
#
#   ./run_tests.sh --suite FILE --case NAME
#       只跑该文件里的某一个 test_* 函数（不跑 __main__ 里列出的其它 case）
#       NAME 为函数名，例如 test_hf_llama_safetensors_name_map
#       注意：--case 必须和 --suite 一起用
#
#   ./run_tests.sh --help
#       打印简要帮助
#
# -----------------------------------------------------------------------
# 示例
# -----------------------------------------------------------------------
#
#   ./run_tests.sh
#   ./run_tests.sh --suite test_weight_loader.py
#   ./run_tests.sh --suite tests/test_weight_loader.py --case test_hf_llama_safetensors_name_map
#
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}/py:${ROOT_DIR}/tests:${PYTHONPATH-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

  (no options)          Run all tests/test_*.py (default)
  --suite FILE          Run one test file (e.g. test_weight_loader.py or tests/test_weight_loader.py)
  --case NAME           With --suite: run only test function NAME in that file
  -h, --help            Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --suite test_weight_loader.py
  $(basename "$0") --suite tests/test_weight_loader.py --case test_hf_llama_safetensors_name_map
EOF
}

SUITE=""
CASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="${2:-}"
      if [[ -z "${SUITE}" ]]; then
        echo "ERROR: --suite requires a test file path" >&2
        exit 1
      fi
      shift 2
      ;;
    --case)
      CASE="${2:-}"
      if [[ -z "${CASE}" ]]; then
        echo "ERROR: --case requires a test function name" >&2
        exit 1
      fi
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${CASE}" && -z "${SUITE}" ]]; then
  echo "ERROR: --case requires --suite" >&2
  exit 1
fi

if [[ -z "${PYTHON:-}" ]]; then
  if python -c "import numpy" 2>/dev/null; then
    PYTHON=python
  elif [[ -x "${HOME}/miniforge3/envs/cuda-ops/bin/python" ]]; then
    PYTHON="${HOME}/miniforge3/envs/cuda-ops/bin/python"
  else
    echo "ERROR: need Python with numpy (e.g. conda activate cuda-ops)" >&2
    exit 1
  fi
fi

echo "Running Python tests with PYTHON=${PYTHON} PYTHONPATH=${PYTHONPATH}"

resolve_suite_path() {
  local suite="$1"
  if [[ -f "${suite}" ]]; then
    printf '%s\n' "$(cd "$(dirname "${suite}")" && pwd)/$(basename "${suite}")"
    return 0
  fi
  if [[ -f "${ROOT_DIR}/tests/${suite}" ]]; then
    printf '%s\n' "${ROOT_DIR}/tests/${suite}"
    return 0
  fi
  if [[ -f "${ROOT_DIR}/${suite}" ]]; then
    printf '%s\n' "${ROOT_DIR}/${suite}"
    return 0
  fi
  return 1
}

run_test_file() {
  local test_file="$1"
  echo "===== ${PYTHON} ${test_file} ====="
  "${PYTHON}" "${test_file}"
}

run_test_case() {
  local test_file="$1"
  local case_name="$2"
  local module_name
  module_name="$(basename "${test_file}" .py)"
  echo "===== ${PYTHON} -c ${module_name}.${case_name}() ====="
  "${PYTHON}" -c "import ${module_name} as _m; _m.${case_name}()"
}

status=0

if [[ -n "${SUITE}" ]]; then
  test_file="$(resolve_suite_path "${SUITE}")" || {
    echo "ERROR: test file not found: ${SUITE}" >&2
    exit 1
  }
  if [[ -n "${CASE}" ]]; then
    if ! run_test_case "${test_file}" "${CASE}"; then
      echo "Test failed: ${test_file} :: ${CASE}"
      status=1
    fi
  else
    if ! run_test_file "${test_file}"; then
      echo "Test failed: ${test_file}"
      status=1
    fi
  fi
  exit "${status}"
fi

while IFS= read -r -d '' test_file; do
  if ! run_test_file "${test_file}"; then
    echo "Test failed: ${test_file}"
    status=1
  fi
done < <(find "${ROOT_DIR}/tests" -maxdepth 2 -name "test_*.py" -print0 | sort -z)

exit "${status}"
