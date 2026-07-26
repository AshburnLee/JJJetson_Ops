#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}/tests:${PYTHONPATH-}"

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

status=0

while IFS= read -r -d '' test_file; do
  echo "===== python ${test_file} ====="
  if ! "${PYTHON}" "${test_file}"; then
    echo "Test failed: ${test_file}"
    status=1
  fi
done < <(find "${ROOT_DIR}/tests" -maxdepth 2 -name "test_*.py" -print0 | sort -z)

exit "${status}"
