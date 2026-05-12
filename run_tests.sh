#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/python:${PYTHONPATH-}"

echo "Running Python tests with PYTHONPATH=${PYTHONPATH}"

status=0

while IFS= read -r -d '' test_file; do
  echo "===== python ${test_file} ====="
  if ! python "${test_file}"; then
    echo "Test failed: ${test_file}"
    status=1
  fi
done < <(find "${ROOT_DIR}/tests" -maxdepth 2 -name "test_*.py" -print0 | sort -z)

exit "${status}"
