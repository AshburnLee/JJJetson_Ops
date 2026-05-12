#!/usr/bin/env bash
# Apply clang-format to C++/CUDA and ruff format to Python tests.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

resolve_clang_format() {
  if [[ -n "${CLANG_FORMAT:-}" ]] && command -v "${CLANG_FORMAT}" >/dev/null 2>&1; then
    echo "${CLANG_FORMAT}"
    return 0
  fi
  for c in clang-format clang-format-19 clang-format-18 clang-format-17 \
           clang-format-16 clang-format-15 clang-format-14; do
    if command -v "${c}" >/dev/null 2>&1; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

main() {
  local cf files
  if cf="$(resolve_clang_format)"; then
    echo "==> clang-format -i ($cf)"
    mapfile -d '' files < <(
      find "${ROOT_DIR}" \( \
        -path "${ROOT_DIR}/build" -o \
        -path "${ROOT_DIR}/build_fresh" -o \
        -path "${ROOT_DIR}/.git" \
      \) -prune -o -type f \( \
        -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o \
        -name "*.h" -o -name "*.hpp" -o -name "*.inl" \
      \) -print0
    )
    if ((${#files[@]})); then
      "${cf}" -i "${files[@]}"
    fi
  else
    echo "WARN: clang-format not found; skipped C++/CUDA." >&2
  fi

  if command -v ruff >/dev/null 2>&1; then
    echo "==> ruff format"
    ruff format tests
  else
    echo "WARN: ruff not found; skipped Python." >&2
  fi

  echo "Done."
}

main "$@"
