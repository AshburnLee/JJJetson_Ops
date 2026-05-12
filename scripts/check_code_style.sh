#!/usr/bin/env bash
# Check C++/CUDA formatting (clang-format) and Python lint/format (ruff).
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

run_clang_format_check() {
  local cf
  if ! cf="$(resolve_clang_format)"; then
    echo "SKIP: clang-format not found (install LLVM clang-format or set CLANG_FORMAT)." >&2
    if [[ "${STRICT:-0}" == "1" ]]; then
      echo "STRICT=1: clang-format is required." >&2
      return 1
    fi
    return 0
  fi
  echo "==> clang-format ($cf) dry-run"
  local bad=0
  while IFS= read -r -d '' f; do
    if ! "${cf}" --dry-run --Werror "${f}" >/dev/null 2>&1; then
      echo "  would reformat: ${f}" >&2
      bad=1
    fi
  done < <(
    find "${ROOT_DIR}" \( \
      -path "${ROOT_DIR}/build" -o \
      -path "${ROOT_DIR}/build_fresh" -o \
      -path "${ROOT_DIR}/.git" \
    \) -prune -o -type f \( \
      -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o \
      -name "*.h" -o -name "*.hpp" -o -name "*.inl" \
    \) -print0
  )
  if [[ "${bad}" -ne 0 ]]; then
    echo "clang-format: failed. Run: ${cf} -i <files> or see scripts/format_code.sh" >&2
    return 1
  fi
}

run_ruff() {
  if ! command -v ruff >/dev/null 2>&1; then
    echo "SKIP: ruff not on PATH (pip install ruff or conda env update -f environment.yml)." >&2
    if [[ "${STRICT:-0}" == "1" ]]; then
      echo "STRICT=1: ruff is required." >&2
      return 1
    fi
    return 0
  fi
  echo "==> ruff check"
  ruff check tests
  echo "==> ruff format --check"
  ruff format --check tests
}

fail=0
run_clang_format_check || fail=1
run_ruff || fail=1
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi
echo "OK: code style checks passed."
