#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=1

usage() {
  cat <<'EOF'
Usage: scripts/cleanup_project.sh [--apply]

Removes generated build and test artifacts from the ONNX-MLIR/Gemmini tree.
The default mode is a dry run; pass --apply to delete the listed paths.

This script intentionally avoids tracked source, tracked tests, and tracked docs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      DRY_RUN=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

remove_path() {
  local path="$1"
  [[ -e "$path" ]] || return 0

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "would remove: $path"
  else
    rm -rf "$path"
    echo "removed: $path"
  fi
}

remove_untracked_path() {
  local path="$1"
  [[ -e "$path" ]] || return 0

  if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    echo "skip tracked: $path"
    return 0
  fi

  remove_path "$path"
}

echo "ONNX-MLIR/Gemmini cleanup"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "mode: dry run"
else
  echo "mode: apply"
fi

remove_untracked_path "gemmini_toolchain_build"
remove_untracked_path "llvm-project/build-x86"
remove_untracked_path "build_rv64"
remove_untracked_path "build_test"
remove_untracked_path "test/accelerators/gemmini/integration/Output"
remove_untracked_path "compile_commands.json"
remove_untracked_path "output.txt"
remove_untracked_path "zoo_report.json"
remove_untracked_path "onnx-mlir.code-workspace"
remove_untracked_path "third_party/onnx/CMakeFiles"
remove_untracked_path "third_party/stablehlo/CMakeFiles"
remove_untracked_path "third_party/benchmark/CMakeFiles"
remove_untracked_path "third_party/rapidcheck/CMakeFiles"
remove_untracked_path "third_party/pybind11/CMakeFiles"

while IFS= read -r path; do
  remove_untracked_path "$path"
done < <(find . \( -path ./.git -o -path ./.venv -o -path ./venv -o -path ./env -o -path ./ENV -o -path ./third_party -o -path ./llvm-project -o -path ./gemmini_toolchain_build -o -path ./test/accelerators/gemmini/integration/Output \) -prune -o -type d \( -name CMakeFiles -o -name CMakeScratch -o -name __pycache__ -o -name rt_obj -o -name gem_rt_obj -o -name onnx_obj -o -name gem_obj -o -name '*.tmp.dir' \) -print)

while IFS= read -r path; do
  remove_untracked_path "$path"
done < <(find . \( -path ./.git -o -path ./.venv -o -path ./venv -o -path ./env -o -path ./ENV -o -path ./third_party -o -path ./llvm-project -o -path ./gemmini_toolchain_build -o -path ./test/accelerators/gemmini/integration/Output \) -prune -o -type f \( -name CMakeCache.txt -o -name cmake_install.cmake -o -name CTestTestfile.cmake -o -name '*.tmp' -o -name '*.pyc' -o -name '*.o' -o -name '*.a' -o -name '*.so' \) -print)

while IFS= read -r path; do
  remove_untracked_path "$path"
done < <(find docs include src -type f -name Makefile -print)

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete. Re-run with --apply to remove these generated artifacts."
else
  echo "Cleanup complete."
fi
