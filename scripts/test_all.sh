#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/gemmini_toolchain_build}"
RUN_GEMMINI_EXAMPLES="${RUN_GEMMINI_EXAMPLES:-auto}"

cd "$REPO_ROOT"

echo "=== ONNX-MLIR/Gemmini checks ==="

echo "Checking Gemmini compile prerequisites..."
scripts/check_prerequisites.sh --mode compile-only

if [[ -d "$BUILD_DIR" ]]; then
  echo "Running lit tests from $BUILD_DIR..."
  cmake --build "$BUILD_DIR" --target check-onnx-lit

  if cmake --build "$BUILD_DIR" --target help | grep -q "check-onnx-accelerator-gemmini"; then
    echo "Running Gemmini accelerator lit target..."
    cmake --build "$BUILD_DIR" --target check-onnx-accelerator-gemmini
  fi
else
  echo "Build directory not found: $BUILD_DIR"
  echo "Set BUILD_DIR=/path/to/build or configure the project before running build-backed checks."
fi

case "$RUN_GEMMINI_EXAMPLES" in
  auto)
    if [[ -x "$BUILD_DIR/Release/bin/onnx-mlir" || -x "$BUILD_DIR/Debug/bin/onnx-mlir" ]]; then
      RUN_GEMMINI_EXAMPLES=1
    else
      RUN_GEMMINI_EXAMPLES=0
    fi
    ;;
  1|true|yes|on) RUN_GEMMINI_EXAMPLES=1 ;;
  0|false|no|off) RUN_GEMMINI_EXAMPLES=0 ;;
  *)
    echo "Invalid RUN_GEMMINI_EXAMPLES=$RUN_GEMMINI_EXAMPLES" >&2
    exit 2
    ;;
esac

if [[ "$RUN_GEMMINI_EXAMPLES" -eq 1 && -x gemmini/tests/run_all_tests.sh ]]; then
  echo "Running Gemmini shell tests..."
  gemmini/tests/run_all_tests.sh
else
  echo "Skipping Gemmini shell tests. Set RUN_GEMMINI_EXAMPLES=1 after configuring a build."
fi

echo "All requested checks completed."
