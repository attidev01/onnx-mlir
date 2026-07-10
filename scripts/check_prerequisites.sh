#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODE="compile-only"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMINI_ROOT="${GEMMINI_ROOT:-$REPO_ROOT/gemmini}"
RISCV_INSTALL="${RISCV_INSTALL:-${RISCV:-$HOME/riscv-gemmini}}"

usage() {
  cat <<'EOF'
Usage: scripts/check_prerequisites.sh [--mode compile-only|hardware|simulator]

Checks the local toolchain pieces required by the ONNX-MLIR Gemmini backend.

Environment:
  GEMMINI_ROOT    Gemmini checkout root. Defaults to ./gemmini.
  RISCV_INSTALL   Spike/pk install prefix. Defaults to $RISCV or ~/riscv-gemmini.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:?missing mode after --mode}"
      shift 2
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

case "$MODE" in
  compile-only|hardware|simulator) ;;
  *)
    echo "invalid mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

failures=0

check_file() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "ok: $label: $path"
  else
    echo "missing: $label: $path" >&2
    failures=$((failures + 1))
  fi
}

check_cmd() {
  local label="$1"
  local cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "ok: $label: $(command -v "$cmd")"
  else
    echo "missing: $label: $cmd" >&2
    failures=$((failures + 1))
  fi
}

check_any_cmd() {
  local label="$1"
  shift
  local cmd
  for cmd in "$@"; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "ok: $label: $(command -v "$cmd")"
      return
    fi
  done
  echo "missing: $label: one of $*" >&2
  failures=$((failures + 1))
}

echo "Gemmini prerequisite check"
echo "  mode: $MODE"
echo "  repo root: $REPO_ROOT"
echo "  GEMMINI_ROOT: $GEMMINI_ROOT"
echo "  RISCV_INSTALL: $RISCV_INSTALL"

check_cmd "cmake" cmake
check_cmd "make" make
check_cmd "c++ compiler" g++
check_file "Gemmini RoCC header" "$GEMMINI_ROOT/lib/gemmini-rocc-tests/include/gemmini.h"
check_file "Gemmini params header" "$GEMMINI_ROOT/lib/gemmini-rocc-tests/include/gemmini_params.h"

if [[ "$MODE" == "hardware" || "$MODE" == "simulator" ]]; then
  check_any_cmd "RISC-V GCC" riscv64-linux-gnu-gcc riscv64-unknown-elf-gcc
fi

if [[ "$MODE" == "simulator" ]]; then
  check_file "Spike executable" "$RISCV_INSTALL/bin/spike"
  if [[ -e "$RISCV_INSTALL/bin/pk" ]]; then
    echo "ok: RISC-V proxy kernel: $RISCV_INSTALL/bin/pk"
  elif [[ -e "$RISCV_INSTALL/riscv64-linux-gnu/bin/pk" ]]; then
    echo "ok: RISC-V proxy kernel: $RISCV_INSTALL/riscv64-linux-gnu/bin/pk"
  else
    echo "missing: RISC-V proxy kernel: $RISCV_INSTALL/bin/pk or $RISCV_INSTALL/riscv64-linux-gnu/bin/pk" >&2
    failures=$((failures + 1))
  fi
  check_file "Gemmini Spike extension" "$RISCV_INSTALL/lib/libgemmini.so"
fi

if [[ "$failures" -ne 0 ]]; then
  cat >&2 <<EOF

Gemmini prerequisite check failed with $failures missing item(s).

For simulator mode, run:
  RISCV_INSTALL="$RISCV_INSTALL" scripts/setup_gemmini_simulator.sh --install

For CMake, use one of:
  -DONNX_MLIR_ENABLE_GEMMINI=ON -DONNX_MLIR_GEMMINI_MODE=compile-only
  -DONNX_MLIR_ENABLE_GEMMINI=ON -DONNX_MLIR_GEMMINI_MODE=hardware
  -DONNX_MLIR_ENABLE_GEMMINI=ON -DONNX_MLIR_GEMMINI_MODE=simulator
EOF
  exit 1
fi

echo "Gemmini prerequisite check passed."
