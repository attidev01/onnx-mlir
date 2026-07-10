#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RISCV_INSTALL="${RISCV_INSTALL:-$HOME/riscv-gemmini}"
WORK_DIR="${WORK_DIR:-/tmp/spike-gemmini-build}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
INSTALL=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_gemmini_simulator.sh [--install] [--check-only]

Prepares the local Spike + Gemmini simulator environment used by the
ONNX-MLIR Gemmini backend.

Default behavior is check-only. Pass --install to run the network/building
installer that clones and builds Spike, riscv-pk, and libgemmini.so.

Environment:
  RISCV_INSTALL   Install prefix. Defaults to ~/riscv-gemmini.
  WORK_DIR        Build directory. Defaults to /tmp/spike-gemmini-build.
  JOBS            Parallel build jobs. Defaults to available CPU count.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      INSTALL=1
      shift
      ;;
    --check-only)
      INSTALL=0
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

echo "Gemmini simulator setup"
echo "  repo root: $REPO_ROOT"
echo "  RISCV_INSTALL: $RISCV_INSTALL"
echo "  WORK_DIR: $WORK_DIR"
echo "  JOBS: $JOBS"

if [[ "$INSTALL" -eq 0 ]]; then
  "$REPO_ROOT/scripts/check_prerequisites.sh" --mode simulator
  exit 0
fi

INSTALLER="$REPO_ROOT/gemmini/examples/tools/install_spike_gemmini.sh"
if [[ ! -x "$INSTALLER" ]]; then
  echo "missing simulator installer: $INSTALLER" >&2
  echo "expected the Gemmini support tree under $REPO_ROOT/gemmini" >&2
  exit 1
fi

echo "Starting Spike + Gemmini simulator install."
echo "This may clone upstream repositories and compile Spike/riscv-pk."
RISCV_INSTALL="$RISCV_INSTALL" WORK_DIR="$WORK_DIR" JOBS="$JOBS" bash "$INSTALLER"

RISCV_INSTALL="$RISCV_INSTALL" "$REPO_ROOT/scripts/check_prerequisites.sh" --mode simulator
