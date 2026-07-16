#!/usr/bin/env bash
set -euo pipefail

# run_resnet18_benchmark.sh
# Generates ResNet-18 artifacts, links, runs on Spike (if available), and
# collects simple timing/disassembly outputs. Intended to run on a Linux host
# or inside a Docker image with riscv tooling and Spike + pk available.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/tests/15-resnet18-float"
MODEL_DIR="$ROOT_DIR/examples/15-resnet18-float"

mkdir -p "$OUT_DIR"

ONNX_MODEL="$MODEL_DIR/resnet18.onnx"
if [ ! -f "$ONNX_MODEL" ]; then
  echo "ERROR: expected ONNX model at $ONNX_MODEL" >&2
  exit 1
fi

# 1. Generate artifacts using ONNX-MLIR
if ! command -v ONNX_MLIR >/dev/null 2>&1; then
  echo "ERROR: ONNX_MLIR not found in PATH. Build ONNX-MLIR with Gemmini support." >&2
  exit 1
fi

echo "Generating MLIR/LLVM/RV64 artifacts..."
ONNX_MLIR --maccel=Gemmini --mtriple=riscv64-unknown-linux-gnu "$ONNX_MODEL" -o "$OUT_DIR/resnet18_rv64"

# The above produces resnet18.o and related files in OUT_DIR
cd "$OUT_DIR" 

# 2. Ensure runtime archives exist (assume they are built in repo or provided)
if [ ! -f libgemmini_f32_rv64.a ] || [ ! -f libcruntime_rv64.a ]; then
  echo "WARNING: libgemmini_f32_rv64.a or libcruntime_rv64.a not found in $OUT_DIR."
  echo "If missing, build Gemmini runtime and ONNX-MLIR runtime and copy archives here."
fi

# 3. Link final ELF (uses riscv64 toolchain)
if ! command -v riscv64-linux-gnu-g++ >/dev/null 2>&1; then
  echo "WARNING: riscv64-linux-gnu-g++ not found. Skipping link step." >&2
else
  echo "Linking final ELF: resnet18_rv64"
  riscv64-linux-gnu-g++ -static -o resnet18_rv64 \
    float_model_runner.o \
    resnet18.o \
    libgemmini_f32_rv64.a \
    libcruntime_rv64.a \
    -lm || echo "Link may have failed; check missing archives or object files."
fi

# 4. Disassemble and collect .insn lines
if [ -f resnet18_rv64 ]; then
  echo "Collecting disassembly and symbol listings..."
  riscv64-linux-gnu-objdump -d resnet18_rv64 > resnet18_rv64.disasm || true
  riscv64-linux-gnu-nm -S resnet18_rv64 > resnet18_rv64.nm || true
  grep -E '\.insn|om_gemmini_' resnet18_rv64.disasm -n -A2 > resnet18_rv64.gemmini_insn || true
  readelf -SW resnet18_rv64 > resnet18_rv64.sections || true
fi

# 5. Run on Spike if available
if command -v spike >/dev/null 2>&1; then
  if [ -f resnet18_rv64 ]; then
    echo "Running resnet18_rv64 on Spike (may be slow)..."
    spike --extension=gemmini --isa=rv64imafdc "$RISCV_PK/bin/pk" ./resnet18_rv64 | tee resnet18_rv64.spike.out || true
  else
    echo "Skipping Spike run: resnet18_rv64 not present." >&2
  fi
else
  echo "Spike not found; skip simulation run. Use a Gemmini-enabled Spike to run the binary." >&2
fi

# 6. Summarize outputs
echo "Outputs written to: $OUT_DIR"
ls -lh "$OUT_DIR" | sed -n '1,200p'

echo "Done. Review resnet18_rv64.disasm and resnet18_rv64.gemmini_insn for .insn evidence."