#!/bin/bash

# Exit on error
set -e

# --- 1. Check for ONNX file argument ---
if [ -z "$1" ] || [ ! -f "$1" ]; then
    echo "Usage: ./scripts/build_rv64.sh <model.onnx>"
    echo "Example: ./scripts/build_rv64.sh models/resnet18-v1-7.onnx"
    exit 1
fi

ONNX_FILE=$1
MODEL_NAME="${ONNX_FILE%.*}" # Removes the .onnx extension
MODEL_BASENAME="$(basename "$MODEL_NAME")"
OUT_DIR="build_rv64/${MODEL_BASENAME}"

# --- Configuration (Absolute paths recommended) ---
OM=/opt/onnx-mlir-x86/bin/onnx-mlir
MLIR_TRANSLATE=$PWD/llvm-project/build-x86/bin/mlir-translate
LLC=$PWD/llvm-project/build-x86/bin/llc
RISCV_GCC=riscv64-linux-gnu-gcc

TRIPLE=riscv64-unknown-linux-gnu
ARCH_GCC=rv64imafdc
ARCH_LLC=+m,+a,+f,+d,+c
ABI=lp64d

RUNTIME_LIB=$PWD/riscv_runtime/libcruntime.a
INCLUDE_DIR=$PWD/include
ONNX_RUNTIME_DIR=$PWD/include/onnx-mlir/Runtime

# --- 2. Create and Setup Directory ---
echo "Setting up folder: $OUT_DIR..."
mkdir -p "$OUT_DIR"

# Copy the onnx file into the folder so the source is preserved there
cp "$ONNX_FILE" "$OUT_DIR/"

# Move into the directory to keep all artifacts isolated
cd "$OUT_DIR"

# --- 3. Emit MLIR --- : rewrite that graph into a structured computer program.
#It is translating everything into a language that the standard LLVM optimizer can understand
echo "Step 1: Emitting LLVM-dialect MLIR..."
$OM "./$ONNX_FILE" -O3 --EmitLLVMIR --mtriple=$TRIPLE -o "$MODEL_BASENAME"

# --- 4. Translate to LLVM IR ---
echo "Step 2: Translating MLIR to LLVM IR..."
$MLIR_TRANSLATE "./${MODEL_BASENAME}.onnx.mlir" --mlir-to-llvmir -o "${MODEL_BASENAME}.ll"

# --- 5. Compile to Object ---
echo "Step 3: Compiling to RISC-V object file..."
$LLC -O3 -mtriple=$TRIPLE -mattr=$ARCH_LLC -target-abi=$ABI -filetype=obj \
    "${MODEL_BASENAME}.ll" -o "${MODEL_BASENAME}.o"

# --- 6. Link Executable ---
echo "Step 4: Linking executable..."
echo "Step 4: Linking executable with LLD..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
$RISCV_GCC -O2 -march=$ARCH_GCC -mabi=$ABI \
    -B"$LLVM_BIN_DIR" \
    -fuse-ld=lld \
    -I"$INCLUDE_DIR" -I"$ONNX_RUNTIME_DIR" \
    "$REPO_ROOT/tools/driver_imagenet.c" "${MODEL_BASENAME}.o" "$RUNTIME_LIB" \
    -lm -Wl,--gc-sections -static -o "${MODEL_BASENAME}_rv64"
echo "----------------------------------------"
echo "Success! Folder '$OUT_DIR' contains:"
ls -F