<!--- SPDX-License-Identifier: Apache-2.0 -->

# Build Guide

This page is the consolidated entry point for building ONNX-MLIR and the
Gemmini accelerator backend.

## Prerequisites

Keep the base ONNX-MLIR prerequisite list in sync with
[Prerequisite.md](Prerequisite.md):

```text
python >= 3.8
gcc >= 6.4
protobuf >= 4.21.12
cmake >= 3.13.4
make >= 4.2.1 or ninja >= 1.10.2
java >= 1.11 (optional)
```

Gemmini-specific prerequisites are documented in
[gemmini/GemminiDependencies.md](gemmini/GemminiDependencies.md). For a quick
local check:

```sh
scripts/check_prerequisites.sh --mode compile-only
```

## LLVM/MLIR

ONNX-MLIR depends on a known-good LLVM commit. Use the repository helper scripts
instead of checking out LLVM by hand:

```sh
utils/clone-mlir.sh
utils/build-mlir.sh
```

The resulting `MLIR_DIR` normally points at:

```sh
llvm-project/build/lib/cmake/mlir
```

See [BuildOnLinuxOSX.md](BuildOnLinuxOSX.md) and
[BuildOnWindows.md](BuildOnWindows.md) for platform-specific details that have
not yet been folded into this page.

## Standard ONNX-MLIR Build

```sh
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR="$PWD/llvm-project/build/lib/cmake/mlir"

cmake --build build
```

Run the main lit suite:

```sh
cmake --build build --target check-onnx-lit
```

## Gemmini Compile-Only Build

Compile-only mode is the recommended development mode. It builds the compiler
backend and Gemmini tests without requiring Spike or a RISC-V simulator.

```sh
cmake -S . -B gemmini_toolchain_build \
  -G Ninja \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DONNX_MLIR_GEMMINI_MODE=compile-only \
  -DONNX_MLIR_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build gemmini_toolchain_build -j"$(nproc)"
```

## Gemmini Simulator Build

Install Spike, riscv-pk, and the Gemmini Spike extension first:

```sh
RISCV_INSTALL="$HOME/riscv-gemmini" \
  scripts/setup_gemmini_simulator.sh --install
```

Then configure with simulator mode:

```sh
cmake -S . -B gemmini_toolchain_build \
  -G Ninja \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DONNX_MLIR_GEMMINI_MODE=simulator \
  -DRISCV_INSTALL="$HOME/riscv-gemmini" \
  -DONNX_MLIR_BUILD_TESTS=ON
```

## Updating ONNX

The ONNX update flow remains in [BuildONNX.md](BuildONNX.md). In short:

1. Update the `third_party/onnx` checkout.
2. Reinstall ONNX from `third_party/onnx`.
3. Regenerate operation definitions.
4. Regenerate supported-op documentation.
5. Run lit, backend, and accelerator tests.

## Testing After Cleanup

Run the consolidated local check script:

```sh
scripts/test_all.sh
```

For a fresh build directory, set `BUILD_DIR`:

```sh
BUILD_DIR="$PWD/build" scripts/test_all.sh
```

See [Testing.md](Testing.md) and
[gemmini/GemminiTesting.md](gemmini/GemminiTesting.md) for the full test matrix.
