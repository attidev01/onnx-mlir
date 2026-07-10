# Gemmini Dependencies

This document lists the tools required to build and validate the ONNX-MLIR
Gemmini accelerator backend.

## Build Modes

The CMake flag `ONNX_MLIR_GEMMINI_MODE` controls dependency strictness:

| Mode | Purpose | Required Gemmini Pieces |
| --- | --- | --- |
| `compile-only` | Build the compiler backend and MLIR tests without running a simulator. | Gemmini RoCC headers. |
| `hardware` | Build for a RISC-V target that can execute Gemmini RoCC instructions. | Gemmini RoCC headers and a RISC-V GCC. |
| `simulator` | Build expecting Spike + Gemmini simulation support. | Gemmini RoCC headers, `spike`, `pk`, and `libgemmini.so`. |

Example:

```sh
cmake -S . -B gemmini_toolchain_build \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DONNX_MLIR_GEMMINI_MODE=compile-only
```

## Required Packages

The descriptive requirements file is `requirements-gemmini.txt`.

Recommended host packages:

```sh
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  make \
  python3 \
  device-tree-compiler
```

Hardware and simulator modes also need RISC-V C and C++ cross-compilers:

```sh
sudo apt-get install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

or a `riscv64-unknown-elf-gcc` / `riscv64-unknown-elf-g++` toolchain on `PATH`.

**Important:** `OMRuntimeGemmini.cpp` is compiled as C++ (step 6 of
`run_float_model_spike.sh` and `run_quantized_spike.sh`). The final link step
uses `$RISCV_GXX` (g++-13 or later) to automatically pull in `libstdc++`. A
C-only cross-toolchain is insufficient — `g++` must be present and discoverable
as `${RISCV_GCC/gcc/g++}`.

Verify the C++ cross-compiler is present:

```sh
riscv64-linux-gnu-g++ --version
# Expected: riscv64-linux-gnu-g++ (Ubuntu 13.x.x ...) 13.x.x
```

## Gemmini Source Tree

By default, CMake expects the support tree at:

```text
./gemmini
```

The required headers are:

```text
gemmini/lib/gemmini-rocc-tests/include/gemmini.h
gemmini/lib/gemmini-rocc-tests/include/gemmini_params.h
```

Use `-DGEMMINI_ROOT=/path/to/gemmini` if the checkout lives elsewhere.

## Simulator Setup

Run the checker first:

```sh
scripts/check_prerequisites.sh --mode simulator
```

To install Spike, riscv-pk, and `libgemmini.so` into a local prefix:

```sh
RISCV_INSTALL="$HOME/riscv-gemmini" scripts/setup_gemmini_simulator.sh --install
```

After installation:

```sh
export RISCV_INSTALL="$HOME/riscv-gemmini"
export LD_LIBRARY_PATH="$RISCV_INSTALL/lib:$LD_LIBRARY_PATH"
scripts/check_prerequisites.sh --mode simulator
```

Simulator mode CMake configuration:

```sh
cmake -S . -B gemmini_toolchain_build \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DONNX_MLIR_GEMMINI_MODE=simulator
```

## Validation

Compile-only prerequisite check:

```sh
scripts/check_prerequisites.sh --mode compile-only
```

Hardware prerequisite check:

```sh
scripts/check_prerequisites.sh --mode hardware
```

Gemmini integration tests:

```sh
cmake --build gemmini_toolchain_build --target check-gemmini-integration
```

## Clear Failure Messages

When `ONNX_MLIR_ENABLE_GEMMINI=ON`, CMake uses `cmake/FindGemmini.cmake`.
Explicit `hardware` and `simulator` modes fail during configuration if their
required tools are missing. The error message points to
`scripts/check_prerequisites.sh` and, for simulator mode,
`scripts/setup_gemmini_simulator.sh`.
