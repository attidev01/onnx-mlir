# Gemmini Backend — Quick Start

Get from zero to a compiled ONNX model in five steps.

**Current status:** 41/41 lit · 22 examples passing (1 external skip) · 18/18 integration · 9/9 model zoo compile-only validation.

---

## Step 1 — Check Prerequisites

```sh
# Compile-only mode (no RISC-V toolchain required)
scripts/check_prerequisites.sh --mode compile-only

# Simulator mode (Spike + Gemmini)
scripts/check_prerequisites.sh --mode simulator
```

Install the simulator when needed:

```sh
RISCV_INSTALL="$HOME/riscv-gemmini" scripts/setup_gemmini_simulator.sh --install
```

---

## Step 2 — Configure and Build

**Compile-only** (recommended for development):

```sh
cmake -S . -B gemmini_toolchain_build \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DONNX_MLIR_GEMMINI_MODE=compile-only \
  -DONNX_MLIR_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja

cmake --build gemmini_toolchain_build -j"$(nproc)"
```

**Simulator mode** (add after `--install` above):

```sh
cmake -S . -B gemmini_toolchain_build \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DONNX_MLIR_GEMMINI_MODE=simulator \
  -DRISCV_INSTALL="$HOME/riscv-gemmini" \
  -DONNX_MLIR_BUILD_TESTS=ON -G Ninja
```

---

## Step 3 — Compile a Model

```sh
ONNX_MLIR=gemmini_toolchain_build/Release/bin/onnx-mlir

# Emit MLIR for inspection
$ONNX_MLIR --maccel=Gemmini --EmitMLIR model.onnx -o /tmp/model_gemmini

# Count Gemmini runtime calls
grep -c 'om_gemmini_' /tmp/model_gemmini.onnx.mlir

# Compile to RV64 ELF (requires RISC-V toolchain)
$ONNX_MLIR --maccel=Gemmini \
  --mtriple=riscv64-unknown-linux-gnu --mcpu=rocket \
  model.onnx -o model_rv64
```

---

## Step 4 — Run Tests

```sh
# MLIR lit tests (41)
cmake --build gemmini_toolchain_build --target check-onnx_mlir-accelerators-gemmini

# Integration tests (18) — includes Spike numerical correctness
cmake --build gemmini_toolchain_build --target check-gemmini-integration

# Example shell suite (22, tiers 1–4)
bash gemmini/tests/run_all_tests.sh

# Model zoo (9 models, compile-only)
python3 test/accelerators/gemmini/tools/run_model_zoo.py \
  --onnx-mlir $ONNX_MLIR --repo-root .
```

---

## Step 5 — Run on Spike

```sh
spike \
  --extension=gemmini \
  --isa=rv64imafdc \
  "$HOME/riscv-gemmini/bin/pk" \
  ./model_rv64
```

Or run the YOLOX-nano end-to-end validation (requires `yolox_nano.onnx`
at repo root):

```sh
bash gemmini/tests/19-yolo-real/run_test.sh
```

---

## What Gets Accelerated

```sh
# After compiling with --EmitMLIR, check what was routed to Gemmini:
grep 'funcName = ' /tmp/model_gemmini.onnx.mlir | sort | uniq -c | sort -rn
```

- `funcName = "om_gemmini_*"` → runtime-delegate path (Conv, Gemm, Sigmoid, Mul, …)
- `gemmini_low.*` ops → direct-RoCC path (INT8 MatMul, CUSTOM_3 instructions)
- Remaining `onnx.*` ops → standard scalar fallback (unsupported shapes/dtypes)

See [GemminiBackend.md](GemminiBackend.md) for the full supported-operator table.

---

## Useful Links

| Document | What it covers |
|---|---|
| [GemminiBackend.md](GemminiBackend.md) | Architecture, full operator table, pass pipeline |
| [GemminiTesting.md](GemminiTesting.md) | All four test suites with commands |
| [GemminiExamples.md](GemminiExamples.md) | Per-operator usage examples |
| [GemminiDependencies.md](GemminiDependencies.md) | Full dependency list |
| [GemminiTroubleshooting.md](GemminiTroubleshooting.md) | Common build and runtime errors |
| [VALIDATION.md](VALIDATION.md) | Full validation report with benchmark numbers |
| [GemminiCodeMap.md](GemminiCodeMap.md) | Source map for the backend implementation |
