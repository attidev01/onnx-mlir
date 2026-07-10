# Gemmini Troubleshooting

## CMake Cannot Find Gemmini

Symptom:

```text
Could NOT find Gemmini
```

Check that the Gemmini support tree exists and contains:

```text
gemmini/lib/gemmini-rocc-tests/include/gemmini.h
gemmini/lib/gemmini-rocc-tests/include/gemmini_params.h
```

Fix:

```sh
cmake -S . -B gemmini_toolchain_build \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DGEMMINI_ROOT=/path/to/gemmini
```

Then run:

```sh
scripts/check_prerequisites.sh --mode compile-only
```

## Invalid Gemmini Mode

Symptom:

```text
Invalid ONNX_MLIR_GEMMINI_MODE
```

Valid values are:

- `compile-only`
- `hardware`
- `simulator`

Fix:

```sh
cmake -S . -B gemmini_toolchain_build \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DONNX_MLIR_GEMMINI_MODE=compile-only
```

## Simulator Dependencies Missing

Symptom:

```text
Gemmini simulator mode requires Gemmini RoCC headers, spike, pk, and libgemmini.so.
```

Check:

```sh
scripts/check_prerequisites.sh --mode simulator
```

Install:

```sh
RISCV_INSTALL="$HOME/riscv-gemmini" scripts/setup_gemmini_simulator.sh --install
```

Then configure:

```sh
cmake -S . -B gemmini_toolchain_build \
  -DONNX_MLIR_ENABLE_GEMMINI=ON \
  -DONNX_MLIR_GEMMINI_MODE=simulator
```

## Hardware Mode Cannot Find RISC-V GCC

Symptom:

```text
Gemmini hardware mode requires Gemmini RoCC headers and a RISC-V GCC.
```

Install or expose one of:

- `riscv64-linux-gnu-gcc`
- `riscv64-unknown-elf-gcc`

Check:

```sh
scripts/check_prerequisites.sh --mode hardware
```

## `--maccel=Gemmini` Is Rejected

Possible causes:

- The compiler was built without `-DONNX_MLIR_ENABLE_GEMMINI=ON`.
- The configured accelerator list did not include `Gemmini`.
- You are running a different `onnx-mlir` binary than the one just built.

Check:

```sh
gemmini_toolchain_build/Release/bin/onnx-mlir --help | rg -i 'maccel|gemmini'
```

Reconfigure:

```sh
cmake -S . -B gemmini_toolchain_build \
  -DONNX_MLIR_ENABLE_GEMMINI=ON
cmake --build gemmini_toolchain_build --target onnx-mlir
```

## Model Compiles But No Gemmini Calls Appear

Symptom:

```sh
rg 'gemmini|om_gemmini' /tmp/model.onnx.mlir
```

finds only module metadata or nothing.

Explanation: selecting the Gemmini accelerator does not guarantee every ONNX
op is legal for Gemmini. The op may have unsupported shape, type, layout,
attributes, or dynamic dimensions.

Debug:

```sh
onnx-mlir --maccel=Gemmini --EmitMLIR model.onnx -o /tmp/model_gemmini
python3 test/accelerators/gemmini/tools/memory_access_checker.py \
  /tmp/model_gemmini.onnx.mlir
```

Look for:

- `om_gemmini_*` runtime calls,
- `gemmini_low.*` direct ops,
- `llvm.inline_asm` in LLVM dialect output.

## RISC-V Inline Assembly Does Not Appear

The direct path only appears after Gemmini dialect lowering. Use:

```sh
onnx-mlir \
  --maccel=Gemmini \
  --mtriple=riscv64-unknown-linux-gnu \
  --EmitLLVMIR \
  model.onnx \
  -o /tmp/model_rv64
```

Then inspect:

```sh
rg 'llvm.inline_asm|\\.insn|CUSTOM_3' /tmp/model_rv64.onnx.mlir
```

If nothing appears, the model likely used the runtime path or did not match a
direct-path lowering.

## Numerical Mismatch

For fp32 MatMul, the current hardware path quantizes to int8, accumulates in
i32, then dequantizes. Some error relative to pure fp32 CPU execution is
expected.

Debug steps:

1. Reduce the model to the smallest failing MatMul.
2. Compare candidate and reference JSON outputs:

```sh
python3 test/accelerators/gemmini/tools/correctness_verifier.py \
  --mode json \
  --candidate candidate.json \
  --reference reference.json \
  --tolerance 1e-5
```

3. Try looser tolerances to determine whether the failure is quantization
   error or a shape/layout bug.

## Scratchpad Range Violations

Symptom:

```json
"reason": "scratchpad row range exceeds 16384 rows"
```

The direct tiling path emitted a tile placement outside the configured Gemmini
scratchpad model.

Debug:

```sh
python3 test/accelerators/gemmini/tools/memory_access_checker.py model.onnx.mlir
```

Then inspect the nearest `gemmini.mvin` or `gemmini_low.mvin` operations and
their `*_spad_offset_rows` attributes.

## FAQ

### Is `ONNX_MLIR_ENABLE_GEMMINI=ON` required?

It is the recommended compatibility flag. It appends `Gemmini` to
`ONNX_MLIR_ACCELERATORS`.

### Can I still use `ONNX_MLIR_ACCELERATORS=Gemmini`?

Yes. The new flag is a convenience path; the backend still uses ONNX-MLIR's
standard accelerator list internally.

### Does compile-only mode run Gemmini hardware?

No. It builds and tests the compiler backend without requiring Spike or real
hardware.

### Does simulator mode install Spike automatically?

No. CMake only validates dependencies. Run
`scripts/setup_gemmini_simulator.sh --install` to build/install simulator
pieces.

### How do I know an op was accelerated?

Inspect emitted IR. Runtime path acceleration appears as `om_gemmini_*`.
Direct path acceleration appears as `gemmini_low.*` before LLVM lowering or
`llvm.inline_asm` after LLVM lowering.

### Why is fp32 MatMul not exact?

The current Gemmini hardware profile is int8 element with int32 accumulation.
fp32 MatMul is routed through quantization and dequantization.
