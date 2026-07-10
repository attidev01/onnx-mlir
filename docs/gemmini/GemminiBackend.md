# Gemmini Backend Architecture

## Overview

The Gemmini backend adds an ONNX-MLIR accelerator path for the UC Berkeley
Gemmini systolic-array architecture. It follows the standard ONNX-MLIR
accelerator interface and is selected with:

```sh
onnx-mlir --maccel=Gemmini model.onnx
```

The backend is **fully validated** for f32 inference on the Spike RISC-V simulator:
41/41 MLIR lit tests, 22 example tests passing with 1 external skip, 18/18 integration tests, 9/9 model zoo compile-only validations.

## Two Execution Paths

### Path 1 — Runtime Delegate (22 ONNX operators)

Supported ONNX ops are converted to `krnl.call` operations in `ONNXToGemmini.cpp`.
Each call targets an `om_gemmini_*` C function in `OMRuntimeGemmini.c`.

For compute-intensive ops (Conv, Gemm, MatMul f32), the runtime quantizes to
int8, calls Gemmini's `tiled_matmul_auto`, and dequantizes the int32 result.
For utility ops (Resize, Pad, Sigmoid, etc.), it runs a scalar CPU loop on
the RV64 core alongside the systolic array.

Patterns use `benefit=10` to win over the standard elementwise path (`benefit=1`).
Ops that fail the predicate fall through to the standard scalar path — no crash.

### Path 2 — Direct RoCC (INT8 MatMul, WS-mode)

INT8 `MatMul` is tiled to 16×16 blocks by `GemminiTiling`, lowered through the
`Gemmini` and `GemminiLow` MLIR dialects to `CUSTOM_3` RoCC inline assembly.
Double-buffering overlaps DMA loads with systolic computation (25–43% benefit).

## Compilation Pipeline

```
ONNX model
    │
    ▼  ONNXToGemmini pass  (benefit=10 patterns)
    ├──────────────────────────────────────────┐
    │ direct-RoCC ops                          │ runtime-delegate ops
    ▼                                          ▼
Gemmini dialect                         krnl.call om_gemmini_*(...)
    │ GemminiTiling                            │
    ▼                                          │ Standard Krnl→LLVM pipeline
GemminiToGemminiLow                           │
    │ GemminiLowRewrite                        │
    ▼                                          │
GemminiLowToLLVM                              │
    │ RISC-V CUSTOM_3 inline asm              │
    └──────────────────────────────────────────┘
    ▼
LLVM dialect → .onnx.mlir / .onnx.ll / .onnx.o / ELF
```

## Supported Operators

| ONNX op | Path | Runtime function(s) | Key constraints |
|---|---|---|---|
| `MatMul` int8 | Direct RoCC | `gemmini_low.*` (no call) | Rank-2, static, 16×16 WS-mode tiles |
| `MatMul` f32 | Runtime | `om_gemmini_matmul_f32_hw` | Rank-2 static f32; quantize→Gemmini→dequantize |
| `MatMul` f32 batched | Runtime | `om_gemmini_matmul_f32_nd_hw` | Rank-N × rank-2; loops over leading dims |
| `MatMulInteger` | Runtime | `om_gemmini_matmulinteger_i8i8acc32[_zp]` | i8×i8→i32; scalar or per-row zero-points |
| `QLinearConv` | Runtime | `om_gemmini_qlinearconv_i8[_bias]` | Rank-4 int8 NCHW; batch=1; group=1; scalar scales/zp |
| `Conv` f32 | Runtime | `om_gemmini_conv_f32[_bias]` | Rank-4 NCHW f32; dynamic batch; static C/H/W |
| `ConvTranspose` f32 | Runtime | `om_gemmini_convtranspose_f32[_bias]` | Rank-4 f32; col2im scatter-add; dynamic batch |
| `Gemm` f32 | Runtime | `om_gemmini_gemm_f32[_bias]` | Rank-2 f32; transA/transB/alpha/beta; optional bias |
| `Relu` f32 | Runtime | `om_gemmini_relu_f32` | Rank-4 f32; dynamic batch |
| `BatchNormalization` f32 | Runtime | `om_gemmini_batchnorm_f32` | Rank-4 NCHW f32; inference only |
| `Add` f32 | Runtime | `om_gemmini_add_f32` | Rank-4 f32; same-shape inputs; dynamic batch |
| `MaxPool` f32 | Runtime | `om_gemmini_maxpool_f32` | Rank-4 NCHW f32; static kernel/stride/pads |
| `AveragePool` f32 | Runtime | `om_gemmini_avgpool_f32` | Rank-4 NCHW f32; static kernel/stride/pads |
| `GlobalAveragePool` f32 | Runtime | `om_gemmini_globalavgpool_f32` | Rank-4 NCHW f32 |
| `Softmax` f32 | Runtime | `om_gemmini_softmax_f32` | Rank-2 or rank-4 f32 |
| `Resize` f32 | Runtime | `om_gemmini_resize_nearest_f32` / `…_linear_f32` | Rank-4 NCHW; nearest or bilinear |
| `Pad` f32 | Runtime | `om_gemmini_pad_constant/reflect/edge_f32` | Rank-4 NCHW; static spatial pads |
| `Concat` f32 | Runtime | `om_gemmini_concat_f32` | 2-input rank-4 NCHW f32; any axis |
| `Slice` f32 | Runtime | `om_gemmini_slice_f32` | Rank-4 f32; constant axes/starts/ends/steps |
| `Transpose` f32 | Runtime | `om_gemmini_transpose_f32_hw` | Rank-4 NCHW f32; constant perm |
| `Split` f32 | Runtime | `om_gemmini_split_{2,3,4}_f32` | Rank-4 f32; 2–4 outputs; any axis |
| `Sigmoid` f32 | Runtime | `om_gemmini_sigmoid_f32` | Rank-4 f32; elementwise; dynamic batch |
| `Mul` f32 | Runtime | `om_gemmini_mul_f32` | Rank-4 f32; same-shape inputs; dynamic batch |

Ops not yet supported: fp16 variants, `Gather`, `Where`, Conv with `group > 1`,
Split with > 4 outputs.

## Hardware Parameters

| Parameter | Value | Source |
|---|---|---|
| Systolic array dimension | `DIM = 16` | `gemmini_params.h` |
| Scratchpad bank rows | `BANK_ROWS = 4096` | `GemminiTargetInfo.hpp` |
| Accumulator rows | `ACC_ROWS = 1024` | `GemminiTargetInfo.hpp` |
| RoCC channel | `XCUSTOM_ACC = 3` → `CUSTOM_3` (`0x7b`) | `gemmini_params.h:7` |
| Dataflow mode | Weight-Stationary (WS) | Gemmini architecture |

## Lowering Pass Pipeline

| Pass | File | Role |
|---|---|---|
| `ONNXToGemmini` | `Conversion/ONNXToGemmini/ONNXToGemmini.cpp` | Intercept supported ops; emit Gemmini dialect or `krnl.call` |
| `GemminiTiling` | `Transform/Gemmini/GemminiTiling.cpp` | Tile direct MatMul to DIM=16; assign ping-pong scratchpad slots |
| `GemminiToGemminiLow` | `Transform/GemminiLow/GemminiToGemminiLow.cpp` | High-level → instruction-level ops |
| `GemminiLowRewrite` | `Transform/GemminiLow/GemminiLowRewrite.cpp` | Peephole: eliminate adjacent redundant config/fence ops |
| `GemminiLowToLLVM` | `Conversion/GemminiLowToLLVM/GemminiLowToLLVM.cpp` | Emit RISC-V CUSTOM_3 inline asm |

## Adding a New Operator

1. Write a failing lit test in `test/mlir/accelerators/gemmini/`.
2. Add a predicate `isSupportedGemmini<Op>F32` in `ONNXToGemmini.cpp`.
3. Add the runtime function in `OMRuntimeGemmini.c`.
4. Add a conversion pattern `ONNX<Op>ToGemminiLowering` (benefit=10).
5. Register in **both** places:
   - `GemminiLegalizationPass::runOnOperation()` → `patterns.insert<>` + `target.addDynamicallyLegalOp<>`
   - `populateONNXToKrnlForGemmini()` → `patterns.insert<>`
6. Write integration test in `test/accelerators/gemmini/integration/`.
7. Write example in `gemmini/examples/` and test in `gemmini/tests/`.

## Compiler Flags

| Flag | Effect |
|---|---|
| `--maccel=Gemmini` | Enable Gemmini pipeline (required) |
| `--EmitMLIR` | Stop after MLIR; write `.onnx.mlir` for inspection |
| `--EmitLLVMIR` | Stop at LLVM dialect MLIR |
| `--mtriple=riscv64-unknown-linux-gnu` | Target RISC-V output |
| `--mcpu=rocket` | RISC-V CPU variant |

Inspect emitted IR:

```sh
# Count Gemmini runtime calls
grep -c 'om_gemmini_' /tmp/model.onnx.mlir

# Check for direct-RoCC asm (CUSTOM_3)
grep -c 'CUSTOM_3\|inline_asm' /tmp/model.onnx.mlir
```
