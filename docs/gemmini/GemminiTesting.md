# Gemmini Testing Guide

## Overview

The backend has four test layers. All must pass before any release.

| Suite | Count | Command |
|---|---|---|
| MLIR lit tests | **41 / 41** | `cmake --build gemmini_toolchain_build --target check-onnx_mlir-accelerators-gemmini` |
| Integration tests | **18 / 18** | `cmake --build gemmini_toolchain_build --target check-gemmini-integration` |
| Example tests | **22 pass, 1 skip** | `bash gemmini/tests/run_all_tests.sh` |
| Model zoo | **9 / 9** | `python3 test/accelerators/gemmini/tools/run_model_zoo.py ...` |

---

## 1. MLIR Lit Tests — 41/41

**Location:** `test/mlir/accelerators/gemmini/`  
**Run:** `cmake --build gemmini_toolchain_build --target check-onnx_mlir-accelerators-gemmini`

Each test pipes MLIR through `onnx-mlir-opt --maccel=Gemmini` and uses
`FileCheck` to assert the expected krnl.call or dialect operation is emitted.

Key tests by category:

| Category | Files |
|---|---|
| MatMul | `matmul_fp32.mlir`, `matmul_fp32_batched.mlir`, `matmul_i8.mlir`, `matmul_fp16.mlir` |
| Direct RoCC | `direct_matmul_rocc.mlir`, `direct_matmul_rocc_edge.mlir`, `direct_matmul_i8_rocc.mlir`, `config_transpose_bits.mlir` |
| Conv / Gemm | `conv2d_gemmini.mlir`, `conv2d_dynamic_batch.mlir`, `conv2d_dynamic_spatial.mlir`, `convtranspose_gemmini.mlir`, `gemm_gemmini.mlir`, `gemm_dynamic_batch.mlir` |
| INT8 | `matmul_i8.mlir` (INT8 MatMulInteger direct-RoCC; QLinearConv and zero-point variants are covered by integration tests) |
| Resize | `resize_nearest.mlir`, `resize_linear.mlir` |
| Pad | `pad_constant.mlir`, `pad_reflect.mlir`, `pad_edge.mlir` |
| Concat | `concat_axis0.mlir`, `concat_axis1.mlir`, `concat_axis2.mlir`, `concat_dynamic_batch.mlir` |
| Slice | `slice_start_end.mlir`, `slice_negative_indices.mlir`, `slice_step.mlir`, `slice_dynamic_batch.mlir` |
| Transpose | `transpose_identity.mlir`, `transpose_nchw_to_nhwc.mlir`, `transpose_swap_hw.mlir`, `transpose_dynamic_batch.mlir` |
| Split | `split_axis1.mlir`, `split_axis2.mlir`, `split_dynamic_batch.mlir`, `split_3_outputs.mlir` |
| Elementwise | `sigmoid_static.mlir`, `sigmoid_dynamic_batch.mlir`, `mul_static.mlir`, `mul_dynamic_batch.mlir` |
| Infrastructure | `gemmini_low_rewrite.mlir`, `gemmini_double_buffer.mlir` |

**Typical RUN line:**

```mlir
// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// CHECK: funcName = "om_gemmini_conv_f32"
// CHECK-NOT: onnx.Conv
```

---

## 2. Integration Tests — 18/18

**Location:** `test/accelerators/gemmini/integration/`  
**Run:** `cmake --build gemmini_toolchain_build --target check-gemmini-integration`

Each test is a Python script that:
1. Generates or loads an ONNX model
2. Compiles with `onnx-mlir --maccel=Gemmini --EmitMLIR`
3. Asserts the expected `om_gemmini_*` symbol appears in the MLIR output

| Test | Operator verified |
|---|---|
| `resnet18_compile_run.py` | Conv, BatchNorm, Relu, MaxPool, GlobalAvgPool, Add, Gemm |
| `mobilenetv2_compile_run.py` | Conv, BatchNorm, Relu, Add |
| `bert_tiny_compile_run.py` | MatMul f32, Gemm f32, Softmax |
| `convtranspose_compile_run.py` | ConvTranspose f32 |
| `resize_compile_run.py` | Resize nearest + linear |
| `pad_compile_run.py` | Pad constant/reflect/edge |
| `concat_compile_run.py` | Concat any axis |
| `slice_compile_run.py` | Slice with steps and negative indices |
| `quantization_roundtrip_run.py` | MatMulInteger scalar + vector zp |
| `simulator_numerical_correctness.py` | Aggregate CI: ResNet-18, MobileNetV2, BERT-tiny on Spike vs ONNX Runtime at 1e-5 |
| `transpose_compile_run.py` | Transpose f32 hardware path |
| `split_compile_run.py` | Split 2-output axis=1 |
| `sigmoid_compile_run.py` | Sigmoid f32 |
| `mul_compile_run.py` | Mul f32 |
| `simulator_matmul_i8_run.py` | Direct-RoCC INT8 MatMul on Spike |
| `simulator_resnet18_run.py` | ResNet-18 on Spike at 1e-5 tolerance |
| `simulator_mobilenetv2_run.py` | MobileNetV2 on Spike at 1e-5 tolerance |
| `simulator_bert_tiny_run.py` | BERT-tiny on Spike at 1e-5 tolerance |

Run one test directly:

```sh
llvm-lit -v test/accelerators/gemmini/integration/sigmoid_compile_run.py
```

---

## 3. Example Tests — 22 pass, 1 skip

**Location:** `gemmini/tests/`  
**Run:** `bash gemmini/tests/run_all_tests.sh [--verbose]`

Shell-based smoke tests organised in four tiers:

| Tier | Tests | What it checks |
|---|---|---|
| **1** — f32 smoke | 01, 11–14, 17, 19-transpose, 20-split, 21-sigmoid, 22-mul | Generate model → compile → check `om_gemmini_*` symbol in MLIR |
| **2** — INT8 | 03, 04, 05, 07, 08 | MatMulInteger (scalar/vector zp), QLinearConv, direct-RoCC MatMul |
| **3** — Networks | 09, 10, 15, 16, 18 | Multi-op compile + link to RV64 ELF |
| **4** — External | 06 (ResNet-18), 19-yolo-real (YOLOX-nano) | SKIP gracefully when model absent |

Each test uses `_shared/run_model_pipeline.sh` to generate the ONNX model
(via a Python generator script), compile it, and grep the emitted MLIR for
the expected runtime function name.

Run a single example:

```sh
bash gemmini/tests/21-sigmoid-f32/run_test.sh
```

---

## 4. Model Zoo Validation — 9/9

**Location:** `test/accelerators/gemmini/tools/run_model_zoo.py`  
**Manifest:** `test/accelerators/gemmini/tools/model_zoo_manifest.json`

```sh
python3 test/accelerators/gemmini/tools/run_model_zoo.py \
  --onnx-mlir gemmini_toolchain_build/Release/bin/onnx-mlir \
  --repo-root . \
  [--download]          # fetch models not cached under ~/.cache/gemmini-model-zoo
  [--spike]             # run spike-feasible models end-to-end
  [--models resnet18 yolov5s]   # subset
  [--report zoo_report.json]
```

Expected output (all 9 PASS):

```
ResNet-18 v1      21 ops  0.82s  PASS
ResNet-50 v1      54 ops  2.72s  PASS
MobileNetV2       36 ops  0.43s  PASS
SqueezeNet 1.1    64 ops  0.18s  PASS
DenseNet-121     426 ops  1.04s  PASS
YOLOX-nano       332 ops  0.15s  PASS  (+Spike end-to-end)
YOLOv5s          201 ops  0.53s  PASS
BERT-tiny         24 ops  0.90s  PASS
Res2Net-101      467 ops  4.19s  PASS
─────────────────────────────────
TOTAL 9  PASS 9  suite 13.3s
```

CMake target (compile-only, no download by default):

```sh
cmake --build gemmini_toolchain_build --target check-gemmini-zoo
```

---

## 5. Numerical Correctness on Spike

```sh
python3 test/accelerators/gemmini/tools/simulator_numerical_correctness.py
```

Compiles ResNet-18, MobileNetV2, and BERT-tiny with `--maccel=Gemmini`, runs
each on Spike+Gemmini, and compares output against ONNX Runtime CPU at
`allclose(atol=1e-5)`. Expected result: `"models_failed": 0`.

---

## 6. Interpreting Results

| What you see in emitted MLIR | Meaning |
|---|---|
| `funcName = "om_gemmini_*"` in a `krnl.call` | Runtime-delegate path was selected |
| `gemmini_low.*` ops | Direct-RoCC path (INT8 MatMul) was selected |
| `llvm.inline_asm` with `CUSTOM_3` | Final lowered RoCC instruction |
| Only `onnx.*` ops remain | Op was not supported; fell through to standard path |

---

## 7. Adding a New Test

**Lit test:**

```mlir
// test/mlir/accelerators/gemmini/my_op.mlir
// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// CHECK: funcName = "om_gemmini_my_op_f32"
// CHECK-NOT: onnx.MyOp
```

**Integration test:**

```python
# test/accelerators/gemmini/integration/my_op_compile_run.py
# RUN: %python %s --onnx-mlir %onnx_mlir_exe --workdir %t.dir
import subprocess, sys
# ... generate model, compile, assert symbol present
```

**Example test:**

```sh
# gemmini/tests/23-my-op/run_test.sh
"$TEST_DIR/../_shared/run_model_pipeline.sh" \
  "$TEST_DIR" "my_op" "generator" \
  "$ROOT_DIR/gemmini/examples/23-my-op/my_op_gen.py"
grep -q 'om_gemmini_my_op_f32' "$TEST_DIR/my_op.onnx.mlir"
```

Add the new test to `gemmini/tests/run_all_tests.sh` under the appropriate tier.
