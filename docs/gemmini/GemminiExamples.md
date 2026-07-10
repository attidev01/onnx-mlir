# Gemmini Examples

## Overview

The backend ships 22 shell-based example tests under `gemmini/tests/`, plus
Python model-generator scripts under `gemmini/tests/<N>-<name>/`.  All 22 pass
as of the current build.  Run the full suite:

```sh
bash gemmini/tests/run_all_tests.sh [--verbose]
```

Run a single example:

```sh
bash gemmini/tests/21-sigmoid-f32/run_test.sh
```

---

## Example Test Map

| # | Directory | Operator(s) | Runtime symbol |
|---|---|---|---|
| 01 | `01-relu-smoke` | Relu f32 | `om_gemmini_relu_f32` |
| 03 | `03-matmulinteger-basic` | MatMulInteger i8→i32 scalar zp | `om_gemmini_matmulinteger_i8i8acc32` |
| 04 | `04-matmulinteger-scalar-zp` | MatMulInteger scalar zero-point | `om_gemmini_matmulinteger_i8i8acc32_zp` |
| 05 | `05-matmulinteger-vector-zp` | MatMulInteger per-row zero-points | `om_gemmini_matmulinteger_i8i8acc32_zp` |
| 06 | `06-resnet-check` | ResNet-18 full model | Conv/BN/Relu/MaxPool/Add/Gemm family |
| 07 | `07-qlinearconv-bias-xzp` | QLinearConv int8 with bias + x zero-point | `om_gemmini_qlinearconv_i8_bias` |
| 08 | `08-direct-matmul-16x16` | Direct-RoCC INT8 MatMul | `gemmini_low.*` / `CUSTOM_3` asm |
| 09 | `09-resnet50-float` | ResNet-50 full model | Conv/BN/Add/Gemm family |
| 10 | `10-squeezenet-float` | SqueezeNet 1.1 | Conv/Relu/MaxPool/GlobalAvgPool |
| 11 | `11-convtranspose-f32` | ConvTranspose f32 | `om_gemmini_convtranspose_f32` |
| 12 | `12-resize-f32` | Resize nearest + bilinear | `om_gemmini_resize_nearest_f32`, `…_linear_f32` |
| 13 | `13-pad-f32` | Pad constant/reflect/edge | `om_gemmini_pad_constant/reflect/edge_f32` |
| 14 | `14-concat-f32` | Concat any axis | `om_gemmini_concat_f32` |
| 15 | `15-resnet18-float` | ResNet-18 recheck | Conv/BN/Relu/MaxPool/Add/Gemm family |
| 16 | `16-bert-tiny` | BERT-tiny | MatMul f32, Gemm f32, Softmax |
| 17 | `17-slice-f32` | Slice with steps and negative indices | `om_gemmini_slice_f32` |
| 18 | `18-yolo-ops` | YOLO utility ops (Sigmoid, Mul) | `om_gemmini_sigmoid_f32`, `om_gemmini_mul_f32` |
| 19 | `19-transpose-f32` | Transpose rank-4 | `om_gemmini_transpose_f32_hw` |
| 19y | `19-yolo-real` | YOLOX-nano end-to-end (Spike) | Full network; skips if model absent |
| 20 | `20-split-f32` | Split 2–4 outputs | `om_gemmini_split_{2,3,4}_f32` |
| 21 | `21-sigmoid-f32` | Sigmoid f32 | `om_gemmini_sigmoid_f32` |
| 22 | `22-mul-f32` | Mul f32 same-shape | `om_gemmini_mul_f32` |

---

## Per-Operator Walkthroughs

### MatMul f32 (runtime delegate)

```sh
ONNX_MLIR=gemmini_toolchain_build/Release/bin/onnx-mlir

# Inspect MLIR output
$ONNX_MLIR --maccel=Gemmini --EmitMLIR test/mlir/accelerators/gemmini/matmul_fp32.mlir \
  -o /tmp/matmul_f32
grep 'funcName' /tmp/matmul_f32.onnx.mlir
# → funcName = "om_gemmini_matmul_f32_hw"
```

For batched MatMul (rank-N × rank-2):

```sh
grep 'funcName' /tmp/matmul_batched.onnx.mlir
# → funcName = "om_gemmini_matmul_f32_nd_hw"
```

### MatMul i8 (direct RoCC path)

```sh
python3 gemmini/tests/08-direct-matmul-16x16/direct_matmul_i32_16x16_gen.py

$ONNX_MLIR --maccel=Gemmini \
  --mtriple=riscv64-unknown-linux-gnu --mcpu=rocket \
  --EmitLLVMIR \
  gemmini/tests/08-direct-matmul-16x16/direct_matmul_i32_16x16.onnx \
  -o /tmp/direct_matmul
grep -c 'CUSTOM_3\|gemmini_low' /tmp/direct_matmul.onnx.mlir
```

Expected: non-zero count of `CUSTOM_3` RoCC instructions.

### Conv f32

```sh
$ONNX_MLIR --maccel=Gemmini --EmitMLIR my_conv_model.onnx -o /tmp/conv_out
grep 'funcName' /tmp/conv_out.onnx.mlir
# → funcName = "om_gemmini_conv_f32" or "om_gemmini_conv_f32_bias"
```

### MatMulInteger (INT8 quantized)

```sh
# Scalar zero-point
python3 gemmini/tests/03-matmulinteger-basic/matmulinteger_i8_gen.py
$ONNX_MLIR --maccel=Gemmini --EmitMLIR \
  gemmini/tests/03-matmulinteger-basic/matmulinteger_i8.onnx \
  -o /tmp/matmulinteger_basic
grep 'funcName' /tmp/matmulinteger_basic.onnx.mlir
# → funcName = "om_gemmini_matmulinteger_i8i8acc32"

# Vector zero-point
python3 gemmini/tests/05-matmulinteger-vector-zp/matmulinteger_i8_vector_zp_gen.py
$ONNX_MLIR --maccel=Gemmini --EmitMLIR \
  gemmini/tests/05-matmulinteger-vector-zp/matmulinteger_i8_vector_zp.onnx \
  -o /tmp/matmulinteger_vzp
grep 'funcName' /tmp/matmulinteger_vzp.onnx.mlir
# → funcName = "om_gemmini_matmulinteger_i8i8acc32_zp"
```

### QLinearConv (INT8 convolution)

```sh
python3 gemmini/tests/07-qlinearconv-bias-xzp/qlinearconv_i8_bias_xzp_gen.py
$ONNX_MLIR --maccel=Gemmini --EmitMLIR \
  gemmini/tests/07-qlinearconv-bias-xzp/qlinearconv_i8_bias_xzp.onnx \
  -o /tmp/qlinearconv
grep 'funcName' /tmp/qlinearconv.onnx.mlir
# → funcName = "om_gemmini_qlinearconv_i8_bias"
```

### Resize, Pad, Concat, Slice

```sh
# Resize (nearest and bilinear)
bash gemmini/tests/12-resize-f32/run_test.sh
grep 'funcName' gemmini/tests/12-resize-f32/resize_*.onnx.mlir

# Pad (constant, reflect, edge modes)
bash gemmini/tests/13-pad-f32/run_test.sh

# Concat (any axis, 2-input)
bash gemmini/tests/14-concat-f32/run_test.sh

# Slice (with negative indices and steps)
bash gemmini/tests/17-slice-f32/run_test.sh
```

### Transpose

```sh
bash gemmini/tests/19-transpose-f32/run_test.sh
grep 'funcName' gemmini/tests/19-transpose-f32/*.onnx.mlir
# → funcName = "om_gemmini_transpose_f32_hw"
```

### Split

```sh
bash gemmini/tests/20-split-f32/run_test.sh
grep 'funcName' gemmini/tests/20-split-f32/*.onnx.mlir
# → funcName = "om_gemmini_split_2_f32"  (or _3_, _4_ for more outputs)
```

### Sigmoid and Mul (SiLU building blocks)

```sh
bash gemmini/tests/21-sigmoid-f32/run_test.sh
bash gemmini/tests/22-mul-f32/run_test.sh
```

In YOLO-family networks SiLU is expressed as `Sigmoid(x) * x`; both ops lower
to `om_gemmini_sigmoid_f32` and `om_gemmini_mul_f32` respectively.

---

## Full Network Examples

### ResNet-18

```sh
$ONNX_MLIR --maccel=Gemmini --EmitMLIR resnet18-v1-7.onnx -o /tmp/resnet18_gemmini

# Count accelerated ops
grep -c 'om_gemmini_' /tmp/resnet18_gemmini.onnx.mlir
# → ~115 Gemmini runtime calls
```

### BERT-tiny

```sh
$ONNX_MLIR --maccel=Gemmini --EmitMLIR bert_tiny.onnx -o /tmp/bert_gemmini
grep -c 'om_gemmini_' /tmp/bert_gemmini.onnx.mlir
# → ~50+ Gemmini runtime calls (MatMul + Gemm + Softmax)
```

### YOLOX-nano

```sh
$ONNX_MLIR --maccel=Gemmini --EmitMLIR yolox_nano.onnx -o /tmp/yolox_gemmini
grep -c 'om_gemmini_' /tmp/yolox_gemmini.onnx.mlir
# → 332 Gemmini runtime calls
```

### YOLOv5s (FP32)

```sh
$ONNX_MLIR --maccel=Gemmini --EmitMLIR yolov5s.onnx -o /tmp/yolov5s_gemmini
grep -c 'om_gemmini_' /tmp/yolov5s_gemmini.onnx.mlir
# → 201 Gemmini runtime calls (Conv, BN, Sigmoid, Mul, ...)
```

---

## Model Zoo Validation

Compile all 9 zoo models and check op counts:

```sh
python3 test/accelerators/gemmini/tools/run_model_zoo.py \
  --onnx-mlir gemmini_toolchain_build/Release/bin/onnx-mlir \
  --repo-root .
```

Expected result (all 9 PASS, ~13 s total):

```
ResNet-18 v1       21 ops  0.82s  PASS
ResNet-50 v1       54 ops  2.72s  PASS
MobileNetV2        36 ops  0.43s  PASS
SqueezeNet 1.1     64 ops  0.18s  PASS
DenseNet-121      426 ops  1.04s  PASS
YOLOX-nano        332 ops  0.15s  PASS
YOLOv5s           201 ops  0.53s  PASS
BERT-tiny          24 ops  0.90s  PASS
Res2Net-101       467 ops  4.19s  PASS
──────────────────────────────────
TOTAL 9  PASS 9  suite 13.3s
```

---

## Running on Spike

After building with simulator mode:

```sh
# Compile to RV64 ELF
$ONNX_MLIR --maccel=Gemmini \
  --mtriple=riscv64-unknown-linux-gnu --mcpu=rocket \
  model.onnx -o model_rv64

# Run on Spike
spike \
  --extension=gemmini \
  --isa=rv64imafdc \
  "$HOME/riscv-gemmini/bin/pk" \
  ./model_rv64
```

For a YOLOX-nano end-to-end test:

```sh
bash gemmini/tests/19-yolo-real/run_test.sh
```

---

## Interpreting Emitted MLIR

| What you see | Meaning |
|---|---|
| `funcName = "om_gemmini_*"` in `krnl.call` | Runtime-delegate path selected |
| `gemmini_low.*` ops | Direct-RoCC path selected (INT8 MatMul tiles) |
| `llvm.inline_asm` with `CUSTOM_3` | Fully lowered RoCC instruction |
| Only `onnx.*` ops remain | Op fell through to standard scalar path |

```sh
# Quick check: how many ops were accelerated?
grep -c 'om_gemmini_' /tmp/your_model.onnx.mlir

# Check that direct-RoCC asm is present (INT8 MatMul only)
grep -c 'CUSTOM_3' /tmp/your_model.onnx.mlir
```

See [GemminiBackend.md](GemminiBackend.md) for the full operator table and path
selection criteria.
