# Gemmini Runtime Functions — Complete Reference

All `om_gemmini_*` functions are declared in
`Runtime/OMRuntimeGemmini.hpp` and implemented in
`Runtime/OMRuntimeGemmini.cpp`.

The compiled model code calls these symbols at runtime on the RISC-V board.

**Important:** These functions are compiled only for RISC-V targets.
The file contains `#error "RISC-V only"` to prevent accidental x86 builds.

---

## How the Model Calls These Functions

The ONNX-MLIR compiler emits `krnl.call` ops in MLIR, which lower to
LLVM `call` instructions pointing at these symbols.

Pattern in MLIR:
```mlir
"krnl.call"(%out, %in, %weights, %bias, %c1_i64, %c0_i64)
  <{funcName = "om_gemmini_conv_f32_bias", numOfOutput = 1 : si64}>
```

Pattern in LLVM IR:
```llvm
call void @om_gemmini_conv_f32_bias(ptr %out, ptr %in, ptr %w, ptr %b,
                                    i64 1, i64 0)
```

The first argument is always the **output tensor**. Input tensors follow.
Integer configuration values come last.

---

## `OMTensor` — the Tensor Wrapper Type

All functions receive tensors as `OMTensor*` pointers.
`OMTensor` is a struct defined in `include/onnx-mlir/Runtime/OMTensor.h`.

It carries:
- A raw data pointer
- Data type (`f32`, `i8`, etc.)
- Rank (number of dimensions)
- Shape array (one value per dimension)
- Strides array (one value per dimension)

Simple example:
```text
OMTensor for a 1×3×416×416 f32 image:
  data_ptr  → raw bytes
  data_type → f32
  rank      → 4
  shape     → [1, 3, 416, 416]
  strides   → [519168, 173056, 416, 1]
```

---

## Section 1: Integer / Quantized Path (Gemmini RoCC Hardware)

These functions use the Gemmini systolic array directly with `int8` data.

---

### `om_gemmini_matmulinteger_i8i8acc32`

```c
void om_gemmini_matmulinteger_i8i8acc32(
    OMTensor *output,
    const OMTensor *lhs,
    const OMTensor *rhs);
```

**Role:** Integer matrix multiply. `int8 × int8 → int32`, no zero-point adjustment.

**When used:** ONNX `MatMulInteger` with no zero-point tensors.

**Simple example:**
```
A (i8): [[1, 2], [3, 4]]
B (i8): [[10, 0], [0, 10]]
C (i32) = A × B = [[10, 20], [30, 40]]
```

---

### `om_gemmini_matmulinteger_i8i8acc32_zp`

```c
void om_gemmini_matmulinteger_i8i8acc32_zp(
    OMTensor *output,
    const OMTensor *lhs,
    const OMTensor *rhs,
    const OMTensor *aZeroPoint,
    const OMTensor *bZeroPoint);
```

**Role:** Integer matrix multiply with per-tensor zero-point correction.

**What is a zero-point?**

In quantization, values are stored as integers but represent floats:
```
real_value = (int_value - zero_point) × scale
```

The zero-point shifts the integer range so zero can be represented exactly.

**Simple example:**
```
A stored as i8 = 128  (but actual value is 128 - 128 = 0 if zero_point=128)
```

This is important for activation tensors in quantized neural networks.

---

### `om_gemmini_qlinearconv_i8`

```c
void om_gemmini_qlinearconv_i8(
    OMTensor *output,
    const OMTensor *input,
    const OMTensor *xScale,    // input quantization scale (scalar)
    const OMTensor *xZeroPoint,// input zero-point (scalar i8)
    const OMTensor *weights,   // i8 filters
    const OMTensor *wScale,    // weight scale (per output channel)
    const OMTensor *wZeroPoint,// weight zero-point (per output channel)
    const OMTensor *yScale,    // output scale (scalar)
    const OMTensor *yZeroPoint,// output zero-point (scalar i8)
    int64_t stride,
    int64_t padding);
```

**Role:** ONNX `QLinearConv` — fully quantized convolution, NCHW layout.
Input, weights, and output are all `int8`. No bias.

---

### `om_gemmini_qlinearconv_i8_bias`

Same as above but with an additional `bias` argument (`int32` per output channel):

```c
void om_gemmini_qlinearconv_i8_bias(
    OMTensor *output, const OMTensor *input,
    const OMTensor *xScale, const OMTensor *xZeroPoint,
    const OMTensor *weights, const OMTensor *wScale,
    const OMTensor *wZeroPoint, const OMTensor *yScale,
    const OMTensor *yZeroPoint,
    const OMTensor *bias,       // ← added
    int64_t stride, int64_t padding);
```

---

## Section 2: Float Path (Quantize → Hardware → Dequantize)

These functions accept `f32` tensors but internally quantize to `i8`,
run the Gemmini systolic array, then dequantize back to `f32`.

```
f32 input → quantize to i8 → Gemmini hardware → dequantize → f32 output
```

This is transparent to the model: it sees only `f32` values.

---

### `om_gemmini_matmul_f32_hw`

```c
void om_gemmini_matmul_f32_hw(
    OMTensor *out,
    const OMTensor *a,  // f32, rank-2
    const OMTensor *b); // f32, rank-2
```

**Role:** `f32` rank-2 matrix multiply via Gemmini hardware path.

**Simple example:**
```
A: [[1.0, 2.0], [3.0, 4.0]]
B: [[1.0, 0.0], [0.0, 1.0]]
C = A × B = [[1.0, 2.0], [3.0, 4.0]]
```

---

### `om_gemmini_matmul_f32_nd_hw`

```c
void om_gemmini_matmul_f32_nd_hw(
    OMTensor *out,
    const OMTensor *a,  // f32, rank ≥ 3
    const OMTensor *b); // f32, rank-2
```

**Role:** Batched matrix multiply for BERT-style transformer operations.

**When used:** The A tensor has extra batch dimensions (e.g., `[batch, seq, hidden]`).
The function treats all leading dimensions as a batch and multiplies each
`[seq × hidden]` slice by the shared B matrix.

**Simple example:**
```
A shape: [2, 3, 4]  (2 batch × 3-token sequences × 4-dim hidden)
B shape: [4, 5]     (shared weight matrix)
C shape: [2, 3, 5]  (2 batch × 3 tokens × 5-dim output)
```

---

### `om_gemmini_matmul_f16_hw`

```c
void om_gemmini_matmul_f16_hw(
    OMTensor *out,
    const OMTensor *a,  // f16
    const OMTensor *b); // f16
```

**Role:** `f16` (half-precision) rank-2 matrix multiply.

---

### `om_gemmini_conv_f32`

```c
void om_gemmini_conv_f32(
    OMTensor *out,
    const OMTensor *x,      // input NCHW f32
    const OMTensor *w,      // weights [out_ch, in_ch, kH, kW] f32
    int64_t stride,
    int64_t pad);
```

**Role:** Conv2D `f32` NCHW, no bias.

**Argument meaning:**
| Argument | Meaning |
|----------|---------|
| `x` | Input feature map, shape `[N, C_in, H, W]`. |
| `w` | Filter weights, shape `[C_out, C_in, kH, kW]`. |
| `stride` | Step size when sliding the filter. `1` = move one pixel, `2` = skip every other. |
| `pad` | Zero-padding added around input. `0` = no padding, `1` = pad 1 pixel each side. |

**Simple example:**
```
input:   1×3×8×8   (1 image, 3 channels, 8×8 pixels)
weights: 16×3×3×3  (16 output filters, each 3×3 applied over 3 channels)
stride=1, pad=1
output:  1×16×8×8  (same spatial size due to padding)
```

---

### `om_gemmini_conv_f32_bias`

```c
void om_gemmini_conv_f32_bias(
    OMTensor *out,
    const OMTensor *x,
    const OMTensor *w,
    const OMTensor *b,      // ← per-output-channel bias [C_out]
    int64_t stride,
    int64_t pad);
```

**Role:** Same as `om_gemmini_conv_f32` but adds a bias value to each
output channel after the convolution.

**Simple example:**
```
After convolution, channel 0 result = 5.0
bias[0] = 0.5
final output for channel 0 = 5.5
```

---

### `om_gemmini_convtranspose_f32` / `_bias`

```c
void om_gemmini_convtranspose_f32(
    OMTensor *out, const OMTensor *x, const OMTensor *w,
    int64_t stride, int64_t pad, int64_t output_pad);

void om_gemmini_convtranspose_f32_bias(
    OMTensor *out, const OMTensor *x, const OMTensor *w, const OMTensor *b,
    int64_t stride, int64_t pad, int64_t output_pad);
```

**Role:** Transposed convolution (also called "deconvolution"). Used to
**upsample** feature maps. Common in decoders and segmentation networks.

**Simple example:**
```
input:  1×64×13×13  (downsampled feature map)
output: 1×64×26×26  (upsampled by stride=2)
```

---

## Section 3: Element-wise and Normalization (CPU Scalar Loops)

These run on the RISC-V scalar core, not the Gemmini systolic array.
They are simple enough that a scalar loop is fast enough.

---

### `om_gemmini_relu_f32`

```c
void om_gemmini_relu_f32(OMTensor *out, const OMTensor *x);
```

**Role:** Standalone ReLU — apply `max(0, x)` to every element on the CPU scalar core.

> **Important:** ReLU also exists as a **fused Gemmini hardware activation**.
> When ReLU immediately follows Conv / MatMul / Gemm, the compiler sets
> `Activation::RELU` in the hardware config instruction and ReLU runs inside
> the systolic array at no extra cycle cost. `om_gemmini_relu_f32` is only
> called when ReLU appears as an isolated standalone op that cannot be fused.

**Simple example — two cases:**
```
Case 1 (fused, zero extra cost):
  Conv → ReLU   →  compiler emits gemmini.config{activation=RELU}
                   result is clipped to ≥0 by the hardware itself

Case 2 (standalone, scalar fallback):
  Add → ReLU    →  compiler emits krnl.call om_gemmini_relu_f32
                   CPU walks every element and applies max(0, x)
```

```
input:  [-1.0, 0.0, 2.0, -0.5]
output: [ 0.0, 0.0, 2.0,  0.0]
```

---

### `om_gemmini_batchnorm_f32`

```c
void om_gemmini_batchnorm_f32(
    OMTensor *out,
    const OMTensor *x,
    const OMTensor *scale,      // per-channel learned scale (γ)
    const OMTensor *bias,       // per-channel learned bias (β)
    const OMTensor *mean,       // per-channel running mean
    const OMTensor *var,        // per-channel running variance
    int64_t epsilonBits);       // small constant to avoid division by zero
```

**Role:** Batch normalization (inference mode).

**Formula per element:**
```
y = scale × (x - mean) / sqrt(var + ε) + bias
```

**Simple example:**
```
x = 10.0, mean = 8.0, var = 4.0, ε = 0.001
scale = 1.0, bias = 0.0
y = 1.0 × (10 - 8) / sqrt(4 + 0.001) + 0 ≈ 1.0
```

---

### `om_gemmini_add_f32`

```c
void om_gemmini_add_f32(
    OMTensor *out,
    const OMTensor *a,
    const OMTensor *b);
```

**Role:** Element-wise addition (broadcast-compatible).

```
a: [1, 2, 3]
b: [10, 20, 30]
out: [11, 22, 33]
```

Used for **residual connections** in ResNet-style networks:
```
output = F(x) + x
```

---

### `om_gemmini_sigmoid_f32`

```c
void om_gemmini_sigmoid_f32(OMTensor *out, const OMTensor *x);
```

**Role:** Apply sigmoid to every element: `1 / (1 + exp(-x))`.

```
x:   [-2.0,  0.0,  2.0]
out: [0.119, 0.5, 0.881]
```

Output is always in `[0, 1]`. Used in YOLO for confidence scores.

---

### `om_gemmini_mul_f32`

```c
void om_gemmini_mul_f32(
    OMTensor *out,
    const OMTensor *a,
    const OMTensor *b);
```

**Role:** Element-wise multiplication (broadcast-compatible).

```
a: [2, 3, 4]
b: [0.5, 0.5, 0.5]
out: [1.0, 1.5, 2.0]
```

Often used after sigmoid to implement **SiLU/Swish** activation:
```
SiLU(x) = x × sigmoid(x)
```

---

### `om_gemmini_softmax_f32`

```c
void om_gemmini_softmax_f32(
    OMTensor *out,
    const OMTensor *x,
    int64_t batch,
    int64_t classes);
```

**Role:** Softmax over the `classes` axis.

**Formula:**
```
softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
```

**Simple example:**
```
x:   [1.0, 2.0, 3.0]
out: [0.09, 0.24, 0.67]   (sums to 1.0)
```

Turns raw scores into probabilities.

---

## Section 4: Pooling (CPU Scalar Loops)

---

### `om_gemmini_globalavgpool_f32`

```c
void om_gemmini_globalavgpool_f32(OMTensor *out, const OMTensor *x);
```

**Role:** Average all values in each channel down to a single number.

```
input:  1×64×7×7   (64 channels, 7×7 spatial)
output: 1×64×1×1   (one average per channel)
```

Used at the end of ResNet before the classifier.

---

### `om_gemmini_maxpool_f32`

```c
void om_gemmini_maxpool_f32(
    OMTensor *out,
    const OMTensor *x,
    int64_t kernel,  // window size (e.g. 3 for 3×3)
    int64_t stride,  // step size
    int64_t pad);    // padding
```

**Role:** Max pooling — keep the largest value in each window.

```
kernel patch:
  1  5
  2  3
max = 5
```

---

### `om_gemmini_avgpool_f32`

```c
void om_gemmini_avgpool_f32(
    OMTensor *out,
    const OMTensor *x,
    int64_t kernel,
    int64_t stride,
    int64_t pad,
    int64_t countIncludePad);  // 1 = include padding zeros in average
```

**Role:** Average pooling — compute the mean of each window.

---

## Section 5: Linear Algebra (Gemmini Hardware)

---

### `om_gemmini_gemm_f32`

```c
void om_gemmini_gemm_f32(
    OMTensor *out,
    const OMTensor *a,
    const OMTensor *b,
    int64_t transA,     // 1 = transpose A
    int64_t transB,     // 1 = transpose B
    int64_t alphaBits,  // float scale for A×B (encoded as int bits)
    int64_t betaBits);  // float scale for C (0 if no bias)
```

**Role:** ONNX `Gemm` operator: `C = alpha × A × B + beta × C`.

**Why alphaBits / betaBits?**
Float values `alpha` and `beta` are passed as their raw IEEE-754 bit
patterns cast to `int64_t`. This avoids floating-point ABI issues.

**Simple example (identity case):**
```
alpha = 1.0, beta = 0.0, no transpose
C = A × B
```

---

### `om_gemmini_gemm_f32_bias`

```c
void om_gemmini_gemm_f32_bias(
    OMTensor *out,
    const OMTensor *a,
    const OMTensor *b,
    const OMTensor *c,  // bias matrix
    int64_t transA, int64_t transB,
    int64_t alphaBits, int64_t betaBits);
```

**Role:** Same as `om_gemmini_gemm_f32` but includes an explicit C bias matrix.

---

## Section 6: Spatial Operations (CPU Scalar Loops)

---

### `om_gemmini_resize_nearest_f32`

```c
void om_gemmini_resize_nearest_f32(
    OMTensor *out,
    const OMTensor *x,
    int64_t coord_mode,    // coordinate transformation mode
    int64_t nearest_mode); // rounding mode for nearest-neighbor
```

**Role:** Upsample using nearest-neighbor interpolation.

```
input:  [A, B]       (width=2)
output: [A, A, B, B] (width=4, scale=2)
```

---

### `om_gemmini_resize_linear_f32`

```c
void om_gemmini_resize_linear_f32(
    OMTensor *out,
    const OMTensor *x,
    int64_t coord_mode);
```

**Role:** Upsample using bilinear interpolation (smoother than nearest).

```
input:  [0.0, 1.0]        (width=2)
output: [0.0, 0.33, 0.67, 1.0]  (width=4, blended)
```

---

### `om_gemmini_pad_constant_f32` / `_reflect_f32` / `_edge_f32`

```c
void om_gemmini_pad_constant_f32(
    OMTensor *out, const OMTensor *x,
    int64_t pad_left, int64_t pad_right,
    int64_t pad_top,  int64_t pad_bottom);
```

**Role:** Add border pixels to a feature map.

| Mode | Fills border with |
|------|-------------------|
| `constant` | zeros |
| `reflect` | mirror of interior values |
| `edge` | copy of nearest edge value |

**Simple example (constant, pad_left=1, pad_right=1):**
```
input:  [1, 2, 3]
output: [0, 1, 2, 3, 0]
```

---

### `om_gemmini_slice_f32`

```c
void om_gemmini_slice_f32(
    OMTensor *out, const OMTensor *x,
    int64_t sn, int64_t en, int64_t tn,   // N dim: start, end, step
    int64_t sc, int64_t ec, int64_t tc,   // C dim
    int64_t sh, int64_t eh, int64_t th,   // H dim
    int64_t sw, int64_t ew, int64_t tw);  // W dim
```

**Role:** Extract a sub-tensor by specifying start/end/step per dimension.

**Simple example (step=2 on H):**
```
input:  1×3×4×4  (4 rows)
sh=0, eh=4, th=2  → take rows 0, 2  (every 2nd row)
output: 1×3×2×4
```

---

### `om_gemmini_concat_f32`

```c
void om_gemmini_concat_f32(
    OMTensor *out,
    const OMTensor *x0,
    const OMTensor *x1,
    int64_t axis);
```

**Role:** Concatenate two tensors along `axis`.

**Simple example (axis=1, channel concat):**
```
x0: 1×16×52×52
x1: 1×16×52×52
out: 1×32×52×52   (channels joined)
```

---

### `om_gemmini_transpose_f32_hw`

```c
void om_gemmini_transpose_f32_hw(
    OMTensor *out, const OMTensor *x,
    int64_t perm0, int64_t perm1,
    int64_t perm2, int64_t perm3);
```

**Role:** Standalone Transpose — reorder tensor dimensions on the CPU scalar core.

> **Important:** Transpose also exists as a **fused Gemmini hardware flag**.
> When a Transpose of the A or B matrix directly feeds a MatMul/Gemm, the
> compiler sets `a_transpose=true` or `b_transpose=true` in `gemmini.config`
> and the hardware reads the matrix in transposed order during the mvin step —
> zero extra cycles, no function call. `om_gemmini_transpose_f32_hw` is only
> called when Transpose appears as a standalone op that cannot be fused.

**Simple example — two cases:**
```
Case 1 (fused, zero extra cost):
  Transpose(B) → MatMul   →  compiler sets gemmini.config{b_transpose=true}
                              hardware reads B columns as rows during mvin

Case 2 (standalone, scalar fallback):
  Transpose(X) → Concat   →  compiler calls om_gemmini_transpose_f32_hw
                              CPU reorders every element in memory
```

**Permutation example (NCHW → NHWC: perm = [0,2,3,1]):**
```
input:  1×3×4×4  (channels first)
perm:   [0,2,3,1]
output: 1×4×4×3  (channels last)
```

---

### `om_gemmini_split_2_f32` / `_3_f32` / `_4_f32`

```c
void om_gemmini_split_2_f32(
    OMTensor *out0, OMTensor *out1,
    const OMTensor *x, int64_t axis);
```

**Role:** Split one tensor into 2 (or 3 or 4) equal parts along `axis`.
Opposite of concat.

**Simple example (axis=1):**
```
x:    1×128×52×52
out0: 1×64×52×52
out1: 1×64×52×52
```

---

## Quick Reference Table

| Function | HW path | ONNX op |
|----------|---------|---------|
| `om_gemmini_matmulinteger_i8i8acc32` | Gemmini HW | MatMulInteger |
| `om_gemmini_matmulinteger_i8i8acc32_zp` | Gemmini HW | MatMulInteger |
| `om_gemmini_qlinearconv_i8` | Gemmini HW | QLinearConv |
| `om_gemmini_qlinearconv_i8_bias` | Gemmini HW | QLinearConv |
| `om_gemmini_matmul_f32_hw` | Gemmini HW | MatMul |
| `om_gemmini_matmul_f32_nd_hw` | Gemmini HW | MatMul (batch) |
| `om_gemmini_matmul_f16_hw` | Gemmini HW | MatMul (f16) |
| `om_gemmini_conv_f32` | Gemmini HW | Conv |
| `om_gemmini_conv_f32_bias` | Gemmini HW | Conv |
| `om_gemmini_convtranspose_f32` | Gemmini HW | ConvTranspose |
| `om_gemmini_convtranspose_f32_bias` | Gemmini HW | ConvTranspose |
| `om_gemmini_gemm_f32` | Gemmini HW | Gemm |
| `om_gemmini_gemm_f32_bias` | Gemmini HW | Gemm |
| `om_gemmini_relu_f32` | CPU scalar (standalone only) | Relu |
| *(fused activation)* | Gemmini HW (no extra call) | Relu fused after Conv/MatMul/Gemm |
| `om_gemmini_batchnorm_f32` | CPU scalar | BatchNormalization |
| `om_gemmini_add_f32` | CPU scalar | Add |
| `om_gemmini_sigmoid_f32` | CPU scalar | Sigmoid |
| `om_gemmini_mul_f32` | CPU scalar | Mul |
| `om_gemmini_softmax_f32` | CPU scalar | Softmax |
| `om_gemmini_globalavgpool_f32` | CPU scalar | GlobalAveragePool |
| `om_gemmini_maxpool_f32` | CPU scalar | MaxPool |
| `om_gemmini_avgpool_f32` | CPU scalar | AveragePool |
| `om_gemmini_resize_nearest_f32` | CPU scalar | Resize (nearest) |
| `om_gemmini_resize_linear_f32` | CPU scalar | Resize (linear) |
| `om_gemmini_pad_constant_f32` | CPU scalar | Pad (constant) |
| `om_gemmini_pad_reflect_f32` | CPU scalar | Pad (reflect) |
| `om_gemmini_pad_edge_f32` | CPU scalar | Pad (edge) |
| `om_gemmini_slice_f32` | CPU scalar | Slice |
| `om_gemmini_concat_f32` | CPU scalar | Concat |
| `om_gemmini_transpose_f32_hw` | CPU scalar (standalone only) | Transpose |
| *(fused flag)* | Gemmini HW (no extra call) | Transpose of A/B fused into MatMul/Gemm via `a_transpose`/`b_transpose` |
| `om_gemmini_split_2_f32` | CPU scalar | Split (2 outputs) |
| `om_gemmini_split_3_f32` | CPU scalar | Split (3 outputs) |
| `om_gemmini_split_4_f32` | CPU scalar | Split (4 outputs) |
