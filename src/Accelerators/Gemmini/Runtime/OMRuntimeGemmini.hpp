/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ OMRuntimeGemmini.hpp ------------------------------===//
//
// Public C ABI declarations for all Gemmini runtime entry points.
//
// Compiled models call these symbols via `krnl.call` nodes emitted by
// ONNXToGemmini.cpp.  The implementations live in OMRuntimeGemmini.cpp
// and are compiled only for RISC-V targets (guarded by #error).
//
// All symbols are C-linkage so they link without name-mangling from both
// C and C++ translation units.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OM_RUNTIME_GEMMINI_HPP
#define ONNX_MLIR_OM_RUNTIME_GEMMINI_HPP

#include "include/onnx-mlir/Runtime/OMTensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// ===--- Integer / quantized (Gemmini RoCC hardware) ---===

/** MatMulInteger i8×i8 → i32, no zero-points. */
void om_gemmini_matmulinteger_i8i8acc32(
    OMTensor *output, const OMTensor *lhs, const OMTensor *rhs);

/** MatMulInteger i8×i8 → i32 with per-tensor or per-row zero-points. */
void om_gemmini_matmulinteger_i8i8acc32_zp(OMTensor *output,
    const OMTensor *lhs, const OMTensor *rhs,
    const OMTensor *aZeroPoint, const OMTensor *bZeroPoint);

/** QLinearConv NCHW i8, no bias. */
void om_gemmini_qlinearconv_i8(OMTensor *output, const OMTensor *input,
    const OMTensor *xScale, const OMTensor *xZeroPoint,
    const OMTensor *weights, const OMTensor *wScale,
    const OMTensor *wZeroPoint, const OMTensor *yScale,
    const OMTensor *yZeroPoint, int64_t stride, int64_t padding);

/** QLinearConv NCHW i8 with bias (i32 per output channel). */
void om_gemmini_qlinearconv_i8_bias(OMTensor *output, const OMTensor *input,
    const OMTensor *xScale, const OMTensor *xZeroPoint,
    const OMTensor *weights, const OMTensor *wScale,
    const OMTensor *wZeroPoint, const OMTensor *yScale,
    const OMTensor *yZeroPoint, const OMTensor *bias,
    int64_t stride, int64_t padding);

// ===--- Float (quantize to i8 path then dequantize) ---===

/** MatMul f32 rank-2 via Gemmini i8 hardware path. */
void om_gemmini_matmul_f32_hw(
    OMTensor *out, const OMTensor *a, const OMTensor *b);

/** MatMul f32 rank-N×rank-2 (BERT-style batched, N≥3). */
void om_gemmini_matmul_f32_nd_hw(
    OMTensor *out, const OMTensor *a, const OMTensor *b);

/** MatMul f16 rank-2 via Gemmini i8 hardware path. */
void om_gemmini_matmul_f16_hw(
    OMTensor *out, const OMTensor *a, const OMTensor *b);

/** Conv2D f32 NCHW, no bias. */
void om_gemmini_conv_f32(OMTensor *out, const OMTensor *x,
    const OMTensor *w, int64_t stride, int64_t pad);

/** Conv2D f32 NCHW with bias. */
void om_gemmini_conv_f32_bias(OMTensor *out, const OMTensor *x,
    const OMTensor *w, const OMTensor *b, int64_t stride, int64_t pad);

/** ConvTranspose f32 NCHW, no bias. */
void om_gemmini_convtranspose_f32(OMTensor *out, const OMTensor *x,
    const OMTensor *w, int64_t stride, int64_t pad, int64_t output_pad);

/** ConvTranspose f32 NCHW with bias. */
void om_gemmini_convtranspose_f32_bias(OMTensor *out, const OMTensor *x,
    const OMTensor *w, const OMTensor *b, int64_t stride, int64_t pad,
    int64_t output_pad);

// ===--- Element-wise and normalization (CPU scalar loops) ---===

/** ReLU f32 element-wise. */
void om_gemmini_relu_f32(OMTensor *out, const OMTensor *x);

/** BatchNormalization f32 (inference: scale, bias, mean, var). */
void om_gemmini_batchnorm_f32(OMTensor *out, const OMTensor *x,
    const OMTensor *scale, const OMTensor *bias,
    const OMTensor *mean, const OMTensor *var, int64_t epsilonBits);

/** Add f32 element-wise (broadcast-compatible). */
void om_gemmini_add_f32(OMTensor *out, const OMTensor *a, const OMTensor *b);

/** Sigmoid f32 element-wise. */
void om_gemmini_sigmoid_f32(OMTensor *out, const OMTensor *x);

/** Mul f32 element-wise (broadcast-compatible). */
void om_gemmini_mul_f32(OMTensor *out, const OMTensor *a, const OMTensor *b);

/** Softmax f32 along classes axis. */
void om_gemmini_softmax_f32(OMTensor *out, const OMTensor *x,
    int64_t batch, int64_t classes);

// ===--- Pooling (CPU scalar loops) ---===

/** GlobalAveragePool f32 NCHW. */
void om_gemmini_globalavgpool_f32(OMTensor *out, const OMTensor *x);

/** MaxPool f32 NCHW (square kernel). */
void om_gemmini_maxpool_f32(OMTensor *out, const OMTensor *x,
    int64_t kernel, int64_t stride, int64_t pad);

/** AveragePool f32 NCHW (square kernel). */
void om_gemmini_avgpool_f32(OMTensor *out, const OMTensor *x,
    int64_t kernel, int64_t stride, int64_t pad, int64_t countIncludePad);

// ===--- Linear algebra (Gemmini hardware) ---===

/** Gemm f32 (optional transpose A/B, alpha/beta encoded as f32 bits). */
void om_gemmini_gemm_f32(OMTensor *out, const OMTensor *a, const OMTensor *b,
    int64_t transA, int64_t transB, int64_t alphaBits, int64_t betaBits);

/** Gemm f32 with C bias matrix. */
void om_gemmini_gemm_f32_bias(OMTensor *out, const OMTensor *a,
    const OMTensor *b, const OMTensor *c, int64_t transA, int64_t transB,
    int64_t alphaBits, int64_t betaBits);

// ===--- Spatial (CPU scalar loops) ---===

/** Resize f32 nearest-neighbor (NCHW). */
void om_gemmini_resize_nearest_f32(OMTensor *out, const OMTensor *x,
    int64_t coord_mode, int64_t nearest_mode);

/** Resize f32 bilinear (NCHW). */
void om_gemmini_resize_linear_f32(OMTensor *out, const OMTensor *x,
    int64_t coord_mode);

/** Pad f32 constant-zero mode (NCHW, left/right/top/bottom). */
void om_gemmini_pad_constant_f32(OMTensor *out, const OMTensor *x,
    int64_t pad_left, int64_t pad_right,
    int64_t pad_top, int64_t pad_bottom);

/** Pad f32 reflect mode (NCHW). */
void om_gemmini_pad_reflect_f32(OMTensor *out, const OMTensor *x,
    int64_t pad_left, int64_t pad_right,
    int64_t pad_top, int64_t pad_bottom);

/** Pad f32 edge mode (NCHW). */
void om_gemmini_pad_edge_f32(OMTensor *out, const OMTensor *x,
    int64_t pad_left, int64_t pad_right,
    int64_t pad_top, int64_t pad_bottom);

/** Slice f32 rank-4 (N/C/H/W per-dim start/end/step). */
void om_gemmini_slice_f32(OMTensor *out, const OMTensor *x,
    int64_t sn, int64_t en, int64_t tn,
    int64_t sc, int64_t ec, int64_t tc,
    int64_t sh, int64_t eh, int64_t th,
    int64_t sw, int64_t ew, int64_t tw);

/** Concat f32 2-input NCHW along axis. */
void om_gemmini_concat_f32(OMTensor *out,
    const OMTensor *x0, const OMTensor *x1, int64_t axis);

/** Transpose f32 rank-4 with explicit permutation. */
void om_gemmini_transpose_f32_hw(OMTensor *out, const OMTensor *x,
    int64_t perm0, int64_t perm1, int64_t perm2, int64_t perm3);

/** Split f32 into 2 outputs along axis. */
void om_gemmini_split_2_f32(OMTensor *out0, OMTensor *out1,
    const OMTensor *x, int64_t axis);

/** Split f32 into 3 outputs along axis. */
void om_gemmini_split_3_f32(OMTensor *out0, OMTensor *out1, OMTensor *out2,
    const OMTensor *x, int64_t axis);

/** Split f32 into 4 outputs along axis. */
void om_gemmini_split_4_f32(OMTensor *out0, OMTensor *out1,
    OMTensor *out2, OMTensor *out3, const OMTensor *x, int64_t axis);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ONNX_MLIR_OM_RUNTIME_GEMMINI_HPP
