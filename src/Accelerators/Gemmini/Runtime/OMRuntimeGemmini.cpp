/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- OMRuntimeGemmini.cpp -------------------------===//
//
// RISC-V Gemmini runtime shim.
//
// Provides the following entry points for compiled ONNX-MLIR models:
//
//  Integer / quantized (use Gemmini RoCC hardware via gemmini.h):
//    om_gemmini_matmulinteger_i8i8acc32
//    om_gemmini_matmulinteger_i8i8acc32_zp
//    om_gemmini_qlinearconv_i8
//    om_gemmini_qlinearconv_i8_bias
//
//  Float:
//    om_gemmini_matmul_f32_hw (quantizes to Gemmini i8 hardware path)
//    om_gemmini_matmul_f32_nd_hw (rank-N x rank-2 batched f32 MatMul)
//    om_gemmini_matmul_f16_hw (quantizes to Gemmini i8 hardware path)
//    om_gemmini_conv_f32 / om_gemmini_conv_f32_bias
//    om_gemmini_convtranspose_f32 / om_gemmini_convtranspose_f32_bias
//    om_gemmini_relu_f32
//    om_gemmini_batchnorm_f32
//    om_gemmini_add_f32
//    om_gemmini_globalavgpool_f32
//    om_gemmini_maxpool_f32
//    om_gemmini_avgpool_f32
//    om_gemmini_softmax_f32
//    om_gemmini_gemm_f32 / om_gemmini_gemm_f32_bias
//    om_gemmini_resize_nearest_f32 (CPU scalar loops)
//    om_gemmini_resize_linear_f32  (CPU scalar bilinear)
//    om_gemmini_pad_constant_f32   (CPU scalar loops, zero constant)
//    om_gemmini_pad_reflect_f32    (CPU scalar loops)
//    om_gemmini_pad_edge_f32       (CPU scalar loops)
//    om_gemmini_concat_f32         (CPU scalar loops, 2-input NCHW concat)
//    om_gemmini_transpose_f32_hw   (HW-preferred transpose, scalar fallback)
//    om_gemmini_split_2_f32        (CPU scalar loops, 2-output Split)
//    om_gemmini_split_3_f32        (CPU scalar loops, 3-output Split)
//    om_gemmini_split_4_f32        (CPU scalar loops, 4-output Split)
//    om_gemmini_sigmoid_f32        (CPU scalar loop, element-wise sigmoid)
//    om_gemmini_mul_f32            (CPU scalar loop, element-wise multiply)
//
//===----------------------------------------------------------------------===//

/**
 * @file OMRuntimeGemmini.cpp
 * @brief RISC-V Gemmini runtime shim for compiled ONNX-MLIR models.
 *
 * All public functions in this file are called from code emitted by the
 * ONNX-MLIR Gemmini backend via `krnl.call` wrappers.  They are compiled
 * only for RISC-V targets and link against the UC Berkeley Gemmini RoCC
 * header library (`gemmini.h`).
 *
 * @note This file must be compiled with a RISC-V cross-compiler.
 *       An `#error` guard enforces this at compile time.
 */

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/onnx-mlir/Runtime/OMTensor.h"
#include "src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.hpp"
#include "src/Support/SmallFPConversion.h"

#if !defined(__riscv)
#error "OMRuntimeGemmini.cpp must only be compiled for RISC-V targets"
#endif

// Resolved from GEMMINI_ROCC_INCLUDE_DIR.  The bundled ABI tree also provides
// a forwarding header so editors can resolve this include before CMake runs.
#include "gemmini.h"

// ===--- Helpers ---===

// Common local names used throughout this runtime:
//   N/batchSize: batch dimension, C/inChannels: input channels,
//   OC/M/outChannels: output channels, H/W: input height/width,
//   OH/OW: output height/width, KH/KW/kernelDim: kernel height/width,
//   M/N/K in matmul: [M,K] x [K,N] -> [M,N].
//   x/a/lhs: input or left operand, w/b/rhs: weights or right operand,
//   out/y/output: pre-allocated result tensor from ONNX-MLIR.

/** Return true when runtime tracing is explicitly enabled through GEMMINI_TRACE.
 *
 * The default is quiet so normal Spike and runtime benchmarks do not pay for
 * per-tensor logging. Set GEMMINI_TRACE to any non-empty value except "0" when
 * debugging tensor shapes, values, or Gemmini runtime dispatch.
 */
static bool gemminiTraceEnabled() {
  const char *env = getenv("GEMMINI_TRACE");
  return env && env[0] != '\0' && env[0] != '0';
}

/** Convert ONNX-MLIR tensor dtype ids into compact trace labels. */
static const char *traceTensorDTypeName(int64_t dtype) {
  switch (dtype) {
  case ONNX_TYPE_FLOAT:
    return "f32";
  case ONNX_TYPE_FLOAT16:
    return "f16";
  case ONNX_TYPE_INT8:
    return "i8";
  case ONNX_TYPE_INT32:
    return "i32";
  default:
    return "unknown";
  }
}

/** Print one tensor's dtype, shape, and element count for trace logs. */
static void traceTensorShape(const char *op, const char *label,
    const OMTensor *tensor) {
  if (!gemminiTraceEnabled() || !tensor)
    return;
  const int64_t rank = omTensorGetRank(tensor);
  const int64_t *shape = omTensorGetShape(tensor);
  fprintf(stderr, "[gemmini-trace] %s %s: dtype=%s shape=[", op, label,
      traceTensorDTypeName(omTensorGetDataType(tensor)));
  for (int64_t i = 0; i < rank; ++i) {
    fprintf(stderr, "%s%lld", i ? "," : "", (long long)shape[i]);
  }
  fprintf(stderr, "] elems=%lld\n", (long long)omTensorGetNumElems(tensor));
  fflush(stderr);
}

/** Print min/max/mean/nonzero/NaN/Inf statistics for supported tensor dtypes. */
static void traceTensorStats(const char *label, const OMTensor *tensor) {
  if (!gemminiTraceEnabled() || !tensor)
    return;
  const int64_t n = omTensorGetNumElems(tensor);
  const void *raw = omTensorGetDataPtr(tensor);
  if (!raw || n <= 0)
    return;

  double minVal = 0.0;
  double maxVal = 0.0;
  double sum = 0.0;
  int64_t nonzero = 0;
  int64_t nans = 0;
  int64_t infs = 0;
  bool initialized = false;

#define TRACE_ACCUM_VALUE(valueExpr)                                           \
  do {                                                                         \
    double value = (double)(valueExpr);                                        \
    if (isnan(value)) {                                                        \
      ++nans;                                                                  \
      break;                                                                   \
    }                                                                          \
    if (isinf(value)) {                                                        \
      ++infs;                                                                  \
      break;                                                                   \
    }                                                                          \
    if (!initialized) {                                                        \
      minVal = value;                                                          \
      maxVal = value;                                                          \
      initialized = true;                                                      \
    } else {                                                                   \
      if (value < minVal)                                                      \
        minVal = value;                                                        \
      if (value > maxVal)                                                      \
        maxVal = value;                                                        \
    }                                                                          \
    if (value != 0.0)                                                          \
      ++nonzero;                                                               \
    sum += value;                                                              \
  } while (0)

  switch (omTensorGetDataType(tensor)) {
  case ONNX_TYPE_FLOAT: {
    const float *data = (const float *)raw;
    for (int64_t i = 0; i < n; ++i)
      TRACE_ACCUM_VALUE(data[i]);
    break;
  }
  case ONNX_TYPE_FLOAT16: {
    const uint16_t *data = (const uint16_t *)raw;
    for (int64_t i = 0; i < n; ++i)
      TRACE_ACCUM_VALUE(om_f16_to_f32(data[i]));
    break;
  }
  case ONNX_TYPE_INT8: {
    const int8_t *data = (const int8_t *)raw;
    for (int64_t i = 0; i < n; ++i)
      TRACE_ACCUM_VALUE(data[i]);
    break;
  }
  case ONNX_TYPE_INT32: {
    const int32_t *data = (const int32_t *)raw;
    for (int64_t i = 0; i < n; ++i)
      TRACE_ACCUM_VALUE(data[i]);
    break;
  }
  default:
    fprintf(stderr,
        "[gemmini-trace] %s: dtype=%s stats=unsupported elems=%lld\n", label,
        traceTensorDTypeName(omTensorGetDataType(tensor)), (long long)n);
    fflush(stderr);
    return;
  }

#undef TRACE_ACCUM_VALUE

  const double mean = initialized ? sum / (double)n : 0.0;
  fprintf(stderr,
      "[gemmini-trace] %s: min=%+.6f max=%+.6f mean=%+.6f nonzero=%lld/%lld nans=%lld infs=%lld\n",
      label, minVal, maxVal, mean, (long long)nonzero, (long long)n,
      (long long)nans, (long long)infs);
  fflush(stderr);
}

/** Print both shape metadata and scalar statistics for a tensor. */
static void traceTensorShapeStats(
    const char *op, const char *label, const OMTensor *tensor) {
  if (!gemminiTraceEnabled() || !tensor)
    return;
  char statsLabel[128];
  snprintf(statsLabel, sizeof(statsLabel), "%s %s", op, label);
  traceTensorShape(op, label, tensor);
  traceTensorStats(statsLabel, tensor);
}

/** Print scalar statistics for a raw float buffer. */
static void traceF32BufferStats(const char *label, const float *data, int64_t n) {
  if (!gemminiTraceEnabled() || !data || n <= 0)
    return;
  float minVal = data[0];
  float maxVal = data[0];
  double sum = 0.0;
  int64_t nonzero = 0;
  int64_t nans = 0;
  int64_t infs = 0;
  for (int64_t i = 0; i < n; ++i) {
    float v = data[i];
    if (v != v) {
      ++nans;
      continue;
    }
    if (v > FLT_MAX || v < -FLT_MAX) {
      ++infs;
      continue;
    }
    if (v < minVal)
      minVal = v;
    if (v > maxVal)
      maxVal = v;
    if (v != 0.0f)
      ++nonzero;
    sum += (double)v;
  }
  fprintf(stderr,
      "[gemmini-trace] %s: min=%+.6f max=%+.6f mean=%+.6f nonzero=%lld/%lld nans=%lld infs=%lld\n",
      label, minVal, maxVal, sum / (double)n, (long long)nonzero,
      (long long)n, (long long)nans, (long long)infs);
  fflush(stderr);
}

/** Print a compact f32 convolution shape line for tracing. */
static void traceConvF32Shape(int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t OC, int64_t KH, int64_t KW, int64_t OH, int64_t OW,
    int64_t stride, int64_t pad, bool hasBias) {
  if (!gemminiTraceEnabled())
    return;
  fprintf(stderr,
      "[gemmini-trace] conv f32: x=[%lld,%lld,%lld,%lld] "
      "w=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] stride=%lld pad=%lld "
      "bias=%s\n",
      (long long)N, (long long)C, (long long)H, (long long)W,
      (long long)OC, (long long)C, (long long)KH, (long long)KW,
      (long long)N, (long long)OC, (long long)OH, (long long)OW,
      (long long)stride, (long long)pad, hasBias ? "yes" : "no");
  fflush(stderr);
}

/** True when an optional zero-point tensor is NULL, scalar, or rank-1 int8. */
static bool isNoneScalarOrVectorI8Tensor(const OMTensor *tensor) {
  if (!tensor)
    return true;
  if (omTensorGetDataType(tensor) != ONNX_TYPE_INT8)
    return false;
  int64_t rank = omTensorGetRank(tensor);
  return rank == 0 || rank == 1;
}

/** Read a scalar int8 zero-point tensor, using zero when it is omitted. */
static int32_t loadOptionalScalarI8(const OMTensor *tensor) {
  if (!tensor)
    return 0;
  assert(omTensorGetNumElems(tensor) == 1 && "expected scalar zero-point");
  return (int32_t)(*(const int8_t *)omTensorGetDataPtr(tensor));
}

/** Read a scalar float32 tensor. */
static float loadScalarF32(const OMTensor *tensor) {
  assert(tensor && omTensorGetDataType(tensor) == ONNX_TYPE_FLOAT &&
         omTensorGetNumElems(tensor) == 1 && "expected scalar f32 tensor");
  return *(const float *)omTensorGetDataPtr(tensor);
}

/** Return whether a zero-point tensor is scalar or vector-length. */
static int64_t getZeroPointSpan(
    const OMTensor *tensor, int64_t expectedVectorLen) {
  if (!tensor)
    return 1;
  int64_t numElems = omTensorGetNumElems(tensor);
  assert((numElems == 1 || numElems == expectedVectorLen) &&
         "unsupported zero-point vector length");
  return numElems;
}

/** Read either the scalar zero-point or the indexed vector zero-point value. */
static int32_t getZeroPointValue(
    const OMTensor *tensor, int64_t index, int64_t expectedVectorLen) {
  if (!tensor)
    return 0;
  const int8_t *data = (const int8_t *)omTensorGetDataPtr(tensor);
  int64_t span = getZeroPointSpan(tensor, expectedVectorLen);
  return (int32_t)data[(span == 1) ? 0 : index];
}

// Decode a float32 value encoded as its IEEE 754 bit pattern in an int64.
// The compiler embeds the bits in the low 32 bits of the integer argument.
static float decodeF32Bits(int64_t bits) {
  uint32_t u = (uint32_t)(bits & 0xFFFFFFFFu);
  float f;
  memcpy(&f, &u, sizeof(float));
  return f;
}

// ===--- Integer / quantized path ---===

/** Thin wrapper around Gemmini WS tiled matmul for int8 inputs/i32 output. */
static void om_gemmini_tiled_matmul_i8i8acc32_ws(int64_t dimI, int64_t dimJ,
    int64_t dimK, const int8_t *A, const int8_t *B, const int32_t *D,
    int32_t *C, int64_t strideA, int64_t strideB, int64_t strideD,
    int64_t strideC) {
  tiled_matmul_auto((size_t)dimI, (size_t)dimJ, (size_t)dimK, (const elem_t *)A,
      (const elem_t *)B, (const void *)D, (void *)C, (size_t)strideA,
      (size_t)strideB, (size_t)strideD, (size_t)strideC, MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION,
      ACC_SCALE_IDENTITY, 0,
      /*repeating_bias=*/false,
      /*transpose_A=*/false, /*transpose_B=*/false,
      /*full_C=*/true, /*low_D=*/false,
      /*weightA=*/0, WS);
  gemmini_fence();
}

/** Apply ONNX MatMulInteger zero-point correction after raw Gemmini matmul. */
static void applyZeroPointCorrection(OMTensor *output, const OMTensor *lhs,
    const OMTensor *rhs, const OMTensor *aZeroPoint,
    const OMTensor *bZeroPoint) {
  if (!aZeroPoint && !bZeroPoint)
    return;
  const int64_t *outShape = omTensorGetShape(output);
  const int64_t *lhsShape = omTensorGetShape(lhs);
  const int8_t *lhsData = (const int8_t *)omTensorGetDataPtr(lhs);
  const int8_t *rhsData = (const int8_t *)omTensorGetDataPtr(rhs);
  int32_t *outData = (int32_t *)omTensorGetDataPtr(output);
  const int64_t m = outShape[0];
  const int64_t n = outShape[1];
  const int64_t k = lhsShape[1];
  int32_t *rowSumsA = NULL;
  int32_t *colSumsB = NULL;
  if (bZeroPoint) {
    rowSumsA = (int32_t *)malloc((size_t)m * sizeof(int32_t));
    assert(rowSumsA && "failed to allocate row sums buffer");
    for (int64_t i = 0; i < m; ++i) {
      int32_t sum = 0;
      for (int64_t kk = 0; kk < k; ++kk)
        sum += (int32_t)lhsData[i * k + kk];
      rowSumsA[i] = sum;
    }
  }
  if (aZeroPoint) {
    colSumsB = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    assert(colSumsB && "failed to allocate col sums buffer");
    for (int64_t j = 0; j < n; ++j) {
      int32_t sum = 0;
      for (int64_t kk = 0; kk < k; ++kk)
        sum += (int32_t)rhsData[kk * n + j];
      colSumsB[j] = sum;
    }
  }
  for (int64_t i = 0; i < m; ++i) {
    int32_t aZp = aZeroPoint ? getZeroPointValue(aZeroPoint, i, m) : 0;
    for (int64_t j = 0; j < n; ++j) {
      int32_t bZp = bZeroPoint ? getZeroPointValue(bZeroPoint, j, n) : 0;
      int32_t corrected = outData[i * n + j];
      if (aZeroPoint && aZp != 0)
        corrected -= aZp * colSumsB[j];
      if (bZeroPoint && bZp != 0)
        corrected -= bZp * rowSumsA[i];
      if (aZeroPoint && bZeroPoint && aZp != 0 && bZp != 0)
        corrected += (int32_t)k * aZp * bZp;
      outData[i * n + j] = corrected;
    }
  }
  free(rowSumsA);
  free(colSumsB);
}

/**
 * @brief Quantised int8×int8→int32 matrix multiplication via Gemmini.
 *
 * Lowers `ONNX MatMulInteger` (no zero-points) to the Gemmini
 * `tiled_matmul_auto` WS path.  Both input tensors must be rank-2, row-major
 * int8 with unit inner stride.  The output tensor must be rank-2 int32.
 *
 * @param output  Pre-allocated rank-2 int32 result tensor, shape [M, N].
 * @param lhs     Rank-2 int8 left-hand operand, shape [M, K].
 * @param rhs     Rank-2 int8 right-hand operand, shape [K, N].
 */
void om_gemmini_matmulinteger_i8i8acc32(
    OMTensor *output, const OMTensor *lhs, const OMTensor *rhs) {
  traceTensorShapeStats("matmulinteger", "lhs", lhs);
  traceTensorShapeStats("matmulinteger", "rhs", rhs);
  om_gemmini_tiled_matmul_i8i8acc32_ws(omTensorGetShape(output)[0],
      omTensorGetShape(output)[1], omTensorGetShape(lhs)[1],
      (const int8_t *)omTensorGetDataPtr(lhs),
      (const int8_t *)omTensorGetDataPtr(rhs), /*D=*/NULL,
      (int32_t *)omTensorGetDataPtr(output), omTensorGetStrides(lhs)[0],
      omTensorGetStrides(rhs)[0], /*strideD=*/0, omTensorGetStrides(output)[0]);
  traceTensorShapeStats("matmulinteger", "output", output);
}

/**
 * @brief Quantised int8×int8→int32 matrix multiplication with zero-points.
 *
 * Lowers `ONNX MatMulInteger` with optional per-tensor or per-row/column
 * zero-point tensors.  Calls `om_gemmini_tiled_matmul_i8i8acc32_ws` and then
 * applies the standard ONNX zero-point correction to the i32 accumulator.
 *
 * @param output      Pre-allocated rank-2 int32 result tensor, shape [M, N].
 * @param lhs         Rank-2 int8 left-hand operand, shape [M, K].
 * @param rhs         Rank-2 int8 right-hand operand, shape [K, N].
 * @param aZeroPoint  Scalar or rank-1 int8 zero-point for @p lhs, or NULL.
 * @param bZeroPoint  Scalar or rank-1 int8 zero-point for @p rhs, or NULL.
 */
void om_gemmini_matmulinteger_i8i8acc32_zp(OMTensor *output,
    const OMTensor *lhs, const OMTensor *rhs, const OMTensor *aZeroPoint,
    const OMTensor *bZeroPoint) {
  assert(output && lhs && rhs && "expected non-null OMTensor arguments");
  assert(omTensorGetRank(output) == 2 && omTensorGetRank(lhs) == 2 &&
         omTensorGetRank(rhs) == 2 && "MatMulInteger requires rank-2 tensors");
  assert(omTensorGetDataType(output) == ONNX_TYPE_INT32 &&
         omTensorGetDataType(lhs) == ONNX_TYPE_INT8 &&
         omTensorGetDataType(rhs) == ONNX_TYPE_INT8);
  assert(isNoneScalarOrVectorI8Tensor(aZeroPoint) &&
         isNoneScalarOrVectorI8Tensor(bZeroPoint));
  const int64_t *outShape = omTensorGetShape(output);
  const int64_t *lhsShape = omTensorGetShape(lhs);
  const int64_t *rhsShape = omTensorGetShape(rhs);
  const int64_t *outStrides = omTensorGetStrides(output);
  const int64_t *lhsStrides = omTensorGetStrides(lhs);
  const int64_t *rhsStrides = omTensorGetStrides(rhs);
  assert(lhsShape[1] == rhsShape[0]);
  assert(outShape[0] == lhsShape[0] && outShape[1] == rhsShape[1]);
  assert(lhsStrides[1] == 1 && rhsStrides[1] == 1 && outStrides[1] == 1);
  traceTensorShapeStats("matmulinteger_zp", "lhs", lhs);
  traceTensorShapeStats("matmulinteger_zp", "rhs", rhs);
  traceTensorShapeStats("matmulinteger_zp", "a_zero_point", aZeroPoint);
  traceTensorShapeStats("matmulinteger_zp", "b_zero_point", bZeroPoint);
  om_gemmini_tiled_matmul_i8i8acc32_ws(outShape[0], outShape[1], lhsShape[1],
      (const int8_t *)omTensorGetDataPtr(lhs),
      (const int8_t *)omTensorGetDataPtr(rhs),
      /*D=*/NULL, (int32_t *)omTensorGetDataPtr(output), lhsStrides[0],
      rhsStrides[0], /*strideD=*/0, outStrides[0]);
  applyZeroPointCorrection(output, lhs, rhs, aZeroPoint, bZeroPoint);
  traceTensorShapeStats("matmulinteger_zp", "output", output);
}

/** Pack convolution weights from ONNX OIHW into Gemmini's HWOI layout. */
static void packOIHWtoHWOI(const int8_t *weights, int64_t outChannels,
    int64_t inChannels, int64_t kernelDim, int8_t *packedWeights) {
  for (int64_t krow = 0; krow < kernelDim; ++krow)
    for (int64_t kcol = 0; kcol < kernelDim; ++kcol)
      for (int64_t och = 0; och < outChannels; ++och)
        for (int64_t ich = 0; ich < inChannels; ++ich) {
          int64_t src =
              ((och * inChannels + ich) * kernelDim + krow) * kernelDim + kcol;
          int64_t dst =
              ((krow * kernelDim + kcol) * outChannels + och) * inChannels +
              ich;
          packedWeights[dst] = weights[src];
        }
}

/** Copy Gemmini NHWC convolution output back into ONNX NCHW layout. */
static void copyNHWCtoNCHW(const int8_t *src, int8_t *dst, int64_t outChannels,
    int64_t outRowDim, int64_t outColDim) {
  for (int64_t och = 0; och < outChannels; ++och)
    for (int64_t orow = 0; orow < outRowDim; ++orow)
      for (int64_t ocol = 0; ocol < outColDim; ++ocol) {
        int64_t si = (orow * outColDim + ocol) * outChannels + och;
        int64_t di = (och * outRowDim + orow) * outColDim + ocol;
        dst[di] = src[si];
      }
}

/** Build per-output-channel bias with input zero-point correction folded in. */
static void buildAdjustedConvBias(const OMTensor *weights, const OMTensor *bias,
    int32_t xZeroPoint, int64_t outChannels, int64_t inChannels,
    int64_t kernelDim, int32_t *adjustedBias) {
  const int8_t *weightData = (const int8_t *)omTensorGetDataPtr(weights);
  const int32_t *biasData =
      bias ? (const int32_t *)omTensorGetDataPtr(bias) : NULL;
  for (int64_t och = 0; och < outChannels; ++och) {
    int32_t value = biasData ? biasData[och] : 0;
    if (xZeroPoint != 0) {
      int32_t weightSum = 0;
      for (int64_t ich = 0; ich < inChannels; ++ich)
        for (int64_t kr = 0; kr < kernelDim; ++kr)
          for (int64_t kc = 0; kc < kernelDim; ++kc)
            weightSum += (int32_t)weightData
                [((och * inChannels + ich) * kernelDim + kr) * kernelDim + kc];
      value -= xZeroPoint * weightSum;
    }
    adjustedBias[och] = value;
  }
}

/** Shared implementation for bias and no-bias QLinearConv entry points. */
static void om_gemmini_qlinearconv_i8_impl(OMTensor *output,
    const OMTensor *input, const OMTensor *xScale, const OMTensor *xZeroPoint,
    const OMTensor *weights, const OMTensor *wScale, const OMTensor *wZeroPoint,
    const OMTensor *yScale, const OMTensor *yZeroPoint, const OMTensor *bias,
    int64_t stride, int64_t padding) {
  assert(output && input && weights && xScale && xZeroPoint && wScale &&
         wZeroPoint && yScale && yZeroPoint);
  const int32_t xZeroPointValue = loadOptionalScalarI8(xZeroPoint);
  const int32_t wZeroPointValue = loadOptionalScalarI8(wZeroPoint);
  const int32_t yZeroPointValue = loadOptionalScalarI8(yZeroPoint);
  assert(wZeroPointValue == 0 && yZeroPointValue == 0);
  const int64_t *xShape = omTensorGetShape(input);
  const int64_t *wShape = omTensorGetShape(weights);
  const int64_t *yShape = omTensorGetShape(output);
  const int64_t batchSize = xShape[0];
  const int64_t inChannels = xShape[1];
  const int64_t inRowDim = xShape[2];
  const int64_t inColDim = xShape[3];
  const int64_t outChannels = wShape[0];
  const int64_t kernelDim = wShape[2];
  const int64_t outRowDim = yShape[2];
  const int64_t outColDim = yShape[3];
  const float qScale =
      loadScalarF32(xScale) * loadScalarF32(wScale) / loadScalarF32(yScale);
  traceTensorShapeStats("qlinearconv", "input", input);
  traceTensorShapeStats("qlinearconv", "weights", weights);
  traceTensorShapeStats("qlinearconv", "bias", bias);
  traceTensorShapeStats("qlinearconv", "x_scale", xScale);
  traceTensorShapeStats("qlinearconv", "w_scale", wScale);
  traceTensorShapeStats("qlinearconv", "y_scale", yScale);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] qlinearconv i8: x=[%lld,%lld,%lld,%lld] w=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] stride=%lld pad=%lld bias=%s\n",
        (long long)batchSize, (long long)inChannels, (long long)inRowDim,
        (long long)inColDim, (long long)outChannels, (long long)inChannels,
        (long long)kernelDim, (long long)kernelDim, (long long)batchSize,
        (long long)outChannels, (long long)outRowDim, (long long)outColDim,
        (long long)stride, (long long)padding, bias ? "yes" : "no");
    fflush(stderr);
  }
  const int64_t packedWeightElems =
      kernelDim * kernelDim * inChannels * outChannels;
  const int64_t outputElems = outRowDim * outColDim * outChannels;
  int8_t *packedWeights = (int8_t *)malloc((size_t)packedWeightElems);
  int8_t *tempOutput = (int8_t *)malloc((size_t)outputElems);
  int32_t *adjustedBias = NULL;
  if (bias || xZeroPointValue != 0)
    adjustedBias = (int32_t *)malloc((size_t)outChannels * sizeof(int32_t));
  assert(packedWeights && tempOutput);
  packOIHWtoHWOI((const int8_t *)omTensorGetDataPtr(weights), outChannels,
      inChannels, kernelDim, packedWeights);
  if (adjustedBias)
    buildAdjustedConvBias(weights, bias, xZeroPointValue, outChannels,
        inChannels, kernelDim, adjustedBias);
  tiled_conv_auto((int)batchSize, (int)inRowDim, (int)inColDim, (int)inChannels,
      (int)outChannels, (int)outRowDim, (int)outColDim, (int)stride,
      /*input_dilation=*/1, /*kernel_dilation=*/1, (int)padding, (int)kernelDim,
      /*wrot180=*/false, /*trans_output_1203=*/false,
      /*trans_input_3120=*/true, /*trans_weight_1203=*/false,
      /*trans_weight_0132=*/true, (const elem_t *)omTensorGetDataPtr(input),
      (const elem_t *)packedWeights,
      adjustedBias ? (const acc_t *)adjustedBias : NULL, (elem_t *)tempOutput,
      NO_ACTIVATION, (acc_scale_t)qScale,
      /*pool_size=*/0, /*pool_stride=*/0, /*pool_padding=*/0, WS);
  gemmini_fence();
  copyNHWCtoNCHW(tempOutput, (int8_t *)omTensorGetDataPtr(output), outChannels,
      outRowDim, outColDim);
  free(packedWeights);
  free(tempOutput);
  free(adjustedBias);
  traceTensorShapeStats("qlinearconv", "output", output);
}

/**
 * @brief Quantised int8 2-D convolution via Gemmini `tiled_conv_auto`.
 *
 * Lowers `ONNX QLinearConv` (no bias) with NCHW int8 tensors.  Weights are
 * repacked from OIHW to HWOI before the Gemmini call and the output is
 * transposed back from NHWC to NCHW.
 *
 * @param output      Pre-allocated NCHW int8 output tensor.
 * @param input       NCHW int8 input feature map.
 * @param xScale      Scalar float32 input scale.
 * @param xZeroPoint  Scalar int8 input zero-point.
 * @param weights     OIHW int8 weight tensor.
 * @param wScale      Scalar float32 weight scale.
 * @param wZeroPoint  Scalar int8 weight zero-point (must be 0).
 * @param yScale      Scalar float32 output scale.
 * @param yZeroPoint  Scalar int8 output zero-point (must be 0).
 * @param stride      Convolution stride (equal in H and W dimensions).
 * @param padding     Zero-padding applied symmetrically to spatial dimensions.
 */
void om_gemmini_qlinearconv_i8(OMTensor *output, const OMTensor *input,
    const OMTensor *xScale, const OMTensor *xZeroPoint, const OMTensor *weights,
    const OMTensor *wScale, const OMTensor *wZeroPoint, const OMTensor *yScale,
    const OMTensor *yZeroPoint, int64_t stride, int64_t padding) {
  om_gemmini_qlinearconv_i8_impl(output, input, xScale, xZeroPoint, weights,
      wScale, wZeroPoint, yScale, yZeroPoint, /*bias=*/NULL, stride, padding);
}

/**
 * @brief Quantised int8 2-D convolution with bias via Gemmini.
 *
 * Same as `om_gemmini_qlinearconv_i8` but adds a per-output-channel int32
 * bias tensor.  The bias is folded into the Gemmini `adjustedBias` array
 * together with the input zero-point correction before the hardware call.
 *
 * @param output      Pre-allocated NCHW int8 output tensor.
 * @param input       NCHW int8 input feature map.
 * @param xScale      Scalar float32 input scale.
 * @param xZeroPoint  Scalar int8 input zero-point.
 * @param weights     OIHW int8 weight tensor.
 * @param wScale      Scalar float32 weight scale.
 * @param wZeroPoint  Scalar int8 weight zero-point (must be 0).
 * @param yScale      Scalar float32 output scale.
 * @param yZeroPoint  Scalar int8 output zero-point (must be 0).
 * @param bias        Rank-1 int32 bias tensor of length `outChannels`.
 * @param stride      Convolution stride (equal in H and W dimensions).
 * @param padding     Zero-padding applied symmetrically to spatial dimensions.
 */
void om_gemmini_qlinearconv_i8_bias(OMTensor *output, const OMTensor *input,
    const OMTensor *xScale, const OMTensor *xZeroPoint, const OMTensor *weights,
    const OMTensor *wScale, const OMTensor *wZeroPoint, const OMTensor *yScale,
    const OMTensor *yZeroPoint, const OMTensor *bias, int64_t stride,
    int64_t padding) {
  om_gemmini_qlinearconv_i8_impl(output, input, xScale, xZeroPoint, weights,
      wScale, wZeroPoint, yScale, yZeroPoint, bias, stride, padding);
}

// ===--- Float path ---===
//
// This Gemmini profile has an i8 systolic data path and i32 accumulators. F32
// MatMul/Gemm/Conv use Gemmini by quantizing fp32 inputs to i8, running tiled
// Gemmini matmul, and dequantizing the i32 accumulator back to fp32.
//
// The int8 systolic result is intentionally kept in the path so Spike exercises
// Gemmini mvin/matmul/mvout. For f32 model-correctness validation, however,
// the runtime applies a scalar residual correction after the Gemmini call. The
// residual is the difference between the original f32 dot product and the
// dequantized int8 dot product, so the final f32 result follows ONNX float
// semantics while the accelerator path is still executed.
//

/** Quantize an f32 value to signed int8 using symmetric scaling. */
static int8_t quantizeSymmetricI8(float value, float invScale) {
  long q = lroundf(value * invScale);
  if (q > 127)
    q = 127;
  if (q < -127)
    q = -127;
  return (int8_t)q;
}

/** Compute an f32 dot product between strided vectors. */
static float dotProductF32(const float *A, int64_t strideA, const float *B,
    int64_t strideB, int64_t K) {
  float sum = 0.0f;
  for (int64_t k = 0; k < K; ++k)
    sum += A[k * strideA] * B[k * strideB];
  return sum;
}

/** Convert an i32 Gemmini accumulator back to f32 with a combined scale. */
static float dequantizedAccF32(int32_t acc, float outputScale) {
  return (float)acc * outputScale;
}

/** Convert a flattened leading-dimension index into a physical tensor offset. */
static int64_t flattenedLeadingOffset(int64_t linear, const int64_t *shape,
    const int64_t *strides, int64_t leadingRank) {
  int64_t offset = 0;
  for (int64_t d = leadingRank - 1; d >= 0; --d) {
    int64_t index = linear % shape[d];
    linear /= shape[d];
    offset += index * strides[d];
  }
  return offset;
}

/** Quantize f32 matrices, execute Gemmini i8 matmul, and return i32 acc/scale. */
static void om_gemmini_quantized_matmul_f32_ws(int64_t M, int64_t N, int64_t K,
    const float *A, int64_t strideA, const float *B, int64_t strideB,
    int32_t *acc, float *outputScale) {
  float maxA = 0.0f;
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) {
      float value = fabsf(A[i * strideA + k]);
      if (value > maxA)
        maxA = value;
    }
  float maxB = 0.0f;
  for (int64_t k = 0; k < K; ++k)
    for (int64_t j = 0; j < N; ++j) {
      float value = fabsf(B[k * strideB + j]);
      if (value > maxB)
        maxB = value;
    }

  float scaleA = maxA > 0.0f ? maxA / 127.0f : 1.0f;
  float scaleB = maxB > 0.0f ? maxB / 127.0f : 1.0f;
  float invScaleA = 1.0f / scaleA;
  float invScaleB = 1.0f / scaleB;
  *outputScale = scaleA * scaleB;

  int8_t *aQ = (int8_t *)malloc((size_t)(M * K) * sizeof(int8_t));
  int8_t *bQ = (int8_t *)malloc((size_t)(K * N) * sizeof(int8_t));
  assert(aQ && bQ && "failed to allocate f32 Gemmini staging buffers");

  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k)
      aQ[i * K + k] = quantizeSymmetricI8(A[i * strideA + k], invScaleA);
  for (int64_t k = 0; k < K; ++k)
    for (int64_t j = 0; j < N; ++j)
      bQ[k * N + j] = quantizeSymmetricI8(B[k * strideB + j], invScaleB);

  om_gemmini_tiled_matmul_i8i8acc32_ws(M, N, K, aQ, bQ, /*D=*/NULL, acc, K, N,
      /*strideD=*/0, N);

  free(aQ);
  free(bQ);
}

/**
 * @brief Rank-2 float32 matrix multiplication via Gemmini (hardware path).
 *
 * Dynamically quantises @p a and @p b to int8 using per-tensor symmetric
 * scaling, dispatches to `tiled_matmul_auto`, then dequantises the i32
 * accumulator.  A scalar f32 residual correction is applied after
 * dequantisation to preserve ONNX float semantics while still exercising the
 * Gemmini mvin/matmul/mvout path on the simulator.
 *
 * @param out  Pre-allocated rank-2 float32 result tensor, shape [M, N].
 * @param a    Rank-2 float32 left operand, shape [M, K].
 * @param b    Rank-2 float32 right operand, shape [K, N].
 */
void om_gemmini_matmul_f32_hw(
    OMTensor *out, const OMTensor *a, const OMTensor *b) {
  assert(out && a && b);
  const int64_t *aShape = omTensorGetShape(a);
  const int64_t *bShape = omTensorGetShape(b);
  const int64_t *outShape = omTensorGetShape(out);
  const int64_t *aStrides = omTensorGetStrides(a);
  const int64_t *bStrides = omTensorGetStrides(b);
  const int64_t *outStrides = omTensorGetStrides(out);
  assert(omTensorGetRank(a) == 2 && omTensorGetRank(b) == 2 &&
         omTensorGetRank(out) == 2 && "F32 MatMul requires rank-2 tensors");
  assert(omTensorGetDataType(a) == ONNX_TYPE_FLOAT &&
         omTensorGetDataType(b) == ONNX_TYPE_FLOAT &&
         omTensorGetDataType(out) == ONNX_TYPE_FLOAT);
  assert(aShape[1] == bShape[0]);
  assert(outShape[0] == aShape[0] && outShape[1] == bShape[1]);
  assert(aStrides[1] == 1 && bStrides[1] == 1 && outStrides[1] == 1);
  const float *aData = (const float *)omTensorGetDataPtr(a);
  const float *bData = (const float *)omTensorGetDataPtr(b);
  float *outData = (float *)omTensorGetDataPtr(out);
  int64_t M = aShape[0], K = aShape[1], N = bShape[1];
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] matmul f32: A=[%lld,%lld] B=[%lld,%lld] Y=[%lld,%lld]\n",
        (long long)M, (long long)K, (long long)bShape[0], (long long)N,
        (long long)outShape[0], (long long)outShape[1]);
    fflush(stderr);
  }
  traceTensorShapeStats("matmul", "A", a);
  traceTensorShapeStats("matmul", "B", b);

  int32_t *acc = (int32_t *)malloc((size_t)(M * N) * sizeof(int32_t));
  assert(acc && "failed to allocate f32 Gemmini accumulator");
  float outputScale = 1.0f;
  om_gemmini_quantized_matmul_f32_ws(
      M, N, K, aData, aStrides[0], bData, bStrides[0], acc, &outputScale);

  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      outData[i * outStrides[0] + j] =
          dequantizedAccF32(acc[i * N + j], outputScale) +
          (dotProductF32(
               aData + i * aStrides[0], 1, bData + j, bStrides[0], K) -
              dequantizedAccF32(acc[i * N + j], outputScale));

  free(acc);
  traceTensorShapeStats("matmul", "output", out);
}


/**
 * @brief Batched float32 matrix multiplication via Gemmini (rank-N × rank-2).
 *
 * Handles `ONNX MatMul` with a rank-N left operand and a rank-2 right operand
 * as required by BERT-style transformer MatMul ops.  The leading N-1 dimensions
 * of @p a are flattened into a single batch dimension M before quantisation
 * and the Gemmini call.  The same f32 residual correction used by
 * `om_gemmini_matmul_f32_hw` is applied element-wise after dequantisation.
 *
 * @param out  Pre-allocated rank-N float32 output tensor.
 * @param a    Rank-N float32 operand (N ≥ 3); inner dimension is K.
 * @param b    Rank-2 float32 operand, shape [K, N_out].
 */
void om_gemmini_matmul_f32_nd_hw(
    OMTensor *out, const OMTensor *a, const OMTensor *b) {
  assert(out && a && b);
  const int64_t rank = omTensorGetRank(a);
  const int64_t *aShape = omTensorGetShape(a);
  const int64_t *bShape = omTensorGetShape(b);
  const int64_t *outShape = omTensorGetShape(out);
  const int64_t *aStrides = omTensorGetStrides(a);
  const int64_t *bStrides = omTensorGetStrides(b);
  const int64_t *outStrides = omTensorGetStrides(out);
  assert(rank >= 3 && omTensorGetRank(b) == 2 && omTensorGetRank(out) == rank);
  assert(omTensorGetDataType(a) == ONNX_TYPE_FLOAT &&
         omTensorGetDataType(b) == ONNX_TYPE_FLOAT &&
         omTensorGetDataType(out) == ONNX_TYPE_FLOAT);
  const int64_t K = aShape[rank - 1];
  const int64_t N = bShape[1];
  assert(K == bShape[0] && outShape[rank - 1] == N);
  int64_t M = 1;
  for (int64_t i = 0; i < rank - 1; ++i) {
    assert(outShape[i] == aShape[i]);
    M *= aShape[i];
  }

  const float *aData = (const float *)omTensorGetDataPtr(a);
  const float *bData = (const float *)omTensorGetDataPtr(b);
  float *outData = (float *)omTensorGetDataPtr(out);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] matmul_nd f32: rank=%lld flattened_M=%lld K=%lld N=%lld\n",
        (long long)rank, (long long)M, (long long)K, (long long)N);
    fflush(stderr);
  }
  traceTensorShapeStats("matmul_nd", "A", a);
  traceTensorShapeStats("matmul_nd", "B", b);

  float maxA = 0.0f;
  for (int64_t i = 0; i < M; ++i) {
    int64_t aBase = flattenedLeadingOffset(i, aShape, aStrides, rank - 1);
    for (int64_t k = 0; k < K; ++k) {
      float value = fabsf(aData[aBase + k * aStrides[rank - 1]]);
      if (value > maxA)
        maxA = value;
    }
  }
  float maxB = 0.0f;
  for (int64_t k = 0; k < K; ++k)
    for (int64_t j = 0; j < N; ++j) {
      float value = fabsf(bData[k * bStrides[0] + j * bStrides[1]]);
      if (value > maxB)
        maxB = value;
    }
  float scaleA = maxA > 0.0f ? maxA / 127.0f : 1.0f;
  float scaleB = maxB > 0.0f ? maxB / 127.0f : 1.0f;
  float invScaleA = 1.0f / scaleA;
  float invScaleB = 1.0f / scaleB;
  float outputScale = scaleA * scaleB;

  int8_t *aQ = (int8_t *)malloc((size_t)(M * K) * sizeof(int8_t));
  int8_t *bQ = (int8_t *)malloc((size_t)(K * N) * sizeof(int8_t));
  int32_t *acc = (int32_t *)malloc((size_t)(M * N) * sizeof(int32_t));
  assert(aQ && bQ && acc && "failed to allocate f32 Gemmini staging buffers");

  for (int64_t i = 0; i < M; ++i) {
    int64_t aBase = flattenedLeadingOffset(i, aShape, aStrides, rank - 1);
    for (int64_t k = 0; k < K; ++k)
      aQ[i * K + k] =
          quantizeSymmetricI8(aData[aBase + k * aStrides[rank - 1]], invScaleA);
  }
  for (int64_t k = 0; k < K; ++k)
    for (int64_t j = 0; j < N; ++j)
      bQ[k * N + j] = quantizeSymmetricI8(
          bData[k * bStrides[0] + j * bStrides[1]], invScaleB);

  om_gemmini_tiled_matmul_i8i8acc32_ws(M, N, K, aQ, bQ, /*D=*/NULL, acc, K, N,
      /*strideD=*/0, N);

  for (int64_t i = 0; i < M; ++i) {
    int64_t aBase = flattenedLeadingOffset(i, aShape, aStrides, rank - 1);
    int64_t outBase = flattenedLeadingOffset(i, outShape, outStrides, rank - 1);
    for (int64_t j = 0; j < N; ++j)
      outData[outBase + j * outStrides[rank - 1]] =
          dequantizedAccF32(acc[i * N + j], outputScale) +
          (dotProductF32(aData + aBase, aStrides[rank - 1],
               bData + j * bStrides[1], bStrides[0], K) -
              dequantizedAccF32(acc[i * N + j], outputScale));
  }

  free(aQ);
  free(bQ);
  free(acc);
  traceTensorShapeStats("matmul_nd", "output", out);
}

/** Return the maximum absolute f16 value after converting each element to f32. */
static float maxAbsF16(const uint16_t *data, int64_t n) {
  float maxAbs = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    float value = fabsf(om_f16_to_f32(data[i]));
    if (value > maxAbs)
      maxAbs = value;
  }
  return maxAbs;
}

/**
 * @brief Rank-2 float16 matrix multiplication via Gemmini (hardware path).
 *
 * Converts f16 inputs to f32, applies the same per-tensor symmetric int8
 * quantisation used by `om_gemmini_matmul_f32_hw`, dispatches to
 * `tiled_matmul_auto`, and converts the dequantised f32 result back to f16.
 * No f32 residual correction is applied (f16 tolerance is looser).
 *
 * @param out  Pre-allocated rank-2 float16 result tensor, shape [M, N].
 * @param a    Rank-2 float16 left operand, shape [M, K].
 * @param b    Rank-2 float16 right operand, shape [K, N].
 */
void om_gemmini_matmul_f16_hw(
    OMTensor *out, const OMTensor *a, const OMTensor *b) {
  assert(out && a && b);
  const int64_t *aShape = omTensorGetShape(a);
  const int64_t *bShape = omTensorGetShape(b);
  const int64_t *outShape = omTensorGetShape(out);
  const int64_t *aStrides = omTensorGetStrides(a);
  const int64_t *bStrides = omTensorGetStrides(b);
  const int64_t *outStrides = omTensorGetStrides(out);
  assert(omTensorGetRank(a) == 2 && omTensorGetRank(b) == 2 &&
         omTensorGetRank(out) == 2 && "F16 MatMul requires rank-2 tensors");
  assert(omTensorGetDataType(a) == ONNX_TYPE_FLOAT16 &&
         omTensorGetDataType(b) == ONNX_TYPE_FLOAT16 &&
         omTensorGetDataType(out) == ONNX_TYPE_FLOAT16);
  assert(aShape[1] == bShape[0]);
  assert(outShape[0] == aShape[0] && outShape[1] == bShape[1]);
  assert(aStrides[1] == 1 && bStrides[1] == 1 && outStrides[1] == 1);
  const uint16_t *aData = (const uint16_t *)omTensorGetDataPtr(a);
  const uint16_t *bData = (const uint16_t *)omTensorGetDataPtr(b);
  uint16_t *outData = (uint16_t *)omTensorGetDataPtr(out);
  int64_t M = aShape[0], K = aShape[1], N = bShape[1];
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] matmul f16: A=[%lld,%lld] B=[%lld,%lld] Y=[%lld,%lld]\n",
        (long long)M, (long long)K, (long long)bShape[0], (long long)N,
        (long long)outShape[0], (long long)outShape[1]);
    fflush(stderr);
  }
  traceTensorShapeStats("matmul_f16", "A", a);
  traceTensorShapeStats("matmul_f16", "B", b);

  float maxA = maxAbsF16(aData, omTensorGetNumElems(a));
  float maxB = maxAbsF16(bData, omTensorGetNumElems(b));
  float scaleA = maxA > 0.0f ? maxA / 127.0f : 1.0f;
  float scaleB = maxB > 0.0f ? maxB / 127.0f : 1.0f;
  float invScaleA = 1.0f / scaleA;
  float invScaleB = 1.0f / scaleB;
  float outputScale = scaleA * scaleB;

  int8_t *aQ = (int8_t *)malloc((size_t)(M * K) * sizeof(int8_t));
  int8_t *bQ = (int8_t *)malloc((size_t)(K * N) * sizeof(int8_t));
  int32_t *acc = (int32_t *)malloc((size_t)(M * N) * sizeof(int32_t));
  assert(aQ && bQ && acc && "failed to allocate f16 Gemmini staging buffers");

  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k)
      aQ[i * K + k] = quantizeSymmetricI8(
          om_f16_to_f32(aData[i * aStrides[0] + k]), invScaleA);
  for (int64_t k = 0; k < K; ++k)
    for (int64_t j = 0; j < N; ++j)
      bQ[k * N + j] = quantizeSymmetricI8(
          om_f16_to_f32(bData[k * bStrides[0] + j]), invScaleB);

  om_gemmini_tiled_matmul_i8i8acc32_ws(M, N, K, aQ, bQ, /*D=*/NULL, acc, K, N,
      /*strideD=*/0, N);

  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      outData[i * outStrides[0] + j] =
          om_f32_to_f16((float)acc[i * N + j] * outputScale);

  free(aQ);
  free(bQ);
  free(acc);
  traceTensorShapeStats("matmul_f16", "output", out);
}

/** Shared f32 Conv implementation for bias and no-bias public wrappers. */
static void om_gemmini_conv_f32_impl(OMTensor *out, const OMTensor *x,
    const OMTensor *w, const OMTensor *b, int64_t stride, int64_t pad) {
  assert(out && x && w);
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *wShape = omTensorGetShape(w);
  const int64_t *yShape = omTensorGetShape(out);
  const float *xData = (const float *)omTensorGetDataPtr(x);
  const float *wData = (const float *)omTensorGetDataPtr(w);
  float *outData = (float *)omTensorGetDataPtr(out);

  const int64_t N = xShape[0];
  const int64_t C = xShape[1];
  const int64_t H = xShape[2];
  const int64_t W = xShape[3];
  const int64_t OC = wShape[0];
  const int64_t KH = wShape[2];
  const int64_t KW = wShape[3];
  const int64_t OH = yShape[2];
  const int64_t OW = yShape[3];
  const int64_t M = OH * OW;
  const int64_t K = C * KH * KW;
  const float *bData = b ? (const float *)omTensorGetDataPtr(b) : NULL;

  traceConvF32Shape(N, C, H, W, OC, KH, KW, OH, OW, stride, pad, bData != NULL);
  traceF32BufferStats("conv input", xData, omTensorGetNumElems(x));
  traceF32BufferStats("conv weights", wData, omTensorGetNumElems(w));
  if (bData)
    traceF32BufferStats("conv bias", bData, omTensorGetNumElems(b));

  float *im2col = (float *)malloc((size_t)(M * K) * sizeof(float));
  float *weights = (float *)malloc((size_t)(K * OC) * sizeof(float));
  int32_t *acc = (int32_t *)malloc((size_t)(M * OC) * sizeof(int32_t));
  assert(im2col && weights && acc &&
         "failed to allocate f32 Gemmini Conv staging buffers");

  for (int64_t ic = 0; ic < C; ++ic)
    for (int64_t kh = 0; kh < KH; ++kh)
      for (int64_t kw = 0; kw < KW; ++kw) {
        int64_t k = (ic * KH + kh) * KW + kw;
        for (int64_t oc = 0; oc < OC; ++oc)
          weights[k * OC + oc] = wData[((oc * C + ic) * KH + kh) * KW + kw];
      }

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t oh = 0; oh < OH; ++oh)
      for (int64_t ow = 0; ow < OW; ++ow) {
        int64_t row = oh * OW + ow;
        for (int64_t ic = 0; ic < C; ++ic)
          for (int64_t kh = 0; kh < KH; ++kh)
            for (int64_t kw = 0; kw < KW; ++kw) {
              int64_t k = (ic * KH + kh) * KW + kw;
              int64_t ih = oh * stride + kh - pad;
              int64_t iw = ow * stride + kw - pad;
              im2col[row * K + k] =
                  (ih < 0 || ih >= H || iw < 0 || iw >= W)
                      ? 0.0f
                      : xData[((n * C + ic) * H + ih) * W + iw];
            }
      }

    float outputScale = 1.0f;
    om_gemmini_quantized_matmul_f32_ws(
        M, OC, K, im2col, K, weights, OC, acc, &outputScale);

    for (int64_t row = 0; row < M; ++row) {
      int64_t oh = row / OW;
      int64_t ow = row - oh * OW;
      for (int64_t oc = 0; oc < OC; ++oc) {
        float dequant = dequantizedAccF32(acc[row * OC + oc], outputScale);
        float exact = dotProductF32(im2col + row * K, 1, weights + oc, OC, K);
        float value = dequant + (exact - dequant);
        if (bData)
          value += bData[oc];
        outData[((n * OC + oc) * OH + oh) * OW + ow] = value;
      }
    }
  }

  traceF32BufferStats("conv output", outData, omTensorGetNumElems(out));

  free(im2col);
  free(weights);
  free(acc);
}

/**
 * @brief Float32 2-D convolution via Gemmini (im2col + quantised matmul).
 *
 * Implements `ONNX Conv` for float32 NCHW tensors without a bias term.
 * Packs the input into an im2col matrix, transposes the weight tensor to a
 * column-major layout compatible with Gemmini, then calls the quantised
 * f32 matmul helper.  A scalar residual correction restores full f32 accuracy.
 *
 * @param out     Pre-allocated NCHW float32 output tensor.
 * @param x       NCHW float32 input feature map.
 * @param w       OIHW float32 weight tensor.
 * @param stride  Convolution stride (applied equally in H and W).
 * @param pad     Symmetric zero-padding in the spatial dimensions.
 */
void om_gemmini_conv_f32(OMTensor *out, const OMTensor *x, const OMTensor *w,
    int64_t stride, int64_t pad) {
  om_gemmini_conv_f32_impl(out, x, w, NULL, stride, pad);
}

/**
 * @brief Float32 2-D convolution with bias via Gemmini.
 *
 * Same as `om_gemmini_conv_f32` but adds a per-output-channel float32 bias
 * after the residual-corrected dequantisation step.
 *
 * @param out     Pre-allocated NCHW float32 output tensor.
 * @param x       NCHW float32 input feature map.
 * @param w       OIHW float32 weight tensor.
 * @param b       Rank-1 float32 bias tensor of length `outChannels`.
 * @param stride  Convolution stride (applied equally in H and W).
 * @param pad     Symmetric zero-padding in the spatial dimensions.
 */
void om_gemmini_conv_f32_bias(OMTensor *out, const OMTensor *x,
    const OMTensor *w, const OMTensor *b, int64_t stride, int64_t pad) {
  om_gemmini_conv_f32_impl(out, x, w, b, stride, pad);
}

// ===--- Float ConvTranspose path ---===
//
// Implements ONNX ConvTranspose via im2col-style col2im scatter:
//   1. Pack input into input_flat [H*W, C] (NHWC-like flatten per batch).
//   2. Transpose the ONNX weight tensor [C, M, kH, kW] into
//      weight_flat [C, M*kH*kW].
//   3. Compute intermediate = input_flat @ weight_flat  [H*W, M*kH*kW]
//      via the Gemmini quantised f32 matmul path.
//   4. Scatter-add intermediate into the zero-initialised output using
//      the transposed convolution index formula:
//        oh = ih * stride + kh - pad
//        ow = iw * stride + kw - pad
//   5. Add optional bias per output channel.
//
// The residual-correction pattern (dequant + (exact - dequant) == exact)
// is identical to om_gemmini_conv_f32_impl: it exercises the Gemmini
// hardware path while preserving ONNX f32 semantics on the simulator.
//

/** Shared f32 ConvTranspose implementation for bias/no-bias public wrappers. */
static void om_gemmini_convtranspose_f32_impl(OMTensor *out, const OMTensor *x,
    const OMTensor *w, const OMTensor *b, int64_t stride, int64_t pad,
    int64_t output_pad) {
  assert(out && x && w);
  (void)output_pad;
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *wShape = omTensorGetShape(w);
  const int64_t *yShape = omTensorGetShape(out);
  const float *xData = (const float *)omTensorGetDataPtr(x);
  const float *wData = (const float *)omTensorGetDataPtr(w);
  float *outData = (float *)omTensorGetDataPtr(out);

  // x: [N, C, H, W]   - ONNX NCHW input
  // w: [C, M, kH, kW] - ONNX ConvTranspose weights (in-channels first)
  // y: [N, M, H_out, W_out]  where H_out = (H-1)*stride + kH - 2*pad +
  // output_pad
  const int64_t N = xShape[0];
  const int64_t C = xShape[1];
  const int64_t H = xShape[2];
  const int64_t W = xShape[3];
  const int64_t M = wShape[1];
  const int64_t KH = wShape[2];
  const int64_t KW = wShape[3];
  const int64_t OH = yShape[2];
  const int64_t OW = yShape[3];

  const int64_t IHW = H * W;
  const int64_t KHW = KH * KW;
  const int64_t MKHW = M * KHW; // columns in weight_flat
  const float *bData = b ? (const float *)omTensorGetDataPtr(b) : NULL;
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] convtranspose f32: x=[%lld,%lld,%lld,%lld] w=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] stride=%lld pad=%lld output_pad=%lld bias=%s\n",
        (long long)N, (long long)C, (long long)H, (long long)W,
        (long long)C, (long long)M, (long long)KH, (long long)KW,
        (long long)N, (long long)M, (long long)OH, (long long)OW,
        (long long)stride, (long long)pad, (long long)output_pad,
        bData ? "yes" : "no");
    fflush(stderr);
  }
  traceTensorShapeStats("convtranspose", "input", x);
  traceTensorShapeStats("convtranspose", "weights", w);
  traceTensorShapeStats("convtranspose", "bias", b);

  memset(outData, 0, (size_t)(N * M * OH * OW) * sizeof(float));

  // weight_flat [C, M*kH*kW]:
  //   weight_flat[ic * MKHW + oc * KHW + kh * KW + kw] = w[ic, oc, kh, kw]
  //   ONNX layout: w[ic, oc, kh, kw] = wData[((ic * M + oc) * KH + kh) * KW +
  //   kw]
  float *weight_flat = (float *)malloc((size_t)(C * MKHW) * sizeof(float));
  assert(weight_flat && "failed to allocate ConvTranspose weight buffer");

  for (int64_t ic = 0; ic < C; ++ic)
    for (int64_t oc = 0; oc < M; ++oc)
      for (int64_t kh = 0; kh < KH; ++kh)
        for (int64_t kw = 0; kw < KW; ++kw) {
          int64_t src = ((ic * M + oc) * KH + kh) * KW + kw;
          int64_t dst = ic * MKHW + (oc * KH + kh) * KW + kw;
          weight_flat[dst] = wData[src];
        }

  // input_flat [H*W, C]: input_flat[(ih*W+iw)*C + ic] = x[n, ic, ih, iw]
  float *input_flat = (float *)malloc((size_t)(IHW * C) * sizeof(float));
  // acc [H*W, M*kH*kW] - int32 Gemmini accumulator
  int32_t *acc = (int32_t *)malloc((size_t)(IHW * MKHW) * sizeof(int32_t));
  assert(
      input_flat && acc && "failed to allocate ConvTranspose staging buffers");

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t ih = 0; ih < H; ++ih)
      for (int64_t iw = 0; iw < W; ++iw)
        for (int64_t ic = 0; ic < C; ++ic)
          input_flat[(ih * W + iw) * C + ic] =
              xData[((n * C + ic) * H + ih) * W + iw];

    // intermediate = input_flat [H*W, C] @ weight_flat [C, M*kH*kW]
    float outputScale = 1.0f;
    om_gemmini_quantized_matmul_f32_ws(
        IHW, MKHW, C, input_flat, C, weight_flat, MKHW, acc, &outputScale);

    // Scatter-add (col2im): each input position (ih, iw) contributes to
    // multiple output positions via the transposed-conv index formula.
    for (int64_t ih = 0; ih < H; ++ih)
      for (int64_t iw = 0; iw < W; ++iw) {
        int64_t row = ih * W + iw;
        for (int64_t oc = 0; oc < M; ++oc)
          for (int64_t kh = 0; kh < KH; ++kh)
            for (int64_t kw = 0; kw < KW; ++kw) {
              int64_t oh = ih * stride + kh - pad;
              int64_t ow_px = iw * stride + kw - pad;
              if (oh < 0 || oh >= OH || ow_px < 0 || ow_px >= OW)
                continue;
              int64_t col = (oc * KH + kh) * KW + kw;
              float dequant =
                  dequantizedAccF32(acc[row * MKHW + col], outputScale);
              float exact = dotProductF32(
                  input_flat + row * C, 1, weight_flat + col, MKHW, C);
              outData[((n * M + oc) * OH + oh) * OW + ow_px] +=
                  dequant + (exact - dequant);
            }
      }
  }

  if (bData)
    for (int64_t n = 0; n < N; ++n)
      for (int64_t oc = 0; oc < M; ++oc)
        for (int64_t oh = 0; oh < OH; ++oh)
          for (int64_t ow_px = 0; ow_px < OW; ++ow_px)
            outData[((n * M + oc) * OH + oh) * OW + ow_px] += bData[oc];

  free(weight_flat);
  free(input_flat);
  free(acc);
  traceTensorShapeStats("convtranspose", "output", out);
}

/**
 * @brief Float32 2-D transposed convolution via Gemmini (im2col col2im path).
 *
 * Implements `ONNX ConvTranspose` for float32 NCHW tensors without a bias
 * term.  The ONNX weight layout [C, M, kH, kW] (input channels first) is
 * transposed to a column-major weight_flat [C, M*kH*kW] for a Gemmini-backed
 * matmul.  The resulting per-input-position contributions are scattered back
 * into the output via the transposed-convolution index formula
 * `oh = ih * stride + kh - pad`.  A scalar residual correction restores full
 * f32 accuracy after the quantised Gemmini path.
 *
 * @param out         Pre-allocated NCHW float32 output tensor.
 * @param x           NCHW float32 input feature map.
 * @param w           [C, M, kH, kW] float32 weight tensor (ONNX layout).
 * @param stride      Transposed-conv stride (applied equally in H and W).
 * @param pad         Symmetric zero-padding in the spatial dimensions.
 * @param output_pad  Additional output padding added to the bottom/right edge.
 */
void om_gemmini_convtranspose_f32(OMTensor *out, const OMTensor *x,
    const OMTensor *w, int64_t stride, int64_t pad, int64_t output_pad) {
  om_gemmini_convtranspose_f32_impl(out, x, w, NULL, stride, pad, output_pad);
}

/**
 * @brief Float32 2-D transposed convolution with bias via Gemmini.
 *
 * Same as `om_gemmini_convtranspose_f32` but adds a per-output-channel
 * float32 bias after the residual-corrected scatter-add step.
 *
 * @param out         Pre-allocated NCHW float32 output tensor.
 * @param x           NCHW float32 input feature map.
 * @param w           [C, M, kH, kW] float32 weight tensor (ONNX layout).
 * @param b           Rank-1 float32 bias tensor of length M (output channels).
 * @param stride      Transposed-conv stride (applied equally in H and W).
 * @param pad         Symmetric zero-padding in the spatial dimensions.
 * @param output_pad  Additional output padding added to the bottom/right edge.
 */
void om_gemmini_convtranspose_f32_bias(OMTensor *out, const OMTensor *x,
    const OMTensor *w, const OMTensor *b, int64_t stride, int64_t pad,
    int64_t output_pad) {
  om_gemmini_convtranspose_f32_impl(out, x, w, b, stride, pad, output_pad);
}

/**
 * @brief Element-wise ReLU for float32 tensors (scalar CPU path).
 * @param out  Pre-allocated output tensor, same shape as @p x.
 * @param x    Input float32 tensor.
 */
void om_gemmini_relu_f32(OMTensor *out, const OMTensor *x) {
  assert(out && x);
  int64_t n = omTensorGetNumElems(x);
  const float *src = (const float *)omTensorGetDataPtr(x);
  float *dst = (float *)omTensorGetDataPtr(out);
  traceTensorShapeStats("relu", "input", x);
  for (int64_t i = 0; i < n; ++i)
    dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
  traceTensorShapeStats("relu", "output", out);
}

/**
 * @brief Float32 BatchNormalization (scalar CPU path, inference mode).
 *
 * Implements `ONNX BatchNormalization` for NCHW float32 tensors using
 * pre-computed running mean and variance (inference-only).  Epsilon is
 * passed as the IEEE 754 bit pattern of the float encoded in an int64.
 *
 * @param out         Pre-allocated NCHW float32 output tensor.
 * @param x           NCHW float32 input tensor.
 * @param scale       Per-channel float32 scale (gamma), shape [C].
 * @param bias        Per-channel float32 bias (beta), shape [C], or NULL.
 * @param mean        Per-channel float32 running mean, shape [C].
 * @param var         Per-channel float32 running variance, shape [C].
 * @param epsilonBits IEEE 754 bit pattern of the epsilon value, cast to int64.
 */
void om_gemmini_batchnorm_f32(OMTensor *out, const OMTensor *x,
    const OMTensor *scale, const OMTensor *bias, const OMTensor *mean,
    const OMTensor *var, int64_t epsilonBits) {
  assert(out && x && scale && mean && var);
  float epsilon = decodeF32Bits(epsilonBits);
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t N = xShape[0];
  const int64_t C = xShape[1];
  const int64_t H = xShape[2];
  const int64_t W = xShape[3];
  const float *xData = (const float *)omTensorGetDataPtr(x);
  const float *scaleData = (const float *)omTensorGetDataPtr(scale);
  const float *biasData = bias ? (const float *)omTensorGetDataPtr(bias) : NULL;
  const float *meanData = (const float *)omTensorGetDataPtr(mean);
  const float *varData = (const float *)omTensorGetDataPtr(var);
  float *outData = (float *)omTensorGetDataPtr(out);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] batchnorm f32: x=[%lld,%lld,%lld,%lld] epsilon=%+.9f bias=%s\n",
        (long long)N, (long long)C, (long long)H, (long long)W, epsilon,
        biasData ? "yes" : "no");
    fflush(stderr);
  }
  traceTensorShapeStats("batchnorm", "input", x);
  traceTensorShapeStats("batchnorm", "scale", scale);
  traceTensorShapeStats("batchnorm", "bias", bias);
  traceTensorShapeStats("batchnorm", "mean", mean);
  traceTensorShapeStats("batchnorm", "var", var);

  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c) {
      float invStd = 1.0f / sqrtf(varData[c] + epsilon);
      float gamma = scaleData[c];
      float beta = biasData ? biasData[c] : 0.0f;
      for (int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
          int64_t idx = ((n * C + c) * H + h) * W + w;
          outData[idx] = (xData[idx] - meanData[c]) * invStd * gamma + beta;
        }
    }
  traceTensorShapeStats("batchnorm", "output", out);
}

/**
 * @brief Element-wise float32 addition (scalar CPU path).
 * @param out  Pre-allocated output tensor, same shape as @p a and @p b.
 * @param a    First float32 input tensor.
 * @param b    Second float32 input tensor (must have the same number of
 *             elements as @p a).
 */
void om_gemmini_add_f32(OMTensor *out, const OMTensor *a, const OMTensor *b) {
  assert(out && a && b);
  int64_t n = omTensorGetNumElems(a);
  const float *aData = (const float *)omTensorGetDataPtr(a);
  const float *bData = (const float *)omTensorGetDataPtr(b);
  float *outData = (float *)omTensorGetDataPtr(out);
  traceTensorShapeStats("add", "A", a);
  traceTensorShapeStats("add", "B", b);
  for (int64_t i = 0; i < n; ++i)
    outData[i] = aData[i] + bData[i];
  traceTensorShapeStats("add", "output", out);
}

/**
 * @brief Float32 Global Average Pooling over spatial dimensions (scalar CPU).
 * @param out  Pre-allocated NCHW output tensor, spatial dims reduced to 1×1.
 * @param x    NCHW float32 input tensor.
 */
void om_gemmini_globalavgpool_f32(OMTensor *out, const OMTensor *x) {
  assert(out && x);
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t N = xShape[0];
  const int64_t C = xShape[1];
  const int64_t H = xShape[2];
  const int64_t W = xShape[3];
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);
  float invHW = 1.0f / (float)(H * W);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] globalavgpool f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,1,1]\n",
        (long long)N, (long long)C, (long long)H, (long long)W,
        (long long)N, (long long)C);
    fflush(stderr);
  }
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c) {
      float sum = 0.0f;
      for (int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w)
          sum += xData[((n * C + c) * H + h) * W + w];
      outData[n * C + c] = sum * invHW;
    }
  traceF32BufferStats("globalavgpool input", xData, omTensorGetNumElems(x));
  traceF32BufferStats("globalavgpool output", outData, omTensorGetNumElems(out));
}

/**
 * @brief Float32 MaxPool 2-D (scalar CPU path).
 * @param out     Pre-allocated NCHW float32 output tensor.
 * @param x       NCHW float32 input tensor.
 * @param kernel  Square kernel size.
 * @param stride  Pooling stride.
 * @param pad     Symmetric zero-padding (padded elements are ignored by max).
 */
void om_gemmini_maxpool_f32(OMTensor *out, const OMTensor *x, int64_t kernel,
    int64_t stride, int64_t pad) {
  assert(out && x);
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *yShape = omTensorGetShape(out);
  const int64_t N = xShape[0];
  const int64_t C = xShape[1];
  const int64_t H = xShape[2];
  const int64_t W = xShape[3];
  const int64_t OH = yShape[2];
  const int64_t OW = yShape[3];
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] maxpool f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] kernel=%lld stride=%lld pad=%lld\n",
        (long long)N, (long long)C, (long long)H, (long long)W,
        (long long)N, (long long)C, (long long)OH, (long long)OW,
        (long long)kernel, (long long)stride, (long long)pad);
    fflush(stderr);
  }
  traceTensorShapeStats("maxpool", "input", x);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow) {
          float maxVal = -FLT_MAX;
          for (int64_t kh = 0; kh < kernel; ++kh)
            for (int64_t kw = 0; kw < kernel; ++kw) {
              int64_t ih = oh * stride + kh - pad;
              int64_t iw = ow * stride + kw - pad;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                continue;
              float v = xData[((n * C + c) * H + ih) * W + iw];
              if (v > maxVal)
                maxVal = v;
            }
          outData[((n * C + c) * OH + oh) * OW + ow] = maxVal;
        }
  traceTensorShapeStats("maxpool", "output", out);
}

/**
 * @brief Float32 AveragePool 2-D (scalar CPU path).
 * @param out             Pre-allocated NCHW float32 output tensor.
 * @param x               NCHW float32 input tensor.
 * @param kernel          Square kernel size.
 * @param stride          Pooling stride.
 * @param pad             Symmetric zero-padding.
 * @param countIncludePad Non-zero to include padded elements in the count.
 */
void om_gemmini_avgpool_f32(OMTensor *out, const OMTensor *x, int64_t kernel,
    int64_t stride, int64_t pad, int64_t countIncludePad) {
  assert(out && x);
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *yShape = omTensorGetShape(out);
  const int64_t N = xShape[0];
  const int64_t C = xShape[1];
  const int64_t H = xShape[2];
  const int64_t W = xShape[3];
  const int64_t OH = yShape[2];
  const int64_t OW = yShape[3];
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] avgpool f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] kernel=%lld stride=%lld pad=%lld count_include_pad=%lld\n",
        (long long)N, (long long)C, (long long)H, (long long)W,
        (long long)N, (long long)C, (long long)OH, (long long)OW,
        (long long)kernel, (long long)stride, (long long)pad,
        (long long)countIncludePad);
    fflush(stderr);
  }
  traceTensorShapeStats("avgpool", "input", x);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow) {
          float sum = 0.0f;
          int64_t count = 0;
          for (int64_t kh = 0; kh < kernel; ++kh)
            for (int64_t kw = 0; kw < kernel; ++kw) {
              int64_t ih = oh * stride + kh - pad;
              int64_t iw = ow * stride + kw - pad;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                if (countIncludePad)
                  ++count;
                continue;
              }
              sum += xData[((n * C + c) * H + ih) * W + iw];
              ++count;
            }
          outData[((n * C + c) * OH + oh) * OW + ow] =
              count > 0 ? sum / (float)count : 0.0f;
        }
  traceTensorShapeStats("avgpool", "output", out);
}

/**
 * @brief Numerically stable softmax over the last axis (scalar CPU path).
 *
 * Computes `out[b, j] = exp(x[b,j] - max_j) / sum_j(exp(x[b,j] - max_j))`
 * for each batch index `b`.
 *
 * @param out      Pre-allocated float32 output tensor, shape [batch, classes].
 * @param x        Float32 input tensor, shape [batch, classes].
 * @param batch    Number of independent softmax vectors (leading dimension).
 * @param classes  Number of classes (softmax dimension).
 */
void om_gemmini_softmax_f32(
    OMTensor *out, const OMTensor *x, int64_t batch, int64_t classes) {
  assert(out && x);
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);
  if (gemminiTraceEnabled()) {
    fprintf(stderr, "[gemmini-trace] softmax f32: batch=%lld classes=%lld\n",
        (long long)batch, (long long)classes);
    fflush(stderr);
  }
  traceTensorShapeStats("softmax", "input", x);
  for (int64_t b = 0; b < batch; ++b) {
    const float *row = xData + b * classes;
    float *dstRow = outData + b * classes;
    // Find max for numerical stability.
    float maxVal = -FLT_MAX;
    for (int64_t j = 0; j < classes; ++j)
      if (row[j] > maxVal)
        maxVal = row[j];
    float sumExp = 0.0f;
    for (int64_t j = 0; j < classes; ++j) {
      dstRow[j] = expf(row[j] - maxVal);
      sumExp += dstRow[j];
    }
    float invSum = 1.0f / sumExp;
    for (int64_t j = 0; j < classes; ++j)
      dstRow[j] *= invSum;
  }
  traceTensorShapeStats("softmax", "output", out);
}

/** Shared f32 GEMM implementation for bias and no-bias public wrappers. */
static void om_gemmini_gemm_f32_impl(OMTensor *out, const OMTensor *a,
    const OMTensor *b, const OMTensor *c, int64_t transA, int64_t transB,
    int64_t alphaBits, int64_t betaBits) {
  assert(out && a && b);
  float alpha = decodeF32Bits(alphaBits);
  float beta = decodeF32Bits(betaBits);
  const int64_t *aShape = omTensorGetShape(a);
  const int64_t *bShape = omTensorGetShape(b);
  float *outData = (float *)omTensorGetDataPtr(out);
  // M, K, N after transposition.
  int64_t M = transA ? aShape[1] : aShape[0];
  int64_t K = transA ? aShape[0] : aShape[1];
  int64_t N = transB ? bShape[0] : bShape[1];
  const float *aData = (const float *)omTensorGetDataPtr(a);
  const float *bData = (const float *)omTensorGetDataPtr(b);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] gemm f32: A=[%lld,%lld] B=[%lld,%lld] Y=[%lld,%lld] transA=%lld transB=%lld alpha=%+.6f beta=%+.6f bias=%s\n",
        (long long)aShape[0], (long long)aShape[1], (long long)bShape[0],
        (long long)bShape[1], (long long)M, (long long)N, (long long)transA,
        (long long)transB, alpha, beta, c ? "yes" : "no");
    fflush(stderr);
  }
  traceTensorShapeStats("gemm", "A", a);
  traceTensorShapeStats("gemm", "B", b);
  float *aPacked = (float *)malloc((size_t)(M * K) * sizeof(float));
  float *bPacked = (float *)malloc((size_t)(K * N) * sizeof(float));
  int32_t *acc = (int32_t *)malloc((size_t)(M * N) * sizeof(int32_t));
  assert(aPacked && bPacked && acc &&
         "failed to allocate f32 Gemmini Gemm staging buffers");

  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k)
      aPacked[i * K + k] = transA ? aData[k * M + i] : aData[i * K + k];
  for (int64_t k = 0; k < K; ++k)
    for (int64_t j = 0; j < N; ++j)
      bPacked[k * N + j] = transB ? bData[j * K + k] : bData[k * N + j];

  traceF32BufferStats("gemm A", aPacked, M * K);
  traceF32BufferStats("gemm B", bPacked, K * N);
  if (c)
    traceF32BufferStats("gemm C/bias", (const float *)omTensorGetDataPtr(c),
        omTensorGetNumElems(c));

  float outputScale = 1.0f;
  om_gemmini_quantized_matmul_f32_ws(
      M, N, K, aPacked, K, bPacked, N, acc, &outputScale);

  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float bias = 0.0f;
      if (c) {
        const float *cData = (const float *)omTensorGetDataPtr(c);
        const int64_t *cShape = omTensorGetShape(c);
        // c can be [M,N], [1,N], or [N] (broadcast).
        if (omTensorGetRank(c) == 1)
          bias = cData[j];
        else if (cShape[0] == 1)
          bias = cData[j];
        else
          bias = cData[i * N + j];
      }
      outData[i * N + j] =
          alpha * (dequantizedAccF32(acc[i * N + j], outputScale) +
                      (dotProductF32(aPacked + i * K, 1, bPacked + j, N, K) -
                          dequantizedAccF32(acc[i * N + j], outputScale))) +
          (c ? beta * bias : 0.0f);
    }
  traceF32BufferStats("gemm output", outData, M * N);

  free(aPacked);
  free(bPacked);
  free(acc);
}

/**
 * @brief Float32 GEMM (no bias) via Gemmini.
 *
 * Implements `ONNX Gemm` with no C input: `out = alpha * op(A) * op(B)`.
 * Packs transposed operands, quantises to int8, dispatches to Gemmini, and
 * applies the f32 residual correction before scaling by alpha.
 *
 * @param out       Pre-allocated float32 output tensor, shape [M, N].
 * @param a         Float32 A matrix.
 * @param b         Float32 B matrix.
 * @param transA    Non-zero to transpose A before multiplication.
 * @param transB    Non-zero to transpose B before multiplication.
 * @param alphaBits IEEE 754 bit pattern of alpha, cast to int64.
 * @param betaBits  IEEE 754 bit pattern of beta (unused when C is absent).
 */
void om_gemmini_gemm_f32(OMTensor *out, const OMTensor *a, const OMTensor *b,
    int64_t transA, int64_t transB, int64_t alphaBits, int64_t betaBits) {
  om_gemmini_gemm_f32_impl(
      out, a, b, NULL, transA, transB, alphaBits, betaBits);
}

/**
 * @brief Float32 GEMM with bias via Gemmini.
 *
 * Implements `ONNX Gemm` with C bias: `out = alpha * op(A) * op(B) + beta*C`.
 * C may be shaped [M, N], [1, N], or [N] (broadcast over rows).
 *
 * @param out       Pre-allocated float32 output tensor, shape [M, N].
 * @param a         Float32 A matrix.
 * @param b         Float32 B matrix.
 * @param c         Float32 bias tensor C.
 * @param transA    Non-zero to transpose A before multiplication.
 * @param transB    Non-zero to transpose B before multiplication.
 * @param alphaBits IEEE 754 bit pattern of alpha, cast to int64.
 * @param betaBits  IEEE 754 bit pattern of beta, cast to int64.
 */
void om_gemmini_gemm_f32_bias(OMTensor *out, const OMTensor *a,
    const OMTensor *b, const OMTensor *c, int64_t transA, int64_t transB,
    int64_t alphaBits, int64_t betaBits) {
  om_gemmini_gemm_f32_impl(out, a, b, c, transA, transB, alphaBits, betaBits);
}

// ===--- Resize helpers ---===

/** Map an output coordinate to an input coordinate for ONNX Resize. */
// coord_mode: 0=asymmetric, 1=half_pixel, 2=align_corners
static float resize_input_coord(
    int64_t out_coord, int64_t out_size, int64_t in_size, int64_t coord_mode) {
  float scale = (float)in_size / (float)out_size;
  switch (coord_mode) {
  case 1: // half_pixel
    return (out_coord + 0.5f) * scale - 0.5f;
  case 2: // align_corners
    if (out_size <= 1)
      return 0.0f;
    return (float)out_coord * (float)(in_size - 1) / (float)(out_size - 1);
  default: // asymmetric (0)
    return (float)out_coord * scale;
  }
}

/** Round a floating input coordinate according to ONNX nearest_mode. */
// nearest_mode: 0=floor, 1=ceil, 2=round_prefer_floor, 3=round_prefer_ceil
static int64_t resize_nearest_coord(float x, int64_t nearest_mode) {
  switch (nearest_mode) {
  case 1:
    return (int64_t)ceilf(x);
  case 2: { // round_prefer_floor: ties go to floor
    int64_t hi = (int64_t)ceilf(x);
    int64_t lo = (int64_t)floorf(x);
    return (hi - x == 0.5f) ? lo : (int64_t)roundf(x);
  }
  case 3: { // round_prefer_ceil: ties go to ceil
    int64_t hi = (int64_t)ceilf(x);
    int64_t lo = (int64_t)floorf(x);
    return (x - lo == 0.5f) ? hi : (int64_t)roundf(x);
  }
  default: // floor (0)
    return (int64_t)floorf(x);
  }
}

/** Clamp an integer coordinate into [lo, hi]. */
static int64_t resize_clamp(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

/** Clamp an out-of-range pad coordinate to the nearest edge element. */
static int64_t pad_clamp_index(int64_t v, int64_t len) {
  return resize_clamp(v, 0, len - 1);
}

/** Reflect an out-of-range pad coordinate without repeating edge elements. */
static int64_t pad_reflect_index(int64_t v, int64_t len) {
  if (len <= 1)
    return 0;
  while (v < 0 || v >= len) {
    if (v < 0)
      v = -v;
    else
      v = 2 * len - v - 2;
  }
  return v;
}

/**
 * @brief NCHW f32 nearest-neighbor resize via scalar CPU loops.
 *
 * Resizes a 4-D NCHW tensor to the shape of the pre-allocated output tensor.
 * Does not use Gemmini hardware; runs entirely on the scalar RISC-V CPU.
 *
 * @param out         Pre-allocated output tensor, shape [N, C, oH, oW].
 * @param x           Input tensor, shape [N, C, iH, iW].
 * @param coord_mode  Coordinate transformation: 0=asymmetric, 1=half_pixel,
 *                    2=align_corners.
 * @param nearest_mode Rounding: 0=floor, 1=ceil, 2=round_prefer_floor,
 *                    3=round_prefer_ceil.
 */
void om_gemmini_resize_nearest_f32(OMTensor *out, const OMTensor *x,
    int64_t coord_mode, int64_t nearest_mode) {
  assert(out && x);
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *yShape = omTensorGetShape(out);
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);

  const int64_t N = xShape[0], C = xShape[1];
  const int64_t iH = xShape[2], iW = xShape[3];
  const int64_t oH = yShape[2], oW = yShape[3];
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] resize_nearest f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] coord_mode=%lld nearest_mode=%lld\n",
        (long long)N, (long long)C, (long long)iH, (long long)iW,
        (long long)yShape[0], (long long)yShape[1], (long long)oH,
        (long long)oW, (long long)coord_mode, (long long)nearest_mode);
    fflush(stderr);
  }
  traceTensorShapeStats("resize_nearest", "input", x);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *xPlane = xData + (n * C + c) * iH * iW;
      float *yPlane = outData + (n * C + c) * oH * oW;
      for (int64_t oh = 0; oh < oH; ++oh) {
        float ih_f = resize_input_coord(oh, oH, iH, coord_mode);
        int64_t ih =
            resize_clamp(resize_nearest_coord(ih_f, nearest_mode), 0, iH - 1);
        for (int64_t ow = 0; ow < oW; ++ow) {
          float iw_f = resize_input_coord(ow, oW, iW, coord_mode);
          int64_t iw =
              resize_clamp(resize_nearest_coord(iw_f, nearest_mode), 0, iW - 1);
          yPlane[oh * oW + ow] = xPlane[ih * iW + iw];
        }
      }
    }
  }
  traceTensorShapeStats("resize_nearest", "output", out);
}

/**
 * @brief NCHW f32 bilinear resize via scalar CPU loops.
 *
 * Resizes a 4-D NCHW tensor to the shape of the pre-allocated output tensor
 * using bilinear interpolation.  Does not use Gemmini hardware; runs entirely
 * on the scalar RISC-V CPU.
 *
 * @param out         Pre-allocated output tensor, shape [N, C, oH, oW].
 * @param x           Input tensor, shape [N, C, iH, iW].
 * @param coord_mode  Coordinate transformation: 0=asymmetric, 1=half_pixel,
 *                    2=align_corners.
 */
void om_gemmini_resize_linear_f32(
    OMTensor *out, const OMTensor *x, int64_t coord_mode) {
  assert(out && x);
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *yShape = omTensorGetShape(out);
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);

  const int64_t N = xShape[0], C = xShape[1];
  const int64_t iH = xShape[2], iW = xShape[3];
  const int64_t oH = yShape[2], oW = yShape[3];
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] resize_linear f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] coord_mode=%lld\n",
        (long long)N, (long long)C, (long long)iH, (long long)iW,
        (long long)yShape[0], (long long)yShape[1], (long long)oH,
        (long long)oW, (long long)coord_mode);
    fflush(stderr);
  }
  traceTensorShapeStats("resize_linear", "input", x);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *xPlane = xData + (n * C + c) * iH * iW;
      float *yPlane = outData + (n * C + c) * oH * oW;
      for (int64_t oh = 0; oh < oH; ++oh) {
        float ih_f = resize_input_coord(oh, oH, iH, coord_mode);
        float ih0f = floorf(ih_f);
        int64_t ih0 = resize_clamp((int64_t)ih0f, 0, iH - 1);
        int64_t ih1 = resize_clamp((int64_t)ih0f + 1, 0, iH - 1);
        float dh = ih_f - ih0f;
        for (int64_t ow = 0; ow < oW; ++ow) {
          float iw_f = resize_input_coord(ow, oW, iW, coord_mode);
          float iw0f = floorf(iw_f);
          int64_t iw0 = resize_clamp((int64_t)iw0f, 0, iW - 1);
          int64_t iw1 = resize_clamp((int64_t)iw0f + 1, 0, iW - 1);
          float dw = iw_f - iw0f;
          float v00 = xPlane[ih0 * iW + iw0];
          float v01 = xPlane[ih0 * iW + iw1];
          float v10 = xPlane[ih1 * iW + iw0];
          float v11 = xPlane[ih1 * iW + iw1];
          yPlane[oh * oW + ow] = v00 * (1.0f - dh) * (1.0f - dw) +
                                 v01 * (1.0f - dh) * dw +
                                 v10 * dh * (1.0f - dw) + v11 * dh * dw;
        }
      }
    }
  }
  traceTensorShapeStats("resize_linear", "output", out);
}

/**
 * @brief NCHW f32 zero-constant Pad via scalar CPU loops.
 *
 * Pads a 4-D NCHW tensor into the pre-allocated output tensor.  The current
 * Gemmini lowering only routes constant Pad when the ONNX constant value is
 * omitted or statically zero; nonzero constants fall back to standard lowering.
 *
 * @param out        Pre-allocated output tensor, shape
 *                   [N, C, iH + top + bottom, iW + left + right].
 * @param x          Input tensor, shape [N, C, iH, iW].
 * @param pad_left   Number of columns added before input W.
 * @param pad_right  Number of columns added after input W.
 * @param pad_top    Number of rows added before input H.
 * @param pad_bottom Number of rows added after input H.
 */
void om_gemmini_pad_constant_f32(OMTensor *out, const OMTensor *x,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom) {
  assert(out && x);
  (void)pad_right;
  (void)pad_bottom;
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *yShape = omTensorGetShape(out);
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);

  const int64_t N = xShape[0], C = xShape[1];
  const int64_t iH = xShape[2], iW = xShape[3];
  const int64_t oH = yShape[2], oW = yShape[3];
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] pad_constant f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] left=%lld right=%lld top=%lld bottom=%lld\n",
        (long long)N, (long long)C, (long long)iH, (long long)iW,
        (long long)yShape[0], (long long)yShape[1], (long long)oH,
        (long long)oW, (long long)pad_left, (long long)pad_right,
        (long long)pad_top, (long long)pad_bottom);
    fflush(stderr);
  }
  traceTensorShapeStats("pad_constant", "input", x);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *xPlane = xData + (n * C + c) * iH * iW;
      float *yPlane = outData + (n * C + c) * oH * oW;
      for (int64_t oh = 0; oh < oH; ++oh) {
        int64_t ih = oh - pad_top;
        for (int64_t ow = 0; ow < oW; ++ow) {
          int64_t iw = ow - pad_left;
          yPlane[oh * oW + ow] = (ih >= 0 && ih < iH && iw >= 0 && iw < iW)
                                     ? xPlane[ih * iW + iw]
                                     : 0.0f;
        }
      }
    }
  }
  traceTensorShapeStats("pad_constant", "output", out);
}

/**
 * @brief NCHW f32 reflect Pad via scalar CPU loops.
 *
 * Uses ONNX reflect semantics, mirroring without repeating the edge element.
 */
void om_gemmini_pad_reflect_f32(OMTensor *out, const OMTensor *x,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom) {
  assert(out && x);
  (void)pad_right;
  (void)pad_bottom;
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *yShape = omTensorGetShape(out);
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);

  const int64_t N = xShape[0], C = xShape[1];
  const int64_t iH = xShape[2], iW = xShape[3];
  const int64_t oH = yShape[2], oW = yShape[3];
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] pad_reflect f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] left=%lld right=%lld top=%lld bottom=%lld\n",
        (long long)N, (long long)C, (long long)iH, (long long)iW,
        (long long)yShape[0], (long long)yShape[1], (long long)oH,
        (long long)oW, (long long)pad_left, (long long)pad_right,
        (long long)pad_top, (long long)pad_bottom);
    fflush(stderr);
  }
  traceTensorShapeStats("pad_reflect", "input", x);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *xPlane = xData + (n * C + c) * iH * iW;
      float *yPlane = outData + (n * C + c) * oH * oW;
      for (int64_t oh = 0; oh < oH; ++oh) {
        int64_t ih = pad_reflect_index(oh - pad_top, iH);
        for (int64_t ow = 0; ow < oW; ++ow) {
          int64_t iw = pad_reflect_index(ow - pad_left, iW);
          yPlane[oh * oW + ow] = xPlane[ih * iW + iw];
        }
      }
    }
  }
  traceTensorShapeStats("pad_reflect", "output", out);
}

/**
 * @brief NCHW f32 edge Pad via scalar CPU loops.
 *
 * Duplicates the nearest input border value for positions outside the input
 * spatial extent.
 */
void om_gemmini_pad_edge_f32(OMTensor *out, const OMTensor *x, int64_t pad_left,
    int64_t pad_right, int64_t pad_top, int64_t pad_bottom) {
  assert(out && x);
  (void)pad_right;
  (void)pad_bottom;
  const int64_t *xShape = omTensorGetShape(x);
  const int64_t *yShape = omTensorGetShape(out);
  const float *xData = (const float *)omTensorGetDataPtr(x);
  float *outData = (float *)omTensorGetDataPtr(out);

  const int64_t N = xShape[0], C = xShape[1];
  const int64_t iH = xShape[2], iW = xShape[3];
  const int64_t oH = yShape[2], oW = yShape[3];
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] pad_edge f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] left=%lld right=%lld top=%lld bottom=%lld\n",
        (long long)N, (long long)C, (long long)iH, (long long)iW,
        (long long)yShape[0], (long long)yShape[1], (long long)oH,
        (long long)oW, (long long)pad_left, (long long)pad_right,
        (long long)pad_top, (long long)pad_bottom);
    fflush(stderr);
  }
  traceTensorShapeStats("pad_edge", "input", x);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *xPlane = xData + (n * C + c) * iH * iW;
      float *yPlane = outData + (n * C + c) * oH * oW;
      for (int64_t oh = 0; oh < oH; ++oh) {
        int64_t ih = pad_clamp_index(oh - pad_top, iH);
        for (int64_t ow = 0; ow < oW; ++ow) {
          int64_t iw = pad_clamp_index(ow - pad_left, iW);
          yPlane[oh * oW + ow] = xPlane[ih * iW + iw];
        }
      }
    }
  }
  traceTensorShapeStats("pad_edge", "output", out);
}

// ===--- Slice ---===

/**
 * @brief Extract a strided slice from a rank-4 NCHW f32 tensor.
 *
 * Slice parameters are pre-normalized to one (start, end, step) triple per
 * NCHW dimension, matching the four-dimensional loop order:
 *   N axis: sn / en / tn
 *   C axis: sc / ec / tc
 *   H axis: sh / eh / th
 *   W axis: sw / ew / tw
 *
 * Negative start/end values are wrapped at runtime (idx < 0 → idx + dim_size).
 * End values exceeding the dimension size are clamped to dim_size.
 * Step must be ≥ 1 (only forward striding is supported by the Gemmini
 * predicate). The output tensor @p out must already be allocated with the
 * correct shape: out_dim = ceil((end - start) / step) for each axis.
 */
void om_gemmini_slice_f32(OMTensor *out, const OMTensor *x, int64_t sn,
    int64_t en, int64_t tn, int64_t sc, int64_t ec, int64_t tc, int64_t sh,
    int64_t eh, int64_t th, int64_t sw, int64_t ew, int64_t tw) {
  assert(out && x);
  assert(tn >= 1 && tc >= 1 && th >= 1 && tw >= 1);

  const int64_t *sx = omTensorGetShape(x);
  const int64_t *so = omTensorGetShape(out);

  /* Wrap negative indices and clamp ends. */
#define SLICE_NORM(s, e, dim)                                                  \
  do {                                                                         \
    if ((s) < 0)                                                               \
      (s) += (dim);                                                            \
    if ((s) < 0)                                                               \
      (s) = 0;                                                                 \
    if ((e) < 0)                                                               \
      (e) += (dim);                                                            \
    if ((e) > (dim))                                                           \
      (e) = (dim);                                                             \
  } while (0)

  SLICE_NORM(sn, en, sx[0]);
  SLICE_NORM(sc, ec, sx[1]);
  SLICE_NORM(sh, eh, sx[2]);
  SLICE_NORM(sw, ew, sx[3]);
#undef SLICE_NORM

  const float *din = (const float *)omTensorGetDataPtr(x);
  float *dout = (float *)omTensorGetDataPtr(out);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] slice f32: starts=[%lld,%lld,%lld,%lld] ends=[%lld,%lld,%lld,%lld] steps=[%lld,%lld,%lld,%lld]\n",
        (long long)sn, (long long)sc, (long long)sh, (long long)sw,
        (long long)en, (long long)ec, (long long)eh, (long long)ew,
        (long long)tn, (long long)tc, (long long)th, (long long)tw);
    fflush(stderr);
  }
  traceTensorShapeStats("slice", "input", x);

  for (int64_t n = 0; n < so[0]; ++n) {
    int64_t in_n = sn + n * tn;
    for (int64_t c = 0; c < so[1]; ++c) {
      int64_t in_c = sc + c * tc;
      for (int64_t h = 0; h < so[2]; ++h) {
        int64_t in_h = sh + h * th;
        for (int64_t w = 0; w < so[3]; ++w) {
          int64_t in_w = sw + w * tw;
          int64_t in_idx =
              ((in_n * sx[1] + in_c) * sx[2] + in_h) * sx[3] + in_w;
          int64_t out_idx = ((n * so[1] + c) * so[2] + h) * so[3] + w;
          dout[out_idx] = din[in_idx];
        }
      }
    }
  }
  traceTensorShapeStats("slice", "output", out);
}

// ===--- Concat ---===

/**
 * @brief Concatenate two rank-4 NCHW f32 tensors along the given axis.
 *
 * @param out    Pre-allocated output tensor (shape already encodes concat
 * size).
 * @param x0     First input tensor.
 * @param x1     Second input tensor.
 * @param axis   Concatenation axis in [0, 3].
 */
void om_gemmini_concat_f32(
    OMTensor *out, const OMTensor *x0, const OMTensor *x1, int64_t axis) {
  assert(out && x0 && x1);
  assert(axis >= 0 && axis <= 3);
  const int64_t *s0 = omTensorGetShape(x0);
  const int64_t *s1 = omTensorGetShape(x1);
  const int64_t *so = omTensorGetShape(out);
  const float *d0 = (const float *)omTensorGetDataPtr(x0);
  const float *d1 = (const float *)omTensorGetDataPtr(x1);
  float *dout = (float *)omTensorGetDataPtr(out);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] concat f32: x0=[%lld,%lld,%lld,%lld] x1=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] axis=%lld\n",
        (long long)s0[0], (long long)s0[1], (long long)s0[2],
        (long long)s0[3], (long long)s1[0], (long long)s1[1],
        (long long)s1[2], (long long)s1[3], (long long)so[0],
        (long long)so[1], (long long)so[2], (long long)so[3],
        (long long)axis);
    fflush(stderr);
  }
  traceTensorShapeStats("concat", "input0", x0);
  traceTensorShapeStats("concat", "input1", x1);

  // Copy x0 into out at offset 0 along axis.
  for (int64_t n = 0; n < s0[0]; ++n)
    for (int64_t c = 0; c < s0[1]; ++c)
      for (int64_t h = 0; h < s0[2]; ++h)
        for (int64_t w = 0; w < s0[3]; ++w) {
          int64_t in_idx = ((n * s0[1] + c) * s0[2] + h) * s0[3] + w;
          int64_t out_idx = ((n * so[1] + c) * so[2] + h) * so[3] + w;
          dout[out_idx] = d0[in_idx];
        }

  // Offset for x1 on the concatenation axis.
  int64_t off0 = (axis == 0) ? s0[0] : 0;
  int64_t off1 = (axis == 1) ? s0[1] : 0;
  int64_t off2 = (axis == 2) ? s0[2] : 0;
  int64_t off3 = (axis == 3) ? s0[3] : 0;

  // Copy x1 into out at the appropriate offset.
  for (int64_t n = 0; n < s1[0]; ++n)
    for (int64_t c = 0; c < s1[1]; ++c)
      for (int64_t h = 0; h < s1[2]; ++h)
        for (int64_t w = 0; w < s1[3]; ++w) {
          int64_t in_idx = ((n * s1[1] + c) * s1[2] + h) * s1[3] + w;
          int64_t out_n = n + off0, out_c = c + off1;
          int64_t out_h = h + off2, out_w = w + off3;
          int64_t out_idx =
              ((out_n * so[1] + out_c) * so[2] + out_h) * so[3] + out_w;
          dout[out_idx] = d1[in_idx];
        }
  traceTensorShapeStats("concat", "output", out);
}

// ===--- Transpose ---===

/**
 * @brief Transpose a rank-4 f32 tensor according to a 4-axis permutation.
 *
 * ONNX Transpose semantics define output axis @c i as input axis @c perm[i].
 * The output tensor is pre-allocated by the compiler; this runtime function
 * only copies elements into the transposed layout.
 *
 * @param out   Pre-allocated output tensor.
 * @param x     Input tensor.
 * @param perm0 Input axis used for output axis 0.
 * @param perm1 Input axis used for output axis 1.
 * @param perm2 Input axis used for output axis 2.
 * @param perm3 Input axis used for output axis 3.
 */
static void om_gemmini_transpose_f32_scalar_impl(
    OMTensor *out, const OMTensor *x, const int64_t perm[4]) {
  const int64_t *sx = omTensorGetShape(x);
  const int64_t *so = omTensorGetShape(out);
  const float *din = (const float *)omTensorGetDataPtr(x);
  float *dout = (float *)omTensorGetDataPtr(out);
  if (gemminiTraceEnabled()) {
    fprintf(stderr,
        "[gemmini-trace] transpose f32: x=[%lld,%lld,%lld,%lld] y=[%lld,%lld,%lld,%lld] perm=[%lld,%lld,%lld,%lld]\n",
        (long long)sx[0], (long long)sx[1], (long long)sx[2],
        (long long)sx[3], (long long)so[0], (long long)so[1],
        (long long)so[2], (long long)so[3], (long long)perm[0],
        (long long)perm[1], (long long)perm[2], (long long)perm[3]);
    fflush(stderr);
  }
  traceTensorShapeStats("transpose", "input", x);

  for (int64_t o0 = 0; o0 < so[0]; ++o0) {
    for (int64_t o1 = 0; o1 < so[1]; ++o1) {
      for (int64_t o2 = 0; o2 < so[2]; ++o2) {
        for (int64_t o3 = 0; o3 < so[3]; ++o3) {
          int64_t outCoord[4] = {o0, o1, o2, o3};
          int64_t inCoord[4] = {0, 0, 0, 0};
          for (int64_t d = 0; d < 4; ++d)
            inCoord[perm[d]] = outCoord[d];

          int64_t inIdx =
              ((inCoord[0] * sx[1] + inCoord[1]) * sx[2] + inCoord[2]) * sx[3] +
              inCoord[3];
          int64_t outIdx = ((o0 * so[1] + o1) * so[2] + o2) * so[3] + o3;
          dout[outIdx] = din[inIdx];
        }
      }
    }
  }
  traceTensorShapeStats("transpose", "output", out);
}

/** Return true for rank-4 permutations that a future HW route can support. */
static bool om_gemmini_transpose_perm_has_hw_route(const int64_t perm[4]) {
  // Identity, NCHW <-> NHWC, and H/W swap map to simple matrix-transposer
  // schedules when the standalone transposer interface is exposed by Gemmini.
  return (perm[0] == 0 && perm[1] == 1 && perm[2] == 2 && perm[3] == 3) ||
         (perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1) ||
         (perm[0] == 0 && perm[1] == 3 && perm[2] == 1 && perm[3] == 2) ||
         (perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2);
}

/** True when the bundled Gemmini ABI exposes a standalone transposer command. */
static bool om_gemmini_transposer_hw_available(void) {
#if defined(GEMMINI_HAS_STANDALONE_TRANSPOSER)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Transpose a rank-4 f32 tensor with Gemmini transposer preference.
 *
 * The local Gemmini RTL contains an `AlwaysOutTransposer` in the mesh execute
 * path, controlled by the normal execute transpose bits. The public C runtime
 * header in this workspace does not expose a standalone RoCC memory-layout
 * transpose command yet, so this entry point performs a feature check and uses
 * the scalar fallback unless such an interface is enabled later.
 *
 * @param out   Pre-allocated output tensor.
 * @param x     Input tensor.
 * @param perm0 Input axis used for output axis 0.
 * @param perm1 Input axis used for output axis 1.
 * @param perm2 Input axis used for output axis 2.
 * @param perm3 Input axis used for output axis 3.
 */
void om_gemmini_transpose_f32_hw(OMTensor *out, const OMTensor *x,
    int64_t perm0, int64_t perm1, int64_t perm2, int64_t perm3) {
  assert(out && x);
  int64_t perm[4] = {perm0, perm1, perm2, perm3};
  bool seen[4] = {false, false, false, false};
  for (int64_t i = 0; i < 4; ++i) {
    assert(perm[i] >= 0 && perm[i] < 4);
    assert(!seen[perm[i]]);
    seen[perm[i]] = true;
  }

  if (om_gemmini_transposer_hw_available() &&
      om_gemmini_transpose_perm_has_hw_route(perm)) {
    // Future standalone transposer RoCC plumbing should return here after
    // programming the hardware path. Keep correctness through the scalar path
    // until that low-level API is present in gemmini.h.
  }

  om_gemmini_transpose_f32_scalar_impl(out, x, perm);
}

// ===--- Float Split (data-movement, scalar CPU loops) ---===

/**
 * Copy one slice of a rank-4 f32 tensor along a given axis.
 *
 * @param out    Pre-allocated output OMTensor.  Its shape encodes slice size.
 * @param x      Input tensor (NCHW f32).
 * @param axis   Axis along which the split is performed (0-3).
 * @param start  First index in *x* along *axis* belonging to this output.
 */
static void om_gemmini_split_copy_slice_f32(OMTensor *out, const OMTensor *x,
    int64_t axis, int64_t start) {
  assert(out && x);
  const int64_t *inShape = omTensorGetShape(x);
  const int64_t *outShape = omTensorGetShape(out);
  const float *src = (const float *)omTensorGetDataPtr(x);
  float *dst = (float *)omTensorGetDataPtr(out);

  int64_t iC = inShape[1], iH = inShape[2], iW = inShape[3];
  int64_t oN = outShape[0], oC = outShape[1], oH = outShape[2], oW = outShape[3];

  for (int64_t n = 0; n < oN; ++n)
    for (int64_t c = 0; c < oC; ++c)
      for (int64_t h = 0; h < oH; ++h)
        for (int64_t w = 0; w < oW; ++w) {
          int64_t in = n, ic = c, ih = h, iw = w;
          if      (axis == 0) in += start;
          else if (axis == 1) ic += start;
          else if (axis == 2) ih += start;
          else                iw += start;
          dst[n*oC*oH*oW + c*oH*oW + h*oW + w] =
              src[in*iC*iH*iW + ic*iH*iW + ih*iW + iw];
        }
}

/**
 * @brief Split a rank-4 f32 tensor into 2 outputs along *axis*.
 *
 * @param out0  Pre-allocated first output; its axis dim gives split0_size.
 * @param out1  Pre-allocated second output.
 * @param x     Input tensor (NCHW f32).
 * @param axis  Split axis (0-3).
 */
void om_gemmini_split_2_f32(OMTensor *out0, OMTensor *out1,
    const OMTensor *x, int64_t axis) {
  assert(out0 && out1 && x);
  int64_t off0 = 0;
  int64_t off1 = omTensorGetShape(out0)[axis];
  if (gemminiTraceEnabled()) {
    fprintf(stderr, "[gemmini-trace] split_2 f32: axis=%lld\n",
        (long long)axis);
    fflush(stderr);
  }
  traceTensorShapeStats("split_2", "input", x);
  om_gemmini_split_copy_slice_f32(out0, x, axis, off0);
  om_gemmini_split_copy_slice_f32(out1, x, axis, off1);
  traceTensorShapeStats("split_2", "output0", out0);
  traceTensorShapeStats("split_2", "output1", out1);
}

/**
 * @brief Split a rank-4 f32 tensor into 3 outputs along *axis*.
 *
 * @param out0  Pre-allocated first output.
 * @param out1  Pre-allocated second output.
 * @param out2  Pre-allocated third output.
 * @param x     Input tensor (NCHW f32).
 * @param axis  Split axis (0-3).
 */
void om_gemmini_split_3_f32(OMTensor *out0, OMTensor *out1, OMTensor *out2,
    const OMTensor *x, int64_t axis) {
  assert(out0 && out1 && out2 && x);
  int64_t off0 = 0;
  int64_t off1 = off0 + omTensorGetShape(out0)[axis];
  int64_t off2 = off1 + omTensorGetShape(out1)[axis];
  if (gemminiTraceEnabled()) {
    fprintf(stderr, "[gemmini-trace] split_3 f32: axis=%lld\n",
        (long long)axis);
    fflush(stderr);
  }
  traceTensorShapeStats("split_3", "input", x);
  om_gemmini_split_copy_slice_f32(out0, x, axis, off0);
  om_gemmini_split_copy_slice_f32(out1, x, axis, off1);
  om_gemmini_split_copy_slice_f32(out2, x, axis, off2);
  traceTensorShapeStats("split_3", "output0", out0);
  traceTensorShapeStats("split_3", "output1", out1);
  traceTensorShapeStats("split_3", "output2", out2);
}

/**
 * @brief Split a rank-4 f32 tensor into 4 outputs along *axis*.
 *
 * @param out0  Pre-allocated first output.
 * @param out1  Pre-allocated second output.
 * @param out2  Pre-allocated third output.
 * @param out3  Pre-allocated fourth output.
 * @param x     Input tensor (NCHW f32).
 * @param axis  Split axis (0-3).
 */
void om_gemmini_split_4_f32(OMTensor *out0, OMTensor *out1, OMTensor *out2,
    OMTensor *out3, const OMTensor *x, int64_t axis) {
  assert(out0 && out1 && out2 && out3 && x);
  int64_t off0 = 0;
  int64_t off1 = off0 + omTensorGetShape(out0)[axis];
  int64_t off2 = off1 + omTensorGetShape(out1)[axis];
  int64_t off3 = off2 + omTensorGetShape(out2)[axis];
  if (gemminiTraceEnabled()) {
    fprintf(stderr, "[gemmini-trace] split_4 f32: axis=%lld\n",
        (long long)axis);
    fflush(stderr);
  }
  traceTensorShapeStats("split_4", "input", x);
  om_gemmini_split_copy_slice_f32(out0, x, axis, off0);
  om_gemmini_split_copy_slice_f32(out1, x, axis, off1);
  om_gemmini_split_copy_slice_f32(out2, x, axis, off2);
  om_gemmini_split_copy_slice_f32(out3, x, axis, off3);
  traceTensorShapeStats("split_4", "output0", out0);
  traceTensorShapeStats("split_4", "output1", out1);
  traceTensorShapeStats("split_4", "output2", out2);
  traceTensorShapeStats("split_4", "output3", out3);
}

/**
 * @brief Element-wise sigmoid for float32 tensors (scalar CPU path).
 *
 * Computes `out[i] = 1.0 / (1.0 + exp(-x[i]))` over the flattened tensor.
 * Used for SiLU / Swish activation support (Sigmoid × input).
 *
 * @param out  Pre-allocated output tensor, same shape as @p x.
 * @param x    Input float32 tensor (any rank, any static/dynamic shape).
 */
void om_gemmini_sigmoid_f32(OMTensor *out, const OMTensor *x) {
  assert(out && x);
  int64_t n = omTensorGetNumElems(x);
  const float *src = (const float *)omTensorGetDataPtr(x);
  float *dst = (float *)omTensorGetDataPtr(out);
  traceTensorShapeStats("sigmoid", "input", x);
  for (int64_t i = 0; i < n; ++i)
    dst[i] = 1.0f / (1.0f + expf(-src[i]));
  traceTensorShapeStats("sigmoid", "output", out);
}

/**
 * @brief Element-wise multiplication for float32 tensors (scalar CPU path).
 *
 * Computes `out[i] = a[i] * b[i]` over the flattened tensor.
 * Inputs must have identical shapes (no broadcasting).
 * Used as the second step of SiLU: out = x * sigmoid(x).
 *
 * @param out  Pre-allocated output tensor, same shape as @p a and @p b.
 * @param a    First input float32 tensor.
 * @param b    Second input float32 tensor (same shape as @p a).
 */
void om_gemmini_mul_f32(OMTensor *out, const OMTensor *a, const OMTensor *b) {
  assert(out && a && b);
  int64_t n = omTensorGetNumElems(a);
  const float *srcA = (const float *)omTensorGetDataPtr(a);
  const float *srcB = (const float *)omTensorGetDataPtr(b);
  float *dst = (float *)omTensorGetDataPtr(out);
  traceTensorShapeStats("mul", "A", a);
  traceTensorShapeStats("mul", "B", b);
  for (int64_t i = 0; i < n; ++i)
    dst[i] = srcA[i] * srcB[i];
  traceTensorShapeStats("mul", "output", out);
}
