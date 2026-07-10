/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ONNXToGemmini.cpp - ONNX to Gemmini Conversion ---------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "src/Accelerators/Gemmini/Conversion/ONNXToGemmini/ONNXToGemmini.hpp"
#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiOps.hpp"
#include "src/Accelerators/Gemmini/Pass/GemminiPasses.hpp"
#include "src/Accelerators/Gemmini/Support/GemminiTargetInfo.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include <cstring>

using namespace mlir;

namespace onnx_mlir {
namespace {

// ===--- Integer / quantized path predicates ---===


static bool isSupportedGemminiMatMul(ONNXMatMulOp op) {
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!aTy || !bTy || !yTy)
    return false;
  if (!aTy.hasStaticShape() || !bTy.hasStaticShape() || !yTy.hasStaticShape())
    return false;
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || yTy.getRank() != 2)
    return false;
  // Only i32 uses the direct Gemmini RoCC path; float is handled separately.
  if (!aTy.getElementType().isInteger(32))
    return false;
  if (aTy.getShape()[1] != bTy.getShape()[0] ||
      yTy.getShape()[0] != aTy.getShape()[0] ||
      yTy.getShape()[1] != bTy.getShape()[1])
    return false;
  return true;
}

static bool isStaticRankedTensorOfType(
    Value value, int64_t rank, Type elementType) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  return tensorType && tensorType.hasStaticShape() &&
         tensorType.getRank() == rank &&
         tensorType.getElementType() == elementType;
}

static bool isStaticScalarTensorOfType(Value value, Type elementType) {
  return isStaticRankedTensorOfType(value, 0, elementType);
}

static bool isSupportedMatMulZeroPoint(
    Value zeroPoint, Type elementType, int64_t expectedLength) {
  if (isNoneValue(zeroPoint))
    return true;
  auto tensorType = dyn_cast<RankedTensorType>(zeroPoint.getType());
  if (!tensorType || !tensorType.hasStaticShape() ||
      tensorType.getElementType() != elementType)
    return false;
  if (tensorType.getRank() == 0)
    return true;
  if (tensorType.getRank() != 1)
    return false;
  int64_t length = tensorType.getShape()[0];
  return length == 1 || length == expectedLength;
}

// Runtime path for INT8 MatMulInteger: rank-2 static i8×i8→i32, scalar or per-channel zero points.
static bool isSupportedGemminiMatMulInteger(ONNXMatMulIntegerOp op) {
  MLIRContext *ctx = op.getContext();
  auto i8Type = IntegerType::get(ctx, 8);
  auto i32Type = IntegerType::get(ctx, 32);
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!aTy || !bTy || !yTy)
    return false;
  if (!aTy.hasStaticShape() || !bTy.hasStaticShape() || !yTy.hasStaticShape())
    return false;
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || yTy.getRank() != 2)
    return false;
  if (aTy.getElementType() != i8Type || bTy.getElementType() != i8Type ||
      yTy.getElementType() != i32Type)
    return false;
  if (aTy.getShape()[1] != bTy.getShape()[0] ||
      yTy.getShape()[0] != aTy.getShape()[0] ||
      yTy.getShape()[1] != bTy.getShape()[1])
    return false;
  return isSupportedMatMulZeroPoint(
             op.getAZeroPoint(), i8Type, aTy.getShape()[0]) &&
         isSupportedMatMulZeroPoint(
             op.getBZeroPoint(), i8Type, bTy.getShape()[1]);
}

// Direct GemminiLow path for DIM×DIM INT8 MatMulInteger without zero points.
// Higher-benefit than the runtime path so it wins on exact-tile shapes.
static bool isSupportedGemminiDirectMatMulInteger(ONNXMatMulIntegerOp op) {
  constexpr int64_t dim = gemmini::GemminiTargetInfo::dim;
  if (!isSupportedGemminiMatMulInteger(op))
    return false;
  if (!isNoneValue(op.getAZeroPoint()) || !isNoneValue(op.getBZeroPoint()))
    return false;
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto bTy = cast<RankedTensorType>(op.getB().getType());
  // Restrict to shapes that fit in exactly one DIM×DIM tile so that the
  // existing runtime path handles larger shapes unchanged.
  return aTy.getShape()[0] == dim && aTy.getShape()[1] == dim &&
         bTy.getShape()[0] == dim && bTy.getShape()[1] == dim;
}

// INT8 QLinearConv: rank-4 NCHW i8 input/weight, f32 scales, optional i32 bias, groups=1.
static bool isSupportedGemminiQLinearConv(ONNXQLinearConvOp op) {
  MLIRContext *ctx = op.getContext();
  auto i8Type = IntegerType::get(ctx, 8);
  auto i32Type = IntegerType::get(ctx, 32);
  auto f32Type = Float32Type::get(ctx);
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  auto wTy = dyn_cast<RankedTensorType>(op.getW().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!xTy || !wTy || !yTy)
    return false;
  if (!xTy.hasStaticShape() || !wTy.hasStaticShape() || !yTy.hasStaticShape())
    return false;
  if (xTy.getRank() != 4 || wTy.getRank() != 4 || yTy.getRank() != 4)
    return false;
  if (xTy.getElementType() != i8Type || wTy.getElementType() != i8Type ||
      yTy.getElementType() != i8Type)
    return false;
  if (!isStaticScalarTensorOfType(op.getXScale(), f32Type) ||
      !isStaticScalarTensorOfType(op.getWScale(), f32Type) ||
      !isStaticScalarTensorOfType(op.getYScale(), f32Type) ||
      !isStaticScalarTensorOfType(op.getXZeroPoint(), i8Type) ||
      !isStaticScalarTensorOfType(op.getWZeroPoint(), i8Type) ||
      !isStaticScalarTensorOfType(op.getYZeroPoint(), i8Type))
    return false;
  if (op.getGroup() != 1)
    return false;
  ArrayRef<int64_t> xShape = xTy.getShape();
  ArrayRef<int64_t> wShape = wTy.getShape();
  ArrayRef<int64_t> yShape = yTy.getShape();
  if (xShape[0] != 1 || yShape[0] != 1)
    return false;
  if (wShape[1] != xShape[1] || wShape[2] != wShape[3])
    return false;
  if (yShape[1] != wShape[0])
    return false;
  if (auto kernelShape = op.getKernelShape()) {
    if (kernelShape->size() != 2)
      return false;
    int64_t kh = cast<IntegerAttr>((*kernelShape)[0]).getInt();
    int64_t kw = cast<IntegerAttr>((*kernelShape)[1]).getInt();
    if (kh != wShape[2] || kw != wShape[3])
      return false;
  }
  int64_t stride = 1;
  if (auto strides = op.getStrides()) {
    if (strides->size() != 2)
      return false;
    int64_t sh = cast<IntegerAttr>((*strides)[0]).getInt();
    int64_t sw = cast<IntegerAttr>((*strides)[1]).getInt();
    if (sh <= 0 || sh != sw)
      return false;
    stride = sh;
  }
  int64_t padding = 0;
  if (auto pads = op.getPads()) {
    if (pads->size() != 4)
      return false;
    int64_t padTop = cast<IntegerAttr>((*pads)[0]).getInt();
    int64_t padLeft = cast<IntegerAttr>((*pads)[1]).getInt();
    int64_t padBottom = cast<IntegerAttr>((*pads)[2]).getInt();
    int64_t padRight = cast<IntegerAttr>((*pads)[3]).getInt();
    if (padTop < 0 || padTop != padLeft || padTop != padBottom ||
        padTop != padRight)
      return false;
    padding = padTop;
  }
  int64_t expectedH = (xShape[2] + 2 * padding - wShape[2]) / stride + 1;
  int64_t expectedW = (xShape[3] + 2 * padding - wShape[3]) / stride + 1;
  if (expectedH != yShape[2] || expectedW != yShape[3])
    return false;
  if (isNoneValue(op.getB()))
    return true;
  auto biasTy = dyn_cast<RankedTensorType>(op.getB().getType());
  return biasTy && biasTy.hasStaticShape() && biasTy.getRank() == 1 &&
         biasTy.getElementType() == i32Type &&
         biasTy.getShape()[0] == wShape[0];
}

// ===--- Float path helpers ---===

static bool isF32StaticTensor(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.hasStaticShape() && ty.getElementType().isF32();
}

static bool hasStaticShapeExceptDims(
    RankedTensorType type, ArrayRef<int64_t> dynamicDims) {
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (!type.isDynamicDim(i))
      continue;
    if (!llvm::is_contained(dynamicDims, i))
      return false;
  }
  return true;
}

static Value createAllocLikeResult(ConversionPatternRewriter &rewriter,
    Location loc, MemRefType resultMemRef, Value dynamicDimSource) {
  SmallVector<Value, 4> dynamicDims;
  for (int64_t i = 0, e = resultMemRef.getRank(); i < e; ++i) {
    if (!ShapedType::isDynamic(resultMemRef.getDimSize(i)))
      continue;
    if (!dynamicDimSource)
      return Value();
    Value dim = arith::ConstantIndexOp::create(rewriter, loc, i);
    dynamicDims.push_back(
        memref::DimOp::create(rewriter, loc, dynamicDimSource, dim));
  }
  return memref::AllocOp::create(rewriter, loc, resultMemRef, dynamicDims);
}

// Compute ONNX Conv output spatial size at runtime:
//   out = (in_dim + 2*pad - kernel) / stride + 1
// Uses i64 arithmetic to correctly handle the signed constant (2*pad - kernel)
// which is negative when the kernel is larger than the total padding.
static Value computeConvOutSpatial(ConversionPatternRewriter &rewriter,
    Location loc, Value inDim, int64_t kernel, int64_t pad, int64_t stride) {
  auto i64Ty = rewriter.getI64Type();
  auto idxTy = rewriter.getIndexType();
  Value in64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, inDim);
  Value adj = arith::ConstantOp::create(
      rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(2 * pad - kernel));
  Value str = arith::ConstantOp::create(
      rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(stride));
  Value one = arith::ConstantOp::create(
      rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(1));
  Value num = arith::AddIOp::create(rewriter, loc, in64, adj);
  Value div = arith::DivSIOp::create(rewriter, loc, num, str);
  Value out64 = arith::AddIOp::create(rewriter, loc, div, one);
  return arith::IndexCastOp::create(rewriter, loc, idxTy, out64);
}

// Allocate a Conv2D output memref, correctly computing any dynamic dims:
//   dim 0 (batch):    copy from input batch dim
//   dim 2 (out H):    (in_H + 2*pad - kH) / stride + 1  [ONNX formula]
//   dim 3 (out W):    (in_W + 2*pad - kW) / stride + 1
//   dim 1 (channels): must be static (required by isSupportedGemminiConvF32)
static Value createConvOutputAlloc(ConversionPatternRewriter &rewriter,
    Location loc, MemRefType resultMemRef, Value inputMemref, int64_t kH,
    int64_t kW, int64_t stride, int64_t pad) {
  SmallVector<Value, 4> dynamicDims;
  for (int64_t i = 0, e = resultMemRef.getRank(); i < e; ++i) {
    if (!ShapedType::isDynamic(resultMemRef.getDimSize(i)))
      continue;
    Value dimIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
    if (i == 0) {
      dynamicDims.push_back(
          memref::DimOp::create(rewriter, loc, inputMemref, dimIdx));
    } else if (i == 2) {
      Value inH = memref::DimOp::create(rewriter, loc, inputMemref, dimIdx);
      dynamicDims.push_back(
          computeConvOutSpatial(rewriter, loc, inH, kH, pad, stride));
    } else if (i == 3) {
      Value inW = memref::DimOp::create(rewriter, loc, inputMemref, dimIdx);
      dynamicDims.push_back(
          computeConvOutSpatial(rewriter, loc, inW, kW, pad, stride));
    } else {
      return Value(); // dim 1 (channels) must always be static
    }
  }
  return memref::AllocOp::create(rewriter, loc, resultMemRef, dynamicDims);
}

// Encode a compile-time float as its IEEE 754 bit pattern in an i64 so it
// can be passed through a KrnlCallOp integer operand and decoded on the C
// side with memcpy(). The float bits occupy the low 32 bits of the i64.
static Value f32AsI64Bits(
    ConversionPatternRewriter &rewriter, Location loc, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(float));
  return arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(static_cast<int64_t>(bits)));
}

// ===--- Float Conv2D predicate ---===

static bool isSupportedGemminiConvF32(ONNXConvOp op) {
  auto f32 = Float32Type::get(op.getContext());
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  auto wTy = dyn_cast<RankedTensorType>(op.getW().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!xTy || !wTy || !yTy)
    return false;
  if (xTy.getRank() != 4 || wTy.getRank() != 4 || yTy.getRank() != 4)
    return false;
  if (xTy.getElementType() != f32 || wTy.getElementType() != f32 ||
      yTy.getElementType() != f32)
    return false;
  if (!wTy.hasStaticShape())
    return false;
  // Allow dynamic batch (dim 0) and dynamic spatial (dims 2, 3).
  // Channels (dim 1) must be static; weights are always fully static.
  if (!hasStaticShapeExceptDims(xTy, {0, 2, 3}) ||
      !hasStaticShapeExceptDims(yTy, {0, 2, 3}))
    return false;
  if (op.getGroup() != 1)
    return false;
  if (!xTy.isDynamicDim(0) && !yTy.isDynamicDim(0) &&
      xTy.getShape()[0] != yTy.getShape()[0])
    return false;
  if (!isNoneValue(op.getB())) {
    auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
    if (!bTy || !bTy.hasStaticShape() || bTy.getRank() != 1 ||
        bTy.getElementType() != f32)
      return false;
    if (bTy.getShape()[0] != wTy.getShape()[0])
      return false;
  }
  ArrayRef<int64_t> xShape = xTy.getShape();
  ArrayRef<int64_t> wShape = wTy.getShape();
  ArrayRef<int64_t> yShape = yTy.getShape();
  if (wShape[1] != xShape[1] || yShape[1] != wShape[0])
    return false;
  if (auto dilations = op.getDilations()) {
    for (auto dAttr : *dilations)
      if (cast<IntegerAttr>(dAttr).getInt() != 1)
        return false;
  }
  if (auto strides = op.getStrides()) {
    if (strides->size() != 2)
      return false;
    int64_t sh = cast<IntegerAttr>((*strides)[0]).getInt();
    int64_t sw = cast<IntegerAttr>((*strides)[1]).getInt();
    if (sh != sw || sh <= 0)
      return false;
  }
  if (auto pads = op.getPads()) {
    if (pads->size() != 4)
      return false;
    int64_t pt = cast<IntegerAttr>((*pads)[0]).getInt();
    int64_t pl = cast<IntegerAttr>((*pads)[1]).getInt();
    int64_t pb = cast<IntegerAttr>((*pads)[2]).getInt();
    int64_t pr = cast<IntegerAttr>((*pads)[3]).getInt();
    if (pt != pl || pt != pb || pt != pr || pt < 0)
      return false;
  }
  int64_t stride = 1;
  if (auto strides = op.getStrides())
    stride = cast<IntegerAttr>((*strides)[0]).getInt();
  int64_t pad = 0;
  if (auto pads = op.getPads())
    pad = cast<IntegerAttr>((*pads)[0]).getInt();
  // Only verify spatial output shape when both input and output spatial dims
  // are static. Dynamic spatial dims are computed at runtime by the lowering.
  if (!xTy.isDynamicDim(2) && !yTy.isDynamicDim(2)) {
    int64_t expectedH = (xShape[2] + 2 * pad - wShape[2]) / stride + 1;
    if (expectedH != yShape[2])
      return false;
  }
  if (!xTy.isDynamicDim(3) && !yTy.isDynamicDim(3)) {
    int64_t expectedW = (xShape[3] + 2 * pad - wShape[3]) / stride + 1;
    if (expectedW != yShape[3])
      return false;
  }
  return true;
}

// ===--- Float ConvTranspose predicate ---===

static bool isSupportedGemminiConvTransposeF32(ONNXConvTransposeOp op) {
  auto f32 = Float32Type::get(op.getContext());
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  auto wTy = dyn_cast<RankedTensorType>(op.getW().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!xTy || !wTy || !yTy)
    return false;
  if (xTy.getRank() != 4 || wTy.getRank() != 4 || yTy.getRank() != 4)
    return false;
  if (xTy.getElementType() != f32 || wTy.getElementType() != f32 ||
      yTy.getElementType() != f32)
    return false;
  // Weights must be fully static; allow dynamic batch + spatial in x and y.
  if (!wTy.hasStaticShape())
    return false;
  if (!hasStaticShapeExceptDims(xTy, {0, 2, 3}) ||
      !hasStaticShapeExceptDims(yTy, {0, 2, 3}))
    return false;
  if (op.getGroup() != 1)
    return false;
  // Reject explicit output_shape override (empty array = not set = OK).
  if (auto outShape = op.getOutputShape())
    if (outShape->size() > 0)
      return false;
  // Reject non-identity auto_pad.
  auto autoPad = op.getAutoPad();
  if (!autoPad.empty() && autoPad != "NOTSET")
    return false;
  // Channel consistency: w[0]=C (input), w[1]=M (output) — always static.
  ArrayRef<int64_t> wShape = wTy.getShape();
  if (!xTy.isDynamicDim(1) && xTy.getShape()[1] != wShape[0])
    return false;
  if (!yTy.isDynamicDim(1) && yTy.getShape()[1] != wShape[1])
    return false;
  // Dilations must all be 1.
  if (auto dilations = op.getDilations())
    for (auto dAttr : *dilations)
      if (cast<IntegerAttr>(dAttr).getInt() != 1)
        return false;
  // Equal H/W stride (>0).
  if (auto strides = op.getStrides()) {
    if (strides->size() != 2)
      return false;
    int64_t sh = cast<IntegerAttr>((*strides)[0]).getInt();
    int64_t sw = cast<IntegerAttr>((*strides)[1]).getInt();
    if (sh != sw || sh <= 0)
      return false;
  }
  // Symmetric pads (all four equal, >= 0).
  if (auto pads = op.getPads()) {
    if (pads->size() != 4)
      return false;
    int64_t pt = cast<IntegerAttr>((*pads)[0]).getInt();
    int64_t pl = cast<IntegerAttr>((*pads)[1]).getInt();
    int64_t pb = cast<IntegerAttr>((*pads)[2]).getInt();
    int64_t pr = cast<IntegerAttr>((*pads)[3]).getInt();
    if (pt != pl || pt != pb || pt != pr || pt < 0)
      return false;
  }
  // Equal H/W output_padding (>=0).
  if (auto opads = op.getOutputPadding()) {
    if (opads->size() != 2)
      return false;
    int64_t oph = cast<IntegerAttr>((*opads)[0]).getInt();
    int64_t opw = cast<IntegerAttr>((*opads)[1]).getInt();
    if (oph != opw || oph < 0)
      return false;
  }
  // Validate static spatial output dims against the ONNX ConvTranspose formula:
  //   O = (I - 1) * stride - 2 * pad + kSize + output_pad
  int64_t stride = 1;
  if (auto strides = op.getStrides())
    stride = cast<IntegerAttr>((*strides)[0]).getInt();
  int64_t pad = 0;
  if (auto pads = op.getPads())
    pad = cast<IntegerAttr>((*pads)[0]).getInt();
  int64_t output_pad = 0;
  if (auto opads = op.getOutputPadding())
    output_pad = cast<IntegerAttr>((*opads)[0]).getInt();
  ArrayRef<int64_t> xShape = xTy.getShape();
  ArrayRef<int64_t> yShape = yTy.getShape();
  int64_t kH = wShape[2], kW = wShape[3];
  if (!xTy.isDynamicDim(2) && !yTy.isDynamicDim(2)) {
    int64_t expectedH = (xShape[2] - 1) * stride - 2 * pad + kH + output_pad;
    if (expectedH != yShape[2])
      return false;
  }
  if (!xTy.isDynamicDim(3) && !yTy.isDynamicDim(3)) {
    int64_t expectedW = (xShape[3] - 1) * stride - 2 * pad + kW + output_pad;
    if (expectedW != yShape[3])
      return false;
  }
  // Validate optional bias.
  if (!isNoneValue(op.getB())) {
    auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
    if (!bTy || !bTy.hasStaticShape() || bTy.getRank() != 1 ||
        bTy.getElementType() != f32)
      return false;
    if (bTy.getShape()[0] != wShape[1])
      return false;
  }
  return true;
}

// Compute ONNX ConvTranspose output spatial size at runtime:
//   out = in * stride + (kernel - stride - 2*pad + output_pad)
// This is equivalent to (in - 1) * stride + kernel - 2*pad + output_pad.
// Uses i64 multiplication so the intermediate product does not overflow index.
static Value computeConvTransposeOutSpatial(ConversionPatternRewriter &rewriter,
    Location loc, Value inDim, int64_t kernel, int64_t pad, int64_t stride,
    int64_t output_pad) {
  auto i64Ty = rewriter.getI64Type();
  auto idxTy = rewriter.getIndexType();
  Value in64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, inDim);
  Value strideVal = arith::ConstantOp::create(
      rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(stride));
  Value adj = arith::ConstantOp::create(rewriter, loc, i64Ty,
      rewriter.getI64IntegerAttr(kernel - stride - 2 * pad + output_pad));
  Value prod = arith::MulIOp::create(rewriter, loc, in64, strideVal);
  Value out64 = arith::AddIOp::create(rewriter, loc, prod, adj);
  return arith::IndexCastOp::create(rewriter, loc, idxTy, out64);
}

// ===--- Float Resize predicate and helpers ---===

// coord_mode codes (passed to runtime as int64):
//   0 = asymmetric, 1 = half_pixel (default), 2 = align_corners
static int64_t resizeCoordModeCode(ONNXResizeOp op) {
  if (auto attr = op.getCoordinateTransformationModeAttr()) {
    StringRef ctm = attr.getValue();
    if (ctm == "asymmetric")
      return 0;
    if (ctm == "align_corners")
      return 2;
  }
  return 1; // half_pixel
}

// nearest_mode codes (passed to runtime as int64):
//   0 = floor, 1 = ceil, 2 = round_prefer_floor (default), 3 =
//   round_prefer_ceil
static int64_t resizeNearestModeCode(ONNXResizeOp op) {
  if (auto attr = op.getNearestModeAttr()) {
    StringRef nm = attr.getValue();
    if (nm == "floor")
      return 0;
    if (nm == "ceil")
      return 1;
    if (nm == "round_prefer_ceil")
      return 3;
  }
  return 2; // round_prefer_floor (ONNX default for nearest)
}

struct GemminiPadSpec {
  int64_t left;
  int64_t right;
  int64_t top;
  int64_t bottom;
};

static bool extractGemminiPadSpec(ONNXPadOp op, GemminiPadSpec &spec) {
  SmallVector<int64_t, 8> pads;
  if (!getI64ValuesFromONNXConstantOp(op.getPads(), pads) || pads.size() != 8)
    return false;

  // ONNX pads are [N_begin, C_begin, H_begin, W_begin,
  //                N_end,   C_end,   H_end,   W_end].
  // Gemmini Pad currently supports spatial NCHW padding only.
  if (pads[0] != 0 || pads[1] != 0 || pads[4] != 0 || pads[5] != 0)
    return false;
  spec.top = pads[2];
  spec.left = pads[3];
  spec.bottom = pads[6];
  spec.right = pads[7];
  return spec.left >= 0 && spec.right >= 0 && spec.top >= 0 && spec.bottom >= 0;
}

static bool isStaticZeroF32Constant(Value value) {
  if (isNoneValue(value))
    return true;
  auto valueTy = dyn_cast<RankedTensorType>(value.getType());
  if (!valueTy ||
      valueTy.getElementType() != Float32Type::get(value.getContext()))
    return false;
  if (valueTy.hasStaticShape() && valueTy.getNumElements() != 1)
    return false;
  auto dense = dyn_cast_or_null<DenseElementsAttr>(
      getElementAttributeFromONNXValue(value));
  if (!dense)
    return false;
  if (!dense.getElementType().isF32())
    return false;
  if (dense.isSplat())
    return dense.getSplatValue<APFloat>().isZero();
  for (APFloat v : dense.getValues<APFloat>())
    if (!v.isZero())
      return false;
  return true;
}

// f32 rank-4 Resize: nearest or bilinear, no antialias, static C/H/W, dynamic batch allowed.
static bool isSupportedGemminiResizeF32(ONNXResizeOp op) {
  auto f32 = Float32Type::get(op.getContext());
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!xTy || !yTy)
    return false;
  if (xTy.getRank() != 4 || yTy.getRank() != 4)
    return false;
  if (xTy.getElementType() != f32 || yTy.getElementType() != f32)
    return false;
  // C, H, W must be statically known; only batch (dim 0) may be dynamic.
  if (!hasStaticShapeExceptDims(xTy, {0}) ||
      !hasStaticShapeExceptDims(yTy, {0}))
    return false;
  // Mode must be "nearest" or "linear".
  auto modeAttr = op.getModeAttr();
  if (!modeAttr)
    return false;
  StringRef mode = modeAttr.getValue();
  if (mode != "nearest" && mode != "linear")
    return false;
  // No antialias (attribute is si64, use getSExtValue to avoid signless
  // assert).
  if (auto aaAttr = op.getAntialiasAttr())
    if (aaAttr.getValue().getSExtValue() != 0)
      return false;
  // No exclude_outside.
  if (auto eoAttr = op.getExcludeOutsideAttr())
    if (eoAttr.getValue().getSExtValue() != 0)
      return false;
  // No axes restriction: we only support full 4-D resize.
  if (op.getAxesAttr())
    return false;
  // Supported coordinate_transformation_mode values only.
  if (auto ctmAttr = op.getCoordinateTransformationModeAttr()) {
    StringRef ctm = ctmAttr.getValue();
    if (!ctm.empty() && ctm != "asymmetric" && ctm != "half_pixel" &&
        ctm != "align_corners")
      return false;
  }
  return true;
}

// f32 rank-4 Pad: constant (zero-only) / reflect / edge mode, static C/H/W, dynamic batch allowed.
static bool isSupportedGemminiPadF32(ONNXPadOp op) {
  auto f32 = Float32Type::get(op.getContext());
  auto xTy = dyn_cast<RankedTensorType>(op.getData().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!xTy || !yTy)
    return false;
  if (xTy.getRank() != 4 || yTy.getRank() != 4)
    return false;
  if (xTy.getElementType() != f32 || yTy.getElementType() != f32)
    return false;
  if (!hasStaticShapeExceptDims(xTy, {0}) ||
      !hasStaticShapeExceptDims(yTy, {0}))
    return false;

  GemminiPadSpec pads;
  if (!extractGemminiPadSpec(op, pads))
    return false;

  StringRef mode = op.getMode();
  if (mode != "constant" && mode != "reflect" && mode != "edge")
    return false;
  if (mode == "constant" && !isStaticZeroF32Constant(op.getConstantValue()))
    return false;

  ArrayRef<int64_t> xShape = xTy.getShape();
  ArrayRef<int64_t> yShape = yTy.getShape();
  if (!ShapedType::isDynamic(xShape[0]) && !ShapedType::isDynamic(yShape[0]) &&
      xShape[0] != yShape[0])
    return false;
  if (xShape[1] != yShape[1])
    return false;
  if (yShape[2] != xShape[2] + pads.top + pads.bottom ||
      yShape[3] != xShape[3] + pads.left + pads.right)
    return false;
  if (mode == "reflect" &&
      (pads.top >= xShape[2] || pads.bottom >= xShape[2] ||
          pads.left >= xShape[3] || pads.right >= xShape[3]))
    return false;
  return true;
}

// ===--- Float Concat predicate ---===

static bool isSupportedGemminiConcatF32(ONNXConcatOp op) {
  auto f32 = Float32Type::get(op.getContext());
  // Require exactly 2 inputs.
  auto inputs = op.getInputs();
  if (inputs.size() != 2)
    return false;
  // Require f32, rank-4, axis in [0,3].
  int64_t axis = op.getAxis();
  if (axis < 0 || axis > 3)
    return false;
  auto yTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!yTy || yTy.getRank() != 4 || yTy.getElementType() != f32)
    return false;
  for (Value inp : inputs) {
    auto iTy = dyn_cast<RankedTensorType>(inp.getType());
    if (!iTy || iTy.getRank() != 4 || iTy.getElementType() != f32)
      return false;
    // C, H, W must be static; only batch (dim 0) may be dynamic.
    if (!hasStaticShapeExceptDims(iTy, {0}))
      return false;
  }
  // For axis == 0 (batch concat), require static batch on all inputs so the
  // output alloc is straightforward (no runtime sum of dynamic dims needed).
  if (axis == 0) {
    for (Value inp : inputs) {
      auto iTy = cast<RankedTensorType>(inp.getType());
      if (iTy.isDynamicDim(0))
        return false;
    }
  }
  return true;
}

// ===--- Float Transpose predicate ---===

static bool extractTransposePerm(
    ONNXTransposeOp op, SmallVectorImpl<int64_t> &perm) {
  perm.clear();
  ArrayAttr permAttr = op.getPermAttr();
  if (!permAttr || permAttr.size() != 4)
    return false;
  bool seen[4] = {false, false, false, false};
  for (int64_t i = 0; i < 4; ++i) {
    int64_t p = ArrayAttrIntVal(permAttr, i);
    if (p < 0 || p > 3 || seen[p])
      return false;
    seen[p] = true;
    perm.push_back(p);
  }
  return true;
}

// f32 rank-4 Transpose: static C/H/W; dynamic batch allowed only when it stays at axis 0.
static bool isSupportedGemminiTransposeF32(ONNXTransposeOp op) {
  auto f32 = Float32Type::get(op.getContext());
  auto xTy = dyn_cast<RankedTensorType>(op.getData().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getTransposed().getType());
  if (!xTy || !yTy)
    return false;
  if (xTy.getRank() != 4 || yTy.getRank() != 4)
    return false;
  if (xTy.getElementType() != f32 || yTy.getElementType() != f32)
    return false;

  SmallVector<int64_t, 4> perm;
  if (!extractTransposePerm(op, perm))
    return false;

  // C/H/W must be static; batch may be dynamic only when it remains axis 0.
  if (!hasStaticShapeExceptDims(xTy, {0}) ||
      !hasStaticShapeExceptDims(yTy, {0}))
    return false;
  if (xTy.isDynamicDim(0) && perm[0] != 0)
    return false;

  ArrayRef<int64_t> xShape = xTy.getShape();
  ArrayRef<int64_t> yShape = yTy.getShape();
  for (int64_t i = 0; i < 4; ++i) {
    int64_t expected = xShape[perm[i]];
    int64_t actual = yShape[i];
    if (ShapedType::isDynamic(expected) || ShapedType::isDynamic(actual)) {
      if (!(perm[i] == 0 && i == 0 && ShapedType::isDynamic(expected) &&
              ShapedType::isDynamic(actual)))
        return false;
      continue;
    }
    if (actual != expected)
      return false;
  }
  return true;
}

// ===--- Float Split predicate ---===

static bool isSupportedGemminiSplitF32(ONNXSplitOp op) {
  auto f32 = Float32Type::get(op.getContext());
  auto xTy = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!xTy || xTy.getRank() != 4 || xTy.getElementType() != f32)
    return false;
  // C, H, W must be static; batch may be dynamic only when axis != 0.
  if (!hasStaticShapeExceptDims(xTy, {0}))
    return false;

  int64_t axis = op.getAxis();
  if (axis < 0)
    axis += 4;
  if (axis < 0 || axis > 3)
    return false;

  // When splitting on the batch axis (0), require static batch so the output
  // allocation is straightforward (no dynamic dim involved).
  if (axis == 0 && xTy.isDynamicDim(0))
    return false;

  // Support 2, 3, or 4 outputs.
  unsigned numOutputs = op.getNumResults();
  if (numOutputs < 2 || numOutputs > 4)
    return false;

  // All output axis dimensions must be static (ONNX shape inference required).
  for (Value res : op.getOutputs()) {
    auto outTy = dyn_cast<RankedTensorType>(res.getType());
    if (!outTy || outTy.getRank() != 4 || outTy.getElementType() != f32)
      return false;
    if (ShapedType::isDynamic(outTy.getDimSize(axis)))
      return false;
    // Non-axis, non-batch dims must be static.
    for (int64_t d = 0; d < 4; ++d) {
      if (d == 0 || d == axis)
        continue;
      if (ShapedType::isDynamic(outTy.getDimSize(d)))
        return false;
    }
  }
  return true;
}

// ===--- Slice predicate ---===

// Extract int64 values from an ONNXConstantOp tensor input.
// Returns true on success, false if the value is absent (NoneType) or not a
// compile-time constant; in the absent case *out is left empty.
static bool extractSliceConstants(Value v, SmallVector<int64_t> &out) {
  out.clear();
  if (!v || isa<NoneType>(v.getType()))
    return true; // absent (optional) – OK
  SmallVector<int64_t> vals;
  if (!getI64ValuesFromONNXConstantOp(v, vals))
    return false;
  out = std::move(vals);
  return true;
}

// f32 rank-4 Slice: compile-time starts/ends required, forward-only step, dynamic batch not sliceable.
static bool isSupportedGemminiSliceF32(ONNXSliceOp op) {
  auto f32 = Float32Type::get(op.getContext());
  auto dataTy = dyn_cast<RankedTensorType>(op.getData().getType());
  if (!dataTy || dataTy.getRank() != 4 || dataTy.getElementType() != f32)
    return false;
  // C, H, W must be static; batch may be dynamic.
  if (!hasStaticShapeExceptDims(dataTy, {0}))
    return false;

  // starts and ends are required constant tensors.
  SmallVector<int64_t> rawStarts, rawEnds, rawAxes, rawSteps;
  if (!extractSliceConstants(op.getStarts(), rawStarts) || rawStarts.empty())
    return false;
  if (!extractSliceConstants(op.getEnds(), rawEnds) || rawEnds.empty())
    return false;
  if (!extractSliceConstants(op.getAxes(), rawAxes))
    return false;
  if (!extractSliceConstants(op.getSteps(), rawSteps))
    return false;

  // starts and ends must have the same length.
  if (rawStarts.size() != rawEnds.size())
    return false;
  // axes (if present) must match starts count.
  if (!rawAxes.empty() && rawAxes.size() != rawStarts.size())
    return false;
  // steps (if present) must match starts count.
  if (!rawSteps.empty() && rawSteps.size() != rawStarts.size())
    return false;

  int64_t numSliced = static_cast<int64_t>(rawStarts.size());

  // Populate rawAxes defaults ([0..numSliced-1]) if absent.
  if (rawAxes.empty()) {
    for (int64_t i = 0; i < numSliced; ++i)
      rawAxes.push_back(i);
  }

  // All specified steps must be ≥ 1 (forward-only).
  for (int64_t step : rawSteps)
    if (step < 1)
      return false;

  // Normalize axes to [0,3]; reject axis 0 when batch is dynamic (we cannot
  // compute the output batch size without knowing the actual input batch dim).
  for (int64_t i = 0; i < numSliced; ++i) {
    int64_t a = rawAxes[i];
    if (a < 0)
      a += 4;
    if (a < 0 || a > 3)
      return false;
    if (a == 0 && dataTy.isDynamicDim(0))
      return false; // dynamic batch cannot be sliced
  }
  return true;
}

static Value createConvTransposeOutputAlloc(ConversionPatternRewriter &rewriter,
    Location loc, MemRefType resultMemRef, Value inputMemref, int64_t kH,
    int64_t kW, int64_t stride, int64_t pad, int64_t output_pad) {
  SmallVector<Value, 4> dynamicDims;
  for (int64_t i = 0, e = resultMemRef.getRank(); i < e; ++i) {
    if (!ShapedType::isDynamic(resultMemRef.getDimSize(i)))
      continue;
    Value dimIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
    if (i == 0) {
      dynamicDims.push_back(
          memref::DimOp::create(rewriter, loc, inputMemref, dimIdx));
    } else if (i == 2) {
      Value inH = memref::DimOp::create(rewriter, loc, inputMemref, dimIdx);
      dynamicDims.push_back(computeConvTransposeOutSpatial(
          rewriter, loc, inH, kH, pad, stride, output_pad));
    } else if (i == 3) {
      Value inW = memref::DimOp::create(rewriter, loc, inputMemref, dimIdx);
      dynamicDims.push_back(computeConvTransposeOutSpatial(
          rewriter, loc, inW, kW, pad, stride, output_pad));
    } else {
      return Value(); // dim 1 (channels) must always be static
    }
  }
  return memref::AllocOp::create(rewriter, loc, resultMemRef, dynamicDims);
}

// Static f32 tensor of any rank (no dynamic dims allowed).
static bool isSupportedGemminiReluF32(ONNXReluOp op) {
  return isF32StaticTensor(op.getX());
}

// f32 elementwise Add: both operands must be static tensors with identical shape (no broadcasting).
static bool isSupportedGemminiAddF32(ONNXAddOp op) {
  if (!isF32StaticTensor(op.getA()) || !isF32StaticTensor(op.getB()))
    return false;
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto bTy = cast<RankedTensorType>(op.getB().getType());
  return aTy.getShape() == bTy.getShape();
}

// f32 rank-4 Sigmoid: static C/H/W required; batch dimension may be dynamic.
static bool isSupportedGemminiSigmoidF32(ONNXSigmoidOp op) {
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  if (!xTy || xTy.getRank() != 4 || !xTy.getElementType().isF32())
    return false;
  // Spatial dims (C, H, W) must be static; batch dimension may be dynamic.
  return hasStaticShapeExceptDims(xTy, {0});
}

// f32 rank-4 elementwise Mul: same shape required (no broadcasting); static C/H/W, dynamic batch ok.
static bool isSupportedGemminiMulF32(ONNXMulOp op) {
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
  if (!aTy || !bTy || aTy.getRank() != 4 || bTy.getRank() != 4)
    return false;
  if (!aTy.getElementType().isF32() || !bTy.getElementType().isF32())
    return false;
  if (!hasStaticShapeExceptDims(aTy, {0}) || !hasStaticShapeExceptDims(bTy, {0}))
    return false;
  // Same shape required — elementwise multiply, no broadcasting.
  return aTy.getShape() == bTy.getShape();
}

// BatchNormalization inference mode: already converted by ONNX preprocessing.
static bool isSupportedGemminiBatchNormF32(
    ONNXBatchNormalizationInferenceModeOp op) {
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  if (!xTy || !xTy.hasStaticShape() || xTy.getRank() != 4)
    return false;
  return xTy.getElementType().isF32();
}

// Fully-static f32 rank-4 GlobalAveragePool (reduces H×W to 1×1 per channel).
static bool isSupportedGemminiGlobalAvgPoolF32(ONNXGlobalAveragePoolOp op) {
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  if (!xTy || !xTy.hasStaticShape() || xTy.getRank() != 4)
    return false;
  return xTy.getElementType().isF32();
}

// f32 rank-4 MaxPool: fully static, square kernel, no ceil_mode, no dilation.
static bool isSupportedGemminiMaxPoolF32(ONNXMaxPoolSingleOutOp op) {
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  if (!xTy || !xTy.hasStaticShape() || xTy.getRank() != 4)
    return false;
  if (!xTy.getElementType().isF32())
    return false;
  auto kernelShape = op.getKernelShape();
  if (!kernelShape || kernelShape.size() != 2)
    return false;
  int64_t kh = cast<IntegerAttr>(kernelShape[0]).getInt();
  int64_t kw = cast<IntegerAttr>(kernelShape[1]).getInt();
  if (kh != kw)
    return false;
  if (op.getCeilMode() != 0)
    return false;
  if (auto dilations = op.getDilationsAttr()) {
    for (auto d : dilations)
      if (cast<IntegerAttr>(d).getInt() != 1)
        return false;
  }
  return true;
}

// f32 rank-4 AveragePool: fully static, square kernel, no ceil_mode.
static bool isSupportedGemminiAvgPoolF32(ONNXAveragePoolOp op) {
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  if (!xTy || !xTy.hasStaticShape() || xTy.getRank() != 4)
    return false;
  if (!xTy.getElementType().isF32())
    return false;
  auto kernelShape = op.getKernelShape();
  if (!kernelShape || kernelShape.size() != 2)
    return false;
  int64_t kh = cast<IntegerAttr>(kernelShape[0]).getInt();
  int64_t kw = cast<IntegerAttr>(kernelShape[1]).getInt();
  if (kh != kw)
    return false;
  if (op.getCeilMode() != 0)
    return false;
  return true;
}

// f32 static Softmax: axis must be -1 or the last dimension (row-wise softmax only).
static bool isSupportedGemminiSoftmaxF32(ONNXSoftmaxOp op) {
  auto inputTy = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!inputTy || !inputTy.hasStaticShape() ||
      !inputTy.getElementType().isF32())
    return false;
  int64_t rank = inputTy.getRank();
  int64_t axis = op.getAxis();
  // Normalize negative axis: ONNX axis=-1 means last dimension.
  return axis == -1 || axis == rank - 1;
}

// f32 rank-2 Gemm: transA/transB supported; optional bias (scalar/1-D/2-D); dynamic batch row ok.
static bool isSupportedGemminiGemmF32(ONNXGemmOp op) {
  auto f32 = Float32Type::get(op.getContext());
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!aTy || !bTy || !yTy)
    return false;
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || yTy.getRank() != 2)
    return false;
  if (aTy.getElementType() != f32 || bTy.getElementType() != f32 ||
      yTy.getElementType() != f32)
    return false;
  if (!hasStaticShapeExceptDims(aTy, {0}) || !bTy.hasStaticShape() ||
      !hasStaticShapeExceptDims(yTy, {0}))
    return false;
  int64_t transA = op.getTransA();
  int64_t transB = op.getTransB();
  int64_t aM = transA ? aTy.getShape()[1] : aTy.getShape()[0];
  int64_t aK = transA ? aTy.getShape()[0] : aTy.getShape()[1];
  int64_t bK = transB ? bTy.getShape()[1] : bTy.getShape()[0];
  int64_t bN = transB ? bTy.getShape()[0] : bTy.getShape()[1];
  if (aK != bK)
    return false;
  if (!ShapedType::isDynamic(aM) && !yTy.isDynamicDim(0) &&
      aM != yTy.getShape()[0])
    return false;
  if (yTy.getShape()[1] != bN)
    return false;
  if (!isNoneValue(op.getC())) {
    auto cTy = dyn_cast<RankedTensorType>(op.getC().getType());
    if (!cTy || !cTy.hasStaticShape() || cTy.getElementType() != f32)
      return false;
    if (cTy.getRank() == 0)
      return true;
    if (cTy.getRank() == 1)
      return cTy.getShape()[0] == bN || cTy.getShape()[0] == 1;
    if (cTy.getRank() == 2)
      return (cTy.getShape()[1] == bN || cTy.getShape()[1] == 1);
    return false;
  }
  return true;
}

// f32 rank-2 MatMul: both operands fully static (no batched / dynamic shapes).
static bool isSupportedGemminiMatMulF32(ONNXMatMulOp op) {
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
  if (!aTy || !bTy || !aTy.hasStaticShape() || !bTy.hasStaticShape())
    return false;
  if (aTy.getRank() != 2 || bTy.getRank() != 2)
    return false;
  return aTy.getElementType().isF32() && bTy.getElementType().isF32();
}

// f32 batched MatMul: rank-N (≥3) left × rank-2 right; inner K must be static; covers BERT attention.
static bool isSupportedGemminiMatMulF32Batched(ONNXMatMulOp op) {
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
  auto yTy = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!aTy || !bTy || !yTy)
    return false;
  if (aTy.getRank() < 3 || bTy.getRank() != 2 || yTy.getRank() != aTy.getRank())
    return false;
  if (!aTy.getElementType().isF32() || !bTy.getElementType().isF32() ||
      !yTy.getElementType().isF32())
    return false;
  if (!bTy.hasStaticShape())
    return false;
  int64_t k = aTy.getShape().back();
  if (ShapedType::isDynamic(k) || k != bTy.getShape()[0])
    return false;
  if (yTy.getShape().back() != bTy.getShape()[1])
    return false;
  for (int64_t i = 0; i < aTy.getRank() - 1; ++i) {
    if (!ShapedType::isDynamic(aTy.getShape()[i]) &&
        !ShapedType::isDynamic(yTy.getShape()[i]) &&
        aTy.getShape()[i] != yTy.getShape()[i])
      return false;
  }
  return true;
}

// f16 rank-2 MatMul: both operands fully static; note FP16 runtime lowering is incomplete (see known limitations).
static bool isSupportedGemminiMatMulF16(ONNXMatMulOp op) {
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
  if (!aTy || !bTy || !aTy.hasStaticShape() || !bTy.hasStaticShape())
    return false;
  if (aTy.getRank() != 2 || bTy.getRank() != 2)
    return false;
  return aTy.getElementType().isF16() && bTy.getElementType().isF16();
}

// ===--- Integer / quantized lowering patterns ---===

struct ONNXMatMulToGemminiLowering : public OpConversionPattern<ONNXMatMulOp> {
  ONNXMatMulToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXMatMulOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXMatMulOp op, ONNXMatMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiMatMul(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Value output = memref::AllocOp::create(rewriter, op.getLoc(), resultMemRef);
    auto ws = rewriter.getStringAttr("ws");
    auto identity = rewriter.getStringAttr("identity");
    gemmini::GemminiMatmulOp gemm = gemmini::GemminiMatmulOp::create(rewriter,
        op.getLoc(), adaptor.getA(), adaptor.getB(), output, identity, ws);
    gemm->setAttr("gemmini.high_level", rewriter.getUnitAttr());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXMatMulIntegerToGemminiLowering
    : public OpConversionPattern<ONNXMatMulIntegerOp> {
  ONNXMatMulIntegerToGemminiLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXMatMulIntegerOp>(
            typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXMatMulIntegerOp op,
      ONNXMatMulIntegerOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiMatMulInteger(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Value output = memref::AllocOp::create(rewriter, op.getLoc(), resultMemRef);
    SmallVector<Value, 8> operands{output, adaptor.getA(), adaptor.getB()};
    StringRef funcName = "om_gemmini_matmulinteger_i8i8acc32";
    if (!isNoneValue(op.getAZeroPoint()) || !isNoneValue(op.getBZeroPoint())) {
      operands.push_back(adaptor.getAZeroPoint());
      operands.push_back(adaptor.getBZeroPoint());
      funcName = "om_gemmini_matmulinteger_i8i8acc32_zp";
    }
    KrnlCallOp::create(rewriter, op.getLoc(), funcName.str(),
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// Direct path for 16×16 INT8 MatMulInteger: emits GemminiMatmulOp so the
// tiling pass lowers it to mvin/matmul/mvout rather than a runtime krnl.call.
struct ONNXMatMulIntegerDirectToGemminiLowering
    : public OpConversionPattern<ONNXMatMulIntegerOp> {
  ONNXMatMulIntegerDirectToGemminiLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXMatMulIntegerOp>(
            typeConverter, ctx, /*benefit=*/11) {}

  LogicalResult matchAndRewrite(ONNXMatMulIntegerOp op,
      ONNXMatMulIntegerOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiDirectMatMulInteger(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Value output = memref::AllocOp::create(rewriter, op.getLoc(), resultMemRef);
    auto ws = rewriter.getStringAttr("ws");
    auto identity = rewriter.getStringAttr("identity");
    gemmini::GemminiMatmulOp gemm = gemmini::GemminiMatmulOp::create(rewriter,
        op.getLoc(), adaptor.getA(), adaptor.getB(), output, identity, ws);
    gemm->setAttr("gemmini.high_level", rewriter.getUnitAttr());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXQLinearConvToGemminiLowering
    : public OpConversionPattern<ONNXQLinearConvOp> {
  ONNXQLinearConvToGemminiLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXQLinearConvOp>(
            typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXQLinearConvOp op,
      ONNXQLinearConvOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiQLinearConv(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Value output = memref::AllocOp::create(rewriter, op.getLoc(), resultMemRef);
    int64_t stride = 1;
    if (auto strides = op.getStrides())
      stride = cast<IntegerAttr>((*strides)[0]).getInt();
    int64_t padding = 0;
    if (auto pads = op.getPads())
      padding = cast<IntegerAttr>((*pads)[0]).getInt();
    SmallVector<Value, 16> operands{output, adaptor.getX(), adaptor.getXScale(),
        adaptor.getXZeroPoint(), adaptor.getW(), adaptor.getWScale(),
        adaptor.getWZeroPoint(), adaptor.getYScale(), adaptor.getYZeroPoint()};
    StringRef funcName = "om_gemmini_qlinearconv_i8";
    if (!isNoneValue(op.getB())) {
      operands.push_back(adaptor.getB());
      funcName = "om_gemmini_qlinearconv_i8_bias";
    }
    auto i64Type = rewriter.getI64Type();
    operands.push_back(arith::ConstantOp::create(
        rewriter, op.getLoc(), i64Type, rewriter.getI64IntegerAttr(stride)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, op.getLoc(), i64Type, rewriter.getI64IntegerAttr(padding)));
    KrnlCallOp::create(rewriter, op.getLoc(), funcName.str(),
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// ===--- Float lowering patterns ---===

struct ONNXConvToGemminiLowering : public OpConversionPattern<ONNXConvOp> {
  ONNXConvToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXConvOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXConvOp op, ONNXConvOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiConvF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();
    int64_t stride = 1;
    if (auto strides = op.getStrides())
      stride = cast<IntegerAttr>((*strides)[0]).getInt();
    int64_t pad = 0;
    if (auto pads = op.getPads())
      pad = cast<IntegerAttr>((*pads)[0]).getInt();
    // Weights are always fully static (required by the predicate).
    auto wMemRef = cast<MemRefType>(adaptor.getW().getType());
    int64_t kH = wMemRef.getShape()[2];
    int64_t kW = wMemRef.getShape()[3];
    Value output = createConvOutputAlloc(
        rewriter, loc, resultMemRef, adaptor.getX(), kH, kW, stride, pad);
    if (!output)
      return failure();
    SmallVector<Value, 8> operands{output, adaptor.getX(), adaptor.getW()};
    StringRef funcName = "om_gemmini_conv_f32";
    if (!isNoneValue(op.getB())) {
      operands.push_back(adaptor.getB());
      funcName = "om_gemmini_conv_f32_bias";
    }
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(stride)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(pad)));
    KrnlCallOp::create(
        rewriter, loc, funcName.str(), /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXConvTransposeToGemminiLowering
    : public OpConversionPattern<ONNXConvTransposeOp> {
  ONNXConvTransposeToGemminiLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXConvTransposeOp>(
            typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXConvTransposeOp op,
      ONNXConvTransposeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiConvTransposeF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    int64_t stride = 1;
    if (auto strides = op.getStrides())
      stride = cast<IntegerAttr>((*strides)[0]).getInt();
    int64_t pad = 0;
    if (auto pads = op.getPads())
      pad = cast<IntegerAttr>((*pads)[0]).getInt();
    int64_t output_pad = 0;
    if (auto opads = op.getOutputPadding())
      output_pad = cast<IntegerAttr>((*opads)[0]).getInt();

    // Weights are always fully static (required by predicate).
    auto wMemRef = cast<MemRefType>(adaptor.getW().getType());
    int64_t kH = wMemRef.getShape()[2];
    int64_t kW = wMemRef.getShape()[3];

    Value output = createConvTransposeOutputAlloc(rewriter, loc, resultMemRef,
        adaptor.getX(), kH, kW, stride, pad, output_pad);
    if (!output)
      return failure();

    SmallVector<Value, 10> operands{output, adaptor.getX(), adaptor.getW()};
    StringRef funcName = "om_gemmini_convtranspose_f32";
    if (!isNoneValue(op.getB())) {
      operands.push_back(adaptor.getB());
      funcName = "om_gemmini_convtranspose_f32_bias";
    }
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(stride)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(pad)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(output_pad)));
    KrnlCallOp::create(
        rewriter, loc, funcName.str(), /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXResizeToGemminiLowering : public OpConversionPattern<ONNXResizeOp> {
  ONNXResizeToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXResizeOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXResizeOp op, ONNXResizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiResizeF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    // Alloc output; dynamic dims (batch only) sourced from the input memref.
    Value output =
        createAllocLikeResult(rewriter, loc, resultMemRef, adaptor.getX());
    if (!output)
      return failure();

    int64_t coordMode = resizeCoordModeCode(op);
    Value coordModeVal = arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(coordMode));

    SmallVector<Value, 6> operands{output, adaptor.getX(), coordModeVal};
    StringRef mode = op.getModeAttr().getValue();
    if (mode == "nearest") {
      int64_t nearestMode = resizeNearestModeCode(op);
      operands.push_back(arith::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(nearestMode)));
      KrnlCallOp::create(rewriter, loc, "om_gemmini_resize_nearest_f32",
          /*numOfOutput=*/1, operands);
    } else {
      KrnlCallOp::create(rewriter, loc, "om_gemmini_resize_linear_f32",
          /*numOfOutput=*/1, operands);
    }
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXPadToGemminiLowering : public OpConversionPattern<ONNXPadOp> {
  ONNXPadToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXPadOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXPadOp op, ONNXPadOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiPadF32(op))
      return failure();
    auto resultMemRef = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!resultMemRef)
      return failure();

    GemminiPadSpec pads;
    if (!extractGemminiPadSpec(op, pads))
      return failure();

    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();
    Value output =
        createAllocLikeResult(rewriter, loc, resultMemRef, adaptor.getData());
    if (!output)
      return failure();

    SmallVector<Value, 8> operands{output, adaptor.getData()};
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(pads.left)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(pads.right)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(pads.top)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(pads.bottom)));

    StringRef funcName = "om_gemmini_pad_constant_f32";
    StringRef mode = op.getMode();
    if (mode == "reflect")
      funcName = "om_gemmini_pad_reflect_f32";
    else if (mode == "edge")
      funcName = "om_gemmini_pad_edge_f32";

    KrnlCallOp::create(
        rewriter, loc, funcName.str(), /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXConcatToGemminiLowering : public OpConversionPattern<ONNXConcatOp> {
  ONNXConcatToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXConcatOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXConcatOp op, ONNXConcatOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiConcatF32(op))
      return failure();
    auto resultMemRef = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!resultMemRef)
      return failure();

    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    // For axis != 0 the batch dim may be dynamic; source it from input 0.
    // For axis == 0 the predicate already required static batch on all inputs,
    // so the output memref is fully static and no dynamic dims are needed.
    ValueRange adaptedInputs = adaptor.getInputs();
    Value output;
    if (op.getAxis() == 0)
      output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    else
      output =
          createAllocLikeResult(rewriter, loc, resultMemRef, adaptedInputs[0]);
    if (!output)
      return failure();

    Value axisVal = arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getAxis()));
    SmallVector<Value, 4> operands{
        output, adaptedInputs[0], adaptedInputs[1], axisVal};
    KrnlCallOp::create(
        rewriter, loc, "om_gemmini_concat_f32", /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXTransposeToGemminiLowering
    : public OpConversionPattern<ONNXTransposeOp> {
  ONNXTransposeToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXTransposeOp>(
            typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXTransposeOp op,
      ONNXTransposeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiTransposeF32(op))
      return failure();
    auto resultMemRef = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getTransposed().getType()));
    if (!resultMemRef)
      return failure();

    SmallVector<int64_t, 4> perm;
    if (!extractTransposePerm(op, perm))
      return failure();

    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    Value output;
    if (resultMemRef.isDynamicDim(0))
      output =
          createAllocLikeResult(rewriter, loc, resultMemRef, adaptor.getData());
    else
      output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    if (!output)
      return failure();

    SmallVector<Value, 6> operands{output, adaptor.getData()};
    for (int64_t p : perm) {
      operands.push_back(arith::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(p)));
    }
    KrnlCallOp::create(rewriter, loc, "om_gemmini_transpose_f32_hw",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXSliceToGemminiLowering : public OpConversionPattern<ONNXSliceOp> {
  ONNXSliceToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXSliceOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXSliceOp op, ONNXSliceOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiSliceF32(op))
      return failure();

    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    // Extract constant starts / ends / axes / steps from original (tensor) ops.
    SmallVector<int64_t> rawStarts, rawEnds, rawAxes, rawSteps;
    if (!extractSliceConstants(op.getStarts(), rawStarts) ||
        !extractSliceConstants(op.getEnds(), rawEnds) ||
        !extractSliceConstants(op.getAxes(), rawAxes) ||
        !extractSliceConstants(op.getSteps(), rawSteps))
      return failure();

    int64_t numSliced = static_cast<int64_t>(rawStarts.size());

    // Defaults: axes = [0..numSliced-1], steps = [1..1].
    if (rawAxes.empty())
      for (int64_t i = 0; i < numSliced; ++i)
        rawAxes.push_back(i);
    if (rawSteps.empty())
      for (int64_t i = 0; i < numSliced; ++i)
        rawSteps.push_back(1);

    // Per-dim (NCHW) start / end / step.  Unsliced dims use identity
    // (start=0, end=INT64_MAX → runtime clamps to actual size, step=1).
    constexpr int64_t kEnd = 0x7FFFFFFFFFFFFFFFLL;
    int64_t ns[4] = {0, 0, 0, 0};
    int64_t ne[4] = {kEnd, kEnd, kEnd, kEnd};
    int64_t nt[4] = {1, 1, 1, 1};
    for (int64_t i = 0; i < numSliced; ++i) {
      int64_t a = rawAxes[i];
      if (a < 0)
        a += 4;
      ns[a] = rawStarts[i];
      ne[a] = rawEnds[i];
      nt[a] = rawSteps[i];
    }

    // Compute output shape for result memref allocation.
    auto dataTy = cast<RankedTensorType>(op.getData().getType());
    ArrayRef<int64_t> inShape = dataTy.getShape();
    SmallVector<int64_t, 4> outShape(4);
    for (int64_t d = 0; d < 4; ++d) {
      bool isSliced = false;
      for (int64_t i = 0; i < numSliced; ++i) {
        int64_t a = rawAxes[i];
        if (a < 0)
          a += 4;
        if (a == d) {
          isSliced = true;
          break;
        }
      }
      if (!isSliced) {
        outShape[d] =
            inShape[d]; // identity; may be ShapedType::kDynamic for d==0
      } else {
        // Predicate guarantees this dim is static when sliced.
        int64_t dimSz = inShape[d];
        int64_t s = ns[d], e = ne[d], t = nt[d];
        if (s < 0)
          s += dimSz;
        if (s < 0)
          s = 0;
        if (e < 0)
          e += dimSz;
        if (e > dimSz)
          e = dimSz;
        if (s >= e)
          return failure();
        outShape[d] = (e - s + t - 1) / t;
      }
    }

    auto resultTy = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getOutput().getType()));
    if (!resultTy)
      return failure();

    // Allocate: dynamic batch (not sliced) → source dim from input memref.
    Value output;
    if (ShapedType::isDynamic(outShape[0]))
      output =
          createAllocLikeResult(rewriter, loc, resultTy, adaptor.getData());
    else
      output = memref::AllocOp::create(rewriter, loc, resultTy);
    if (!output)
      return failure();

    auto cI64 = [&](int64_t v) -> Value {
      return arith::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(v));
    };
    SmallVector<Value, 14> operands{output, adaptor.getData(), cI64(ns[0]),
        cI64(ne[0]), cI64(nt[0]), cI64(ns[1]), cI64(ne[1]), cI64(nt[1]),
        cI64(ns[2]), cI64(ne[2]), cI64(nt[2]), cI64(ns[3]), cI64(ne[3]),
        cI64(nt[3])};
    KrnlCallOp::create(
        rewriter, loc, "om_gemmini_slice_f32", /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXSplitToGemminiLowering : public OpConversionPattern<ONNXSplitOp> {
  ONNXSplitToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXSplitOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXSplitOp op, ONNXSplitOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiSplitF32(op))
      return failure();

    unsigned numOutputs = op.getNumResults();
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += 4;

    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    // Allocate each output.  For axis != 0 the batch dim may be dynamic;
    // source it from the converted input memref (dim 0 copies from input).
    // For axis == 0 the predicate already required static batch.
    SmallVector<Value, 4> allocs;
    for (unsigned i = 0; i < numOutputs; ++i) {
      auto outMemRef = dyn_cast<MemRefType>(
          typeConverter->convertType(op.getOutputs()[i].getType()));
      if (!outMemRef)
        return failure();
      Value alloc;
      if (axis != 0 && outMemRef.isDynamicDim(0))
        alloc = createAllocLikeResult(
            rewriter, loc, outMemRef, adaptor.getInput());
      else
        alloc = memref::AllocOp::create(rewriter, loc, outMemRef);
      if (!alloc)
        return failure();
      allocs.push_back(alloc);
    }

    // Select runtime function name based on output count.
    static const char *kFuncNames[] = {nullptr, nullptr,
        "om_gemmini_split_2_f32", "om_gemmini_split_3_f32",
        "om_gemmini_split_4_f32"};
    StringRef funcName = kFuncNames[numOutputs];

    // Operands: [out0, out1, ..., input, axis].
    Value axisVal = arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(axis));
    SmallVector<Value> operands(allocs.begin(), allocs.end());
    operands.push_back(adaptor.getInput());
    operands.push_back(axisVal);

    KrnlCallOp::create(rewriter, loc, funcName.str(),
        /*numOfOutput=*/static_cast<int64_t>(numOutputs), operands);
    rewriter.replaceOp(op, allocs);
    return success();
  }
};

struct ONNXReluToGemminiLowering : public OpConversionPattern<ONNXReluOp> {
  ONNXReluToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXReluOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXReluOp op, ONNXReluOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiReluF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    SmallVector<Value, 4> operands{output, adaptor.getX()};
    KrnlCallOp::create(rewriter, loc, "om_gemmini_relu_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXAddToGemminiLowering : public OpConversionPattern<ONNXAddOp> {
  ONNXAddToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXAddOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXAddOp op, ONNXAddOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiAddF32(op))
      return failure();
    auto resultMemRef = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    SmallVector<Value, 4> operands{output, adaptor.getA(), adaptor.getB()};
    KrnlCallOp::create(rewriter, loc, "om_gemmini_add_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXSigmoidToGemminiLowering : public OpConversionPattern<ONNXSigmoidOp> {
  ONNXSigmoidToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXSigmoidOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXSigmoidOp op, ONNXSigmoidOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiSigmoidF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output =
        createAllocLikeResult(rewriter, loc, resultMemRef, adaptor.getX());
    if (!output)
      return failure();
    SmallVector<Value, 2> operands{output, adaptor.getX()};
    KrnlCallOp::create(rewriter, loc, "om_gemmini_sigmoid_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXMulToGemminiLowering : public OpConversionPattern<ONNXMulOp> {
  ONNXMulToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXMulOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXMulOp op, ONNXMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiMulF32(op))
      return failure();
    auto resultMemRef = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getC().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output =
        createAllocLikeResult(rewriter, loc, resultMemRef, adaptor.getA());
    if (!output)
      return failure();
    SmallVector<Value, 4> operands{output, adaptor.getA(), adaptor.getB()};
    KrnlCallOp::create(rewriter, loc, "om_gemmini_mul_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXBatchNormToGemminiLowering
    : public OpConversionPattern<ONNXBatchNormalizationInferenceModeOp> {
  ONNXBatchNormToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXBatchNormalizationInferenceModeOp>(
            typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXBatchNormalizationInferenceModeOp op,
      ONNXBatchNormalizationInferenceModeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiBatchNormF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getO_Y().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    float epsilonF = (float)op.getEpsilon().convertToDouble();
    SmallVector<Value, 8> operands{output, adaptor.getX(), adaptor.getScale(),
        adaptor.getB(), adaptor.getMean(), adaptor.getVar()};
    operands.push_back(f32AsI64Bits(rewriter, loc, epsilonF));
    KrnlCallOp::create(rewriter, loc, "om_gemmini_batchnorm_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXGlobalAvgPoolToGemminiLowering
    : public OpConversionPattern<ONNXGlobalAveragePoolOp> {
  ONNXGlobalAvgPoolToGemminiLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXGlobalAveragePoolOp>(
            typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXGlobalAveragePoolOp op,
      ONNXGlobalAveragePoolOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiGlobalAvgPoolF32(op))
      return failure();
    auto resultMemRef = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    SmallVector<Value, 4> operands{output, adaptor.getX()};
    KrnlCallOp::create(rewriter, loc, "om_gemmini_globalavgpool_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXMaxPoolToGemminiLowering
    : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
  ONNXMaxPoolToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXMaxPoolSingleOutOp>(
            typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op,
      ONNXMaxPoolSingleOutOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiMaxPoolF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getO_Y().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    auto i64Type = rewriter.getI64Type();
    auto kernelShape = op.getKernelShape();
    int64_t kernel = cast<IntegerAttr>(kernelShape[0]).getInt();
    int64_t stride = kernel; // default non-overlapping
    if (auto strides = op.getStrides()) {
      if (!strides->empty())
        stride = cast<IntegerAttr>((*strides)[0]).getInt();
    }
    int64_t pad = 0;
    if (auto pads = op.getPads()) {
      if (!pads->empty())
        pad = cast<IntegerAttr>((*pads)[0]).getInt();
    }
    SmallVector<Value, 6> operands{output, adaptor.getX()};
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(kernel)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(stride)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(pad)));
    KrnlCallOp::create(rewriter, loc, "om_gemmini_maxpool_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXAvgPoolToGemminiLowering
    : public OpConversionPattern<ONNXAveragePoolOp> {
  ONNXAvgPoolToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXAveragePoolOp>(
            typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXAveragePoolOp op,
      ONNXAveragePoolOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiAvgPoolF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    auto i64Type = rewriter.getI64Type();
    auto kernelShape = op.getKernelShape();
    int64_t kernel = cast<IntegerAttr>(kernelShape[0]).getInt();
    int64_t stride = kernel;
    if (auto strides = op.getStrides()) {
      if (!strides->empty())
        stride = cast<IntegerAttr>((*strides)[0]).getInt();
    }
    int64_t pad = 0;
    if (auto pads = op.getPads()) {
      if (!pads->empty())
        pad = cast<IntegerAttr>((*pads)[0]).getInt();
    }
    int64_t countIncludePad = op.getCountIncludePad();
    SmallVector<Value, 8> operands{output, adaptor.getX()};
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(kernel)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(stride)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(pad)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(countIncludePad)));
    KrnlCallOp::create(rewriter, loc, "om_gemmini_avgpool_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXSoftmaxToGemminiLowering
    : public OpConversionPattern<ONNXSoftmaxOp> {
  ONNXSoftmaxToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXSoftmaxOp>(typeConverter, ctx, /*benefit=*/10) {
  }

  LogicalResult matchAndRewrite(ONNXSoftmaxOp op, ONNXSoftmaxOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiSoftmaxF32(op))
      return failure();
    auto inputTy = cast<RankedTensorType>(op.getInput().getType());
    auto resultMemRef = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    // Flatten to [batch, classes] where classes is the last dimension.
    int64_t batch = 1;
    for (int64_t i = 0; i < inputTy.getRank() - 1; ++i)
      batch *= inputTy.getShape()[i];
    int64_t classes = inputTy.getShape().back();
    auto i64Type = rewriter.getI64Type();
    SmallVector<Value, 6> operands{output, adaptor.getInput()};
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(batch)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(classes)));
    KrnlCallOp::create(rewriter, loc, "om_gemmini_softmax_f32",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct ONNXGemmToGemminiLowering : public OpConversionPattern<ONNXGemmOp> {
  ONNXGemmToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXGemmOp>(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXGemmOp op, ONNXGemmOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiGemmF32(op))
      return failure();
    auto resultMemRef = dyn_cast<MemRefType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output =
        createAllocLikeResult(rewriter, loc, resultMemRef, adaptor.getA());
    if (!output)
      return failure();
    auto i64Type = rewriter.getI64Type();
    float alphaF = op.getAlpha().convertToFloat();
    float betaF = op.getBeta().convertToFloat();
    int64_t transA = op.getTransA();
    int64_t transB = op.getTransB();
    SmallVector<Value, 10> operands{output, adaptor.getA(), adaptor.getB()};
    StringRef funcName = "om_gemmini_gemm_f32";
    if (!isNoneValue(op.getC())) {
      operands.push_back(adaptor.getC());
      funcName = "om_gemmini_gemm_f32_bias";
    }
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(transA)));
    operands.push_back(arith::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(transB)));
    operands.push_back(f32AsI64Bits(rewriter, loc, alphaF));
    operands.push_back(f32AsI64Bits(rewriter, loc, betaF));
    KrnlCallOp::create(
        rewriter, loc, funcName.str(), /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// Float MatMul dispatches to a Gemmini-backed runtime function that quantizes
// fp32 inputs to the i8 Gemmini data path and dequantizes the accumulator.
struct ONNXMatMulF32ToGemminiLowering
    : public OpConversionPattern<ONNXMatMulOp> {
  ONNXMatMulF32ToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXMatMulOp>(typeConverter, ctx, /*benefit=*/5) {}

  LogicalResult matchAndRewrite(ONNXMatMulOp op, ONNXMatMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiMatMulF32(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    SmallVector<Value, 4> operands{output, adaptor.getA(), adaptor.getB()};
    KrnlCallOp::create(rewriter, loc, "om_gemmini_matmul_f32_hw",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// Rank-N f32 MatMul where rhs is a static 2D weight matrix. The runtime
// flattens the leading lhs dimensions into M and dispatches MxK * KxN to
// Gemmini.
struct ONNXMatMulF32BatchedToGemminiLowering
    : public OpConversionPattern<ONNXMatMulOp> {
  ONNXMatMulF32BatchedToGemminiLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXMatMulOp>(typeConverter, ctx, /*benefit=*/5) {}

  LogicalResult matchAndRewrite(ONNXMatMulOp op, ONNXMatMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiMatMulF32Batched(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output =
        createAllocLikeResult(rewriter, loc, resultMemRef, adaptor.getA());
    if (!output)
      return failure();
    SmallVector<Value, 4> operands{output, adaptor.getA(), adaptor.getB()};
    KrnlCallOp::create(rewriter, loc, "om_gemmini_matmul_f32_nd_hw",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// F16 MatMul uses the same Gemmini-backed quantized hardware route as f32,
// with runtime conversion between fp16 storage and fp32 scale selection.
struct ONNXMatMulF16ToGemminiLowering
    : public OpConversionPattern<ONNXMatMulOp> {
  ONNXMatMulF16ToGemminiLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ONNXMatMulOp>(typeConverter, ctx, /*benefit=*/5) {}

  LogicalResult matchAndRewrite(ONNXMatMulOp op, ONNXMatMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedGemminiMatMulF16(op))
      return failure();
    auto resultMemRef =
        dyn_cast<MemRefType>(typeConverter->convertType(op.getY().getType()));
    if (!resultMemRef)
      return failure();
    Location loc = op.getLoc();
    Value output = memref::AllocOp::create(rewriter, loc, resultMemRef);
    SmallVector<Value, 4> operands{output, adaptor.getA(), adaptor.getB()};
    KrnlCallOp::create(rewriter, loc, "om_gemmini_matmul_f16_hw",
        /*numOfOutput=*/1, operands);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// ===--- Pass ---===

struct GemminiLegalizationPass
    : public PassWrapper<GemminiLegalizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemminiLegalizationPass)

  StringRef getArgument() const override { return "convert-onnx-to-gemmini"; }

  StringRef getDescription() const override {
    return "Lower supported ONNX ops to Gemmini dialect or krnl.call stubs";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](TensorType type) -> std::optional<Type> {
      auto ranked = dyn_cast<RankedTensorType>(type);
      if (!ranked)
        return std::nullopt;
      return MemRefType::get(ranked.getShape(), ranked.getElementType());
    });
    typeConverter.addSourceMaterialization(
        [&](OpBuilder &builder, Type resultType, ValueRange inputs,
            Location loc) -> Value {
          if (inputs.size() != 1)
            return Value();
          return UnrealizedConversionCastOp::create(
              builder, loc, resultType, inputs)
              .getResult(0);
        });
    typeConverter.addTargetMaterialization(
        [&](OpBuilder &builder, Type resultType, ValueRange inputs,
            Location loc) -> Value {
          if (inputs.size() != 1)
            return Value();
          return UnrealizedConversionCastOp::create(
              builder, loc, resultType, inputs)
              .getResult(0);
        });

    patterns.insert<
        // Integer / quantized paths
        ONNXMatMulToGemminiLowering,
        ONNXMatMulIntegerDirectToGemminiLowering,
        ONNXMatMulIntegerToGemminiLowering,
        ONNXQLinearConvToGemminiLowering,
        // Float CPU paths
        ONNXConvToGemminiLowering, ONNXConvTransposeToGemminiLowering,
        ONNXResizeToGemminiLowering, ONNXPadToGemminiLowering,
        ONNXConcatToGemminiLowering, ONNXTransposeToGemminiLowering,
        ONNXSliceToGemminiLowering, ONNXSplitToGemminiLowering,
        ONNXReluToGemminiLowering, ONNXSigmoidToGemminiLowering,
        ONNXMulToGemminiLowering,
        ONNXAddToGemminiLowering, ONNXBatchNormToGemminiLowering,
        ONNXGlobalAvgPoolToGemminiLowering, ONNXMaxPoolToGemminiLowering,
        ONNXAvgPoolToGemminiLowering, ONNXSoftmaxToGemminiLowering,
        ONNXGemmToGemminiLowering, ONNXMatMulF32ToGemminiLowering,
        ONNXMatMulF32BatchedToGemminiLowering, ONNXMatMulF16ToGemminiLowering>(
        typeConverter, ctx);

    ConversionTarget target(*ctx);
    // Legal output dialects.
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
        memref::MemRefDialect, KrnlDialect, gemmini::GemminiDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    // Mark each supported op as dynamically illegal when our predicate matches,
    // legal otherwise (so unsupported configurations fall through to standard
    // ONNX→Krnl lowering).
    target.addDynamicallyLegalOp<ONNXMatMulOp>([](ONNXMatMulOp op) {
      return !isSupportedGemminiMatMul(op) &&
             !isSupportedGemminiMatMulF32(op) &&
             !isSupportedGemminiMatMulF32Batched(op) &&
             !isSupportedGemminiMatMulF16(op);
    });
    target.addDynamicallyLegalOp<ONNXMatMulIntegerOp>(
        [](ONNXMatMulIntegerOp op) {
          return !isSupportedGemminiMatMulInteger(op);
        });
    target.addDynamicallyLegalOp<ONNXQLinearConvOp>([](ONNXQLinearConvOp op) {
      return !isSupportedGemminiQLinearConv(op);
    });
    target.addDynamicallyLegalOp<ONNXConvOp>(
        [](ONNXConvOp op) { return !isSupportedGemminiConvF32(op); });
    target.addDynamicallyLegalOp<ONNXConvTransposeOp>(
        [](ONNXConvTransposeOp op) {
          return !isSupportedGemminiConvTransposeF32(op);
        });
    target.addDynamicallyLegalOp<ONNXResizeOp>(
        [](ONNXResizeOp op) { return !isSupportedGemminiResizeF32(op); });
    target.addDynamicallyLegalOp<ONNXPadOp>(
        [](ONNXPadOp op) { return !isSupportedGemminiPadF32(op); });
    target.addDynamicallyLegalOp<ONNXConcatOp>(
        [](ONNXConcatOp op) { return !isSupportedGemminiConcatF32(op); });
    target.addDynamicallyLegalOp<ONNXTransposeOp>(
        [](ONNXTransposeOp op) { return !isSupportedGemminiTransposeF32(op); });
    target.addDynamicallyLegalOp<ONNXSliceOp>(
        [](ONNXSliceOp op) { return !isSupportedGemminiSliceF32(op); });
    target.addDynamicallyLegalOp<ONNXSplitOp>(
        [](ONNXSplitOp op) { return !isSupportedGemminiSplitF32(op); });
    target.addDynamicallyLegalOp<ONNXReluOp>(
        [](ONNXReluOp op) { return !isSupportedGemminiReluF32(op); });
    target.addDynamicallyLegalOp<ONNXSigmoidOp>(
        [](ONNXSigmoidOp op) { return !isSupportedGemminiSigmoidF32(op); });
    target.addDynamicallyLegalOp<ONNXMulOp>(
        [](ONNXMulOp op) { return !isSupportedGemminiMulF32(op); });
    target.addDynamicallyLegalOp<ONNXAddOp>(
        [](ONNXAddOp op) { return !isSupportedGemminiAddF32(op); });
    target.addDynamicallyLegalOp<ONNXBatchNormalizationInferenceModeOp>(
        [](ONNXBatchNormalizationInferenceModeOp op) {
          return !isSupportedGemminiBatchNormF32(op);
        });
    target.addDynamicallyLegalOp<ONNXGlobalAveragePoolOp>(
        [](ONNXGlobalAveragePoolOp op) {
          return !isSupportedGemminiGlobalAvgPoolF32(op);
        });
    target.addDynamicallyLegalOp<ONNXMaxPoolSingleOutOp>(
        [](ONNXMaxPoolSingleOutOp op) {
          return !isSupportedGemminiMaxPoolF32(op);
        });
    target.addDynamicallyLegalOp<ONNXAveragePoolOp>(
        [](ONNXAveragePoolOp op) { return !isSupportedGemminiAvgPoolF32(op); });
    target.addDynamicallyLegalOp<ONNXSoftmaxOp>(
        [](ONNXSoftmaxOp op) { return !isSupportedGemminiSoftmaxF32(op); });
    target.addDynamicallyLegalOp<ONNXGemmOp>(
        [](ONNXGemmOp op) { return !isSupportedGemminiGemmF32(op); });

    // All other ops are legal (pass through to standard lowering).
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createONNXToGemminiPass() {
  return std::make_unique<GemminiLegalizationPass>();
}

void populateONNXToKrnlForGemmini(mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  patterns.insert<ONNXMatMulToGemminiLowering,
      ONNXMatMulIntegerDirectToGemminiLowering,
      ONNXMatMulIntegerToGemminiLowering, ONNXQLinearConvToGemminiLowering,
      ONNXConvToGemminiLowering, ONNXConvTransposeToGemminiLowering,
      ONNXResizeToGemminiLowering, ONNXPadToGemminiLowering,
      ONNXConcatToGemminiLowering, ONNXTransposeToGemminiLowering,
      ONNXSliceToGemminiLowering, ONNXSplitToGemminiLowering,
      ONNXReluToGemminiLowering, ONNXSigmoidToGemminiLowering,
      ONNXMulToGemminiLowering,
      ONNXAddToGemminiLowering, ONNXBatchNormToGemminiLowering,
      ONNXGlobalAvgPoolToGemminiLowering, ONNXMaxPoolToGemminiLowering,
      ONNXAvgPoolToGemminiLowering, ONNXSoftmaxToGemminiLowering,
      ONNXGemmToGemminiLowering, ONNXMatMulF32ToGemminiLowering,
      ONNXMatMulF32BatchedToGemminiLowering, ONNXMatMulF16ToGemminiLowering>(
      typeConverter, ctx);
}

} // namespace onnx_mlir
