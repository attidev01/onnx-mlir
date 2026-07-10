/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- GemminiLowToLLVM.cpp - Lowering from Gemmini to LLVM --------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Accelerators/Gemmini/Conversion/GemminiLowToLLVM/GemminiLowToLLVM.hpp"
#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.hpp"
#include "src/Accelerators/Gemmini/Support/GemminiTargetInfo.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

// funct field values used in COMPUTE_PRELOADED and COMPUTE_ACCUMULATED only;
// config/mvin/mvout/preload functs are embedded directly in the asm strings.
static constexpr uint64_t kComputePreloadedFunct = 4;
static constexpr uint64_t kComputeAccumulatedFunct = 5;

static void emitInlineAsmVoid(ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<Value> operands, StringRef asmString) {
  LLVM::InlineAsmOp::create(rewriter, loc,
      LLVM::LLVMVoidType::get(rewriter.getContext()), operands, asmString,
      "r,r", /*has_side_effects=*/true, /*is_align_stack=*/false,
      LLVM::TailCallKind::None, LLVM::AsmDialectAttr(), ArrayAttr());
}

static Value getI8PtrFromMemRef(
    ConversionPatternRewriter &rewriter, Location loc, Value llvmMemRef) {
  MemRefDescriptor descriptor(llvmMemRef);
  Value alignedPtr = descriptor.alignedPtr(rewriter, loc);
  return LLVM::BitcastOp::create(
      rewriter, loc, krnl::getI8PointerType(rewriter.getContext()), alignedPtr);
}

static Value i64Const(ConversionPatternRewriter &rewriter, Location loc,
    uint64_t value) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(value));
}

static uint64_t getI64Attr(
    Operation *op, StringRef name, uint64_t defaultValue) {
  if (auto attr = op->template getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return defaultValue;
}

class GemminiConfigLowering
    : public ConvertOpToLLVMPattern<gemmini::GemminiLowConfigOp> {
public:
  GemminiConfigLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gemmini::GemminiLowConfigOp>(typeConverter) {}

  LogicalResult matchAndRewrite(gemmini::GemminiLowConfigOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Location loc = op.getLoc();
    uint64_t dataflow = op.getDataflow() == "ws" ? 1 : 0;
    // CONFIG_EX rs1/rs2 layout (rs1[1:0] = 0b00):
    //   rs1[63:32] sys_acc_shift — IEEE 754 float; 1.0f = 0x3F800000
    //   rs1[31:16] a_stride     — A row stride in scratchpad (1 = contiguous rows)
    //   rs1[9]     b_transpose
    //   rs1[8]     a_transpose
    //   rs1[2]     dataflow     — 1=WS, 0=OS
    //   rs2[63:48] c_stride     — C row stride in accumulator (1 = contiguous rows)
    //   rs2[31:0]  sys_shift    — right-shift from acc_t to elem_t (0 = none)
    constexpr uint64_t kSysAccShift = 0x3F800000ULL; // 1.0f identity
    uint64_t aT = op.getATranspose().value_or(false) ? 1ULL : 0ULL;
    uint64_t bT = op.getBTranspose().value_or(false) ? 1ULL : 0ULL;
    Value rs1 = i64Const(rewriter, loc,
        (kSysAccShift << 32) | (1ULL << 16) | (bT << 9) | (aT << 8) | (uint64_t(dataflow) << 2));
    Value rs2 = i64Const(rewriter, loc, 1ULL << 48); // c_stride=1
    emitInlineAsmVoid(
        rewriter, loc, {rs1, rs2}, ".insn r CUSTOM_3, 0x3, 0, x0, $0, $1");
    rewriter.eraseOp(op);
    return success();
  }
};

class GemminiMvinLowering
    : public ConvertOpToLLVMPattern<gemmini::GemminiLowMvinOp> {
public:
  GemminiMvinLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gemmini::GemminiLowMvinOp>(typeConverter) {}

  LogicalResult matchAndRewrite(gemmini::GemminiLowMvinOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // CONFIG_MVIN (funct=0, rs1[1:0]=0b01): set load stride and scale.
    // The simulator zero-initializes load_scale, so this must be sent before
    // every mvin to set MVIN_SCALE_IDENTITY (1.0f) and the row stride in bytes.
    //   rs1[63:32] = mvin_scale  (1.0f = 0x3F800000)
    //   rs1[4:3]   = state_id=0  (matches mvin_funct=2 → mvin(xs1,xs2,0))
    //   rs1[1:0]   = 0b01        (CONFIG_MVIN mode)
    //   rs2        = row stride in bytes = tile_cols * sizeof(elem_t)
    auto srcTy = dyn_cast<MemRefType>(op.getSource().getType());
    uint64_t elemBytes =
        srcTy ? uint64_t(srcTy.getElementType().getIntOrFloatBitWidth() / 8) : 1;
    uint64_t mvinStride = uint64_t(op.getTileCols()) * elemBytes;
    constexpr uint64_t kMvinScaleIdentity = 0x3F800000ULL;
    Value cfgRs1 = i64Const(rewriter, loc, (kMvinScaleIdentity << 32) | 0x01ULL);
    Value cfgRs2 = i64Const(rewriter, loc, mvinStride);
    emitInlineAsmVoid(
        rewriter, loc, {cfgRs1, cfgRs2}, ".insn r CUSTOM_3, 0x3, 0, x0, $0, $1");
    Value rs1 = getI8PtrFromMemRef(rewriter, loc, adaptor.getSource());
    uint64_t packed = (uint64_t(op.getTileRows()) << (32 + 16)) |
                      (uint64_t(op.getTileCols()) << 32) |
                      uint64_t(op.getSpadOffsetRows());
    Value rs2 = i64Const(rewriter, loc, packed);
    emitInlineAsmVoid(
        rewriter, loc, {rs1, rs2}, ".insn r CUSTOM_3, 0x3, 2, x0, $0, $1");
    rewriter.eraseOp(op);
    return success();
  }
};

class GemminiMvoutLowering
    : public ConvertOpToLLVMPattern<gemmini::GemminiLowMvoutOp> {
public:
  GemminiMvoutLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gemmini::GemminiLowMvoutOp>(typeConverter) {}

  LogicalResult matchAndRewrite(gemmini::GemminiLowMvoutOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Determine output width from the destination memref element type.
    // full=1 (bit 29 in sp_addr): writes acc_t (int32, 4 bytes) without clipping.
    // full=0: applies acc_scale, clips to elem_t (int8, 1 byte).
    auto destTy = dyn_cast<MemRefType>(op.getDest().getType());
    bool fullRead = destTy && destTy.getElementType().isInteger(32);
    uint64_t bytesPerElem = fullRead ? 4u : 1u;

    // CONFIG_MVOUT: rs1[1:0]=0b10, rs2[31:0]=store_stride_bytes, rs2[63:32]=acc_scale
    constexpr uint64_t kAccScaleIdentity = 0x3F800000ULL;
    uint64_t strideBytes = uint64_t(op.getTileCols()) * bytesPerElem;
    Value cfgRs1 = i64Const(rewriter, loc, 0x02ULL);
    Value cfgRs2 = i64Const(rewriter, loc, (kAccScaleIdentity << 32) | strideBytes);
    emitInlineAsmVoid(
        rewriter, loc, {cfgRs1, cfgRs2}, ".insn r CUSTOM_3, 0x3, 0, x0, $0, $1");
    Value rs1 = getI8PtrFromMemRef(rewriter, loc, adaptor.getDest());
    // Bit 31: accumulator (not scratchpad) source.
    // Bit 29: full-width read — writes acc_t (int32) directly; without it the
    //         hardware applies acc_scale and clips to elem_t (int8).
    uint64_t accAddr = 0x80000000ULL |
                       (fullRead ? 0x20000000ULL : 0ULL) |
                       uint64_t(op.getSpadOffsetRows());
    uint64_t packed = (uint64_t(op.getTileRows()) << (32 + 16)) |
                      (uint64_t(op.getTileCols()) << 32) |
                      accAddr;
    Value rs2 = i64Const(rewriter, loc, packed);
    emitInlineAsmVoid(
        rewriter, loc, {rs1, rs2}, ".insn r CUSTOM_3, 0x3, 3, x0, $0, $1");
    rewriter.eraseOp(op);
    return success();
  }
};

class GemminiFenceLowering
    : public ConvertOpToLLVMPattern<gemmini::GemminiLowFenceOp> {
public:
  GemminiFenceLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gemmini::GemminiLowFenceOp>(typeConverter) {}

  LogicalResult matchAndRewrite(gemmini::GemminiLowFenceOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    LLVM::InlineAsmOp::create(rewriter, op.getLoc(),
        LLVM::LLVMVoidType::get(rewriter.getContext()), ValueRange(), "fence",
        "", /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::TailCallKind::None, LLVM::AsmDialectAttr(), ArrayAttr());
    rewriter.eraseOp(op);
    return success();
  }
};

class GemminiMatmulLowering
    : public ConvertOpToLLVMPattern<gemmini::GemminiLowMatmulOp> {
public:
  GemminiMatmulLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gemmini::GemminiLowMatmulOp>(typeConverter) {}

  LogicalResult matchAndRewrite(gemmini::GemminiLowMatmulOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Location loc = op.getLoc();
    uint64_t lhsOffset = getI64Attr(op, "lhs_spad_offset_rows", 0);
    uint64_t rhsOffset = getI64Attr(
        op, "rhs_spad_offset_rows", gemmini::GemminiTargetInfo::dim);
    uint64_t accOffset = getI64Attr(op, "acc_offset_rows", 0);
    uint64_t rows = gemmini::GemminiTargetInfo::dim;
    uint64_t cols = gemmini::GemminiTargetInfo::dim;
    uint64_t lhsPacked = (rows << (32 + 16)) | (cols << 32) | lhsOffset;
    uint64_t funct = op.getMode() == "identity"
                         ? kComputePreloadedFunct
                         : kComputeAccumulatedFunct;
    if (funct == kComputePreloadedFunct) {
      // PRELOAD: rs1=B(spad), rs2=C(accumulator, bit31 set)
      uint64_t bPacked = (rows << (32 + 16)) | (cols << 32) | rhsOffset;
      uint64_t cPacked =
          (rows << (32 + 16)) | (cols << 32) | (0x80000000ULL | accOffset);
      Value pRs1 = i64Const(rewriter, loc, bPacked);
      Value pRs2 = i64Const(rewriter, loc, cPacked);
      emitInlineAsmVoid(
          rewriter, loc, {pRs1, pRs2}, ".insn r CUSTOM_3, 0x3, 6, x0, $0, $1");
    }
    Value rs1 = i64Const(rewriter, loc, lhsPacked);
    // rs2=0xFFFFFFFF: sentinel for "no D bias" — Gemmini initialises results to 0
    Value rs2 = i64Const(rewriter, loc, 0xFFFFFFFFULL);
    emitInlineAsmVoid(rewriter, loc, {rs1, rs2},
        funct == kComputePreloadedFunct
            ? ".insn r CUSTOM_3, 0x3, 4, x0, $0, $1"
            : ".insn r CUSTOM_3, 0x3, 5, x0, $0, $1");
    rewriter.eraseOp(op);
    return success();
  }
};

struct GemminiLowToLLVMPass
    : public PassWrapper<GemminiLowToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemminiLowToLLVMPass)

  StringRef getArgument() const override {
    return "convert-gemmini-low-to-llvm";
  }
  StringRef getDescription() const override {
    return "Lower Gemmini ops to direct RISC-V custom3 inline assembly";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);
    patterns.insert<GemminiConfigLowering, GemminiMvinLowering,
        GemminiMvoutLowering, GemminiFenceLowering,
        GemminiMatmulLowering>(typeConverter);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect, func::FuncDialect,
        scf::SCFDialect>();
    target.addIllegalDialect<gemmini::GemminiLowDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isa<gemmini::GemminiLowConfigOp, gemmini::GemminiLowMvinOp,
          gemmini::GemminiLowMvoutOp, gemmini::GemminiLowFenceOp,
          gemmini::GemminiLowMatmulOp>(op);
    });

    if (failed(applyPartialConversion(getOperation(), target,
            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createGemminiLowToLLVMPass() {
  return std::make_unique<GemminiLowToLLVMPass>();
}

} // namespace onnx_mlir
