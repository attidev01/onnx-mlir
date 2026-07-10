/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- GemminiTiling.cpp - Gemmini Transform Pass -----------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiOps.hpp"
#include "src/Accelerators/Gemmini/Support/GemminiTargetInfo.hpp"
#include "src/Accelerators/Gemmini/Transform/Gemmini/GemminiTiling.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

struct GemminiTilingPass
    : public PassWrapper<GemminiTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemminiTilingPass)

  StringRef getArgument() const override { return "gemmini-tiling"; }

  StringRef getDescription() const override {
    return "Tile Gemmini matmul ops into 16x16 loop nests with explicit mvin/mvout";
  }

  void runOnOperation() override {
    SmallVector<gemmini::GemminiMatmulOp> worklist;
    getOperation().walk([&](gemmini::GemminiMatmulOp op) {
      if (op->hasAttr("gemmini.high_level"))
        worklist.push_back(op);
    });

    for (gemmini::GemminiMatmulOp op : worklist)
      tileMatmul(op);
  }

  void tileMatmul(gemmini::GemminiMatmulOp op) {
    constexpr int64_t tile = gemmini::GemminiTargetInfo::dim;
    auto lhsTy = dyn_cast<MemRefType>(op.getLhs().getType());
    auto rhsTy = dyn_cast<MemRefType>(op.getRhs().getType());
    auto outTy = dyn_cast<MemRefType>(op.getOut().getType());
    if (!lhsTy || !rhsTy || !outTy || !lhsTy.hasStaticShape() ||
        !rhsTy.hasStaticShape() || !outTy.hasStaticShape())
      return;

    PatternRewriter rewriter(op.getContext());
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, tile);
    Value upperM =
        arith::ConstantIndexOp::create(rewriter, loc, lhsTy.getShape()[0]);
    Value upperN =
        arith::ConstantIndexOp::create(rewriter, loc, rhsTy.getShape()[1]);
    Value upperK =
        arith::ConstantIndexOp::create(rewriter, loc, lhsTy.getShape()[1]);

    auto ws = op.getDataflowAttr();
    auto identity = rewriter.getStringAttr("identity");
    auto accumulate = rewriter.getStringAttr("accumulate");
    Type inputElemTy = lhsTy.getElementType();
    Type outputElemTy = outTy.getElementType();
    auto inputTileTy = MemRefType::get({tile, tile}, inputElemTy);
    auto outputTileTy = MemRefType::get({tile, tile}, outputElemTy);

    gemmini::GemminiConfigOp::create(rewriter, loc, ws, mlir::BoolAttr{}, mlir::BoolAttr{});

    auto iLoop = scf::ForOp::create(rewriter, loc, zero, upperM, step);
    rewriter.setInsertionPointToStart(iLoop.getBody());
    auto jLoop = scf::ForOp::create(rewriter, loc, zero, upperN, step);
    rewriter.setInsertionPointToStart(jLoop.getBody());

    Value mRemaining =
        arith::SubIOp::create(rewriter, loc, upperM, iLoop.getInductionVar());
    Value nRemaining =
        arith::SubIOp::create(rewriter, loc, upperN, jLoop.getInductionVar());
    Value tileRows = minIndex(rewriter, loc, mRemaining, step);
    Value tileCols = minIndex(rewriter, loc, nRemaining, step);

    SmallVector<OpFoldResult> outOffsets{jLoop.getInductionVar(), zero};
    outOffsets[0] = iLoop.getInductionVar();
    outOffsets[1] = jLoop.getInductionVar();
    SmallVector<OpFoldResult> outSizes{tileRows, tileCols};
    SmallVector<OpFoldResult> outStrides{rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)};
    Value cTile = memref::SubViewOp::create(
        rewriter, loc, op.getOut(), outOffsets, outSizes, outStrides);
    Value paddedOutTile =
        createPaddedTile(rewriter, loc, outputTileTy, tileRows, tileCols);

    auto kLoop = scf::ForOp::create(rewriter, loc, zero, upperK, step);
    rewriter.setInsertionPointToStart(kLoop.getBody());

    Value kIsZero = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
        kLoop.getInductionVar(), zero);
    Value kRemaining =
        arith::SubIOp::create(rewriter, loc, upperK, kLoop.getInductionVar());
    Value tileDepth = minIndex(rewriter, loc, kRemaining, step);
    Value two = arith::ConstantIndexOp::create(rewriter, loc, 2);
    Value kTileOrdinal = arith::DivUIOp::create(
        rewriter, loc, kLoop.getInductionVar(), step);
    Value bufferSlot = arith::RemUIOp::create(rewriter, loc, kTileOrdinal, two);
    Value useBufferZero = arith::CmpIOp::create(rewriter, loc,
        arith::CmpIPredicate::eq, bufferSlot, zero);

    SmallVector<OpFoldResult> aOffsets{iLoop.getInductionVar(),
        kLoop.getInductionVar()};
    SmallVector<OpFoldResult> bOffsets{kLoop.getInductionVar(),
        jLoop.getInductionVar()};
    SmallVector<OpFoldResult> aTileSizes{tileRows, tileDepth};
    SmallVector<OpFoldResult> bTileSizes{tileDepth, tileCols};
    SmallVector<OpFoldResult> unitStrides{rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)};
    Value aTile = memref::SubViewOp::create(
        rewriter, loc, op.getLhs(), aOffsets, aTileSizes, unitStrides);
    Value bTile = memref::SubViewOp::create(
        rewriter, loc, op.getRhs(), bOffsets, bTileSizes, unitStrides);

    Value paddedATile = createPaddedOperandTile(
        rewriter, loc, inputTileTy, aTile, tileRows, tileDepth);
    Value paddedBTile = createPaddedOperandTile(
        rewriter, loc, inputTileTy, bTile, tileDepth, tileCols);

    auto bufferIfOp = scf::IfOp::create(rewriter, loc, useBufferZero, true);
    rewriter.setInsertionPointToStart(&bufferIfOp.getThenRegion().front());
    emitBufferedMatmulTile(rewriter, loc, paddedATile, paddedBTile,
        paddedOutTile, kIsZero, identity, accumulate, ws,
        /*bufferSlot=*/0, /*lhsSpadOffset=*/0, /*rhsSpadOffset=*/tile);
    rewriter.setInsertionPointToStart(&bufferIfOp.getElseRegion().front());
    emitBufferedMatmulTile(rewriter, loc, paddedATile, paddedBTile,
        paddedOutTile, kIsZero, identity, accumulate, ws,
        /*bufferSlot=*/1, /*lhsSpadOffset=*/2 * tile,
        /*rhsSpadOffset=*/3 * tile);

    rewriter.setInsertionPointAfter(kLoop);
    gemmini::GemminiFenceOp::create(rewriter, loc);
    gemmini::GemminiMvoutOp::create(
        rewriter, loc, paddedOutTile, rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(tile), rewriter.getI64IntegerAttr(tile));
    copyBackOutputTile(rewriter, loc, paddedOutTile, cTile, tileRows, tileCols);

    rewriter.eraseOp(op);
  }

  static void setMatmulScheduleAttrs(PatternRewriter &rewriter,
      gemmini::GemminiMatmulOp op, int64_t bufferSlot, int64_t lhsSpadOffset,
      int64_t rhsSpadOffset) {
    op->setAttr("gemmini.lowered", rewriter.getUnitAttr());
    op->setAttr("gemmini.buffer_slot",
        rewriter.getI64IntegerAttr(bufferSlot));
    op->setAttr("lhs_spad_offset_rows",
        rewriter.getI64IntegerAttr(lhsSpadOffset));
    op->setAttr("rhs_spad_offset_rows",
        rewriter.getI64IntegerAttr(rhsSpadOffset));
  }

  static void emitBufferedMatmulTile(PatternRewriter &rewriter, Location loc,
      Value paddedATile, Value paddedBTile, Value paddedOutTile, Value kIsZero,
      StringAttr identity, StringAttr accumulate, StringAttr dataflow,
      int64_t bufferSlot, int64_t lhsSpadOffset, int64_t rhsSpadOffset) {
    constexpr int64_t tile = gemmini::GemminiTargetInfo::dim;
    gemmini::GemminiMvinOp::create(rewriter, loc, paddedATile,
        rewriter.getI64IntegerAttr(lhsSpadOffset),
        rewriter.getI64IntegerAttr(tile), rewriter.getI64IntegerAttr(tile));
    gemmini::GemminiMvinOp::create(rewriter, loc, paddedBTile,
        rewriter.getI64IntegerAttr(rhsSpadOffset),
        rewriter.getI64IntegerAttr(tile), rewriter.getI64IntegerAttr(tile));

    auto modeIfOp = scf::IfOp::create(rewriter, loc, kIsZero, true);
    rewriter.setInsertionPointToStart(&modeIfOp.getThenRegion().front());
    auto idOp = gemmini::GemminiMatmulOp::create(rewriter, loc, paddedATile,
        paddedBTile, paddedOutTile, identity, dataflow);
    setMatmulScheduleAttrs(
        rewriter, idOp, bufferSlot, lhsSpadOffset, rhsSpadOffset);
    rewriter.setInsertionPointToStart(&modeIfOp.getElseRegion().front());
    auto accOp = gemmini::GemminiMatmulOp::create(rewriter, loc, paddedATile,
        paddedBTile, paddedOutTile, accumulate, dataflow);
    setMatmulScheduleAttrs(
        rewriter, accOp, bufferSlot, lhsSpadOffset, rhsSpadOffset);
    rewriter.setInsertionPointAfter(modeIfOp);
  }

  static Value minIndex(PatternRewriter &rewriter, Location loc, Value lhs,
      Value rhs) {
    Value lhsLessThanRhs = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, lhs, rhs);
    return arith::SelectOp::create(rewriter, loc, lhsLessThanRhs, lhs, rhs);
  }

  static Value createZeroValue(
      PatternRewriter &rewriter, Location loc, Type elemTy) {
    auto zeroAttr = rewriter.getZeroAttr(elemTy);
    return arith::ConstantOp::create(rewriter, loc, elemTy, zeroAttr);
  }

  static void zeroFillTile(PatternRewriter &rewriter, Location loc, Value tile) {
    auto tileTy = cast<MemRefType>(tile.getType());
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upper0 =
        arith::ConstantIndexOp::create(rewriter, loc, tileTy.getShape()[0]);
    Value upper1 =
        arith::ConstantIndexOp::create(rewriter, loc, tileTy.getShape()[1]);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value zeroElem = createZeroValue(rewriter, loc, tileTy.getElementType());

    auto outer = scf::ForOp::create(rewriter, loc, zero, upper0, step);
    rewriter.setInsertionPointToStart(outer.getBody());
    auto inner = scf::ForOp::create(rewriter, loc, zero, upper1, step);
    rewriter.setInsertionPointToStart(inner.getBody());
    memref::StoreOp::create(rewriter, loc, zeroElem, tile,
        ValueRange{outer.getInductionVar(), inner.getInductionVar()});
    rewriter.setInsertionPointAfter(inner);
    rewriter.setInsertionPointAfter(outer);
  }

  static Value createPaddedTile(PatternRewriter &rewriter, Location loc,
      MemRefType tileTy, Value usedRows, Value usedCols) {
    Value paddedTile = memref::AllocaOp::create(rewriter, loc, tileTy);
    zeroFillTile(rewriter, loc, paddedTile);
    (void)usedRows;
    (void)usedCols;
    return paddedTile;
  }

  static Value createPaddedOperandTile(PatternRewriter &rewriter, Location loc,
      MemRefType tileTy, Value srcTile, Value usedRows, Value usedCols) {
    Value paddedTile = createPaddedTile(rewriter, loc, tileTy, usedRows, usedCols);
    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{usedRows, usedCols};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)};
    Value dstSubview = memref::SubViewOp::create(
        rewriter, loc, paddedTile, offsets, sizes, strides);
    memref::CopyOp::create(rewriter, loc, srcTile, dstSubview);
    return paddedTile;
  }

  static void copyBackOutputTile(PatternRewriter &rewriter, Location loc,
      Value paddedTile, Value dstTile, Value usedRows, Value usedCols) {
    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{usedRows, usedCols};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)};
    Value srcSubview = memref::SubViewOp::create(
        rewriter, loc, paddedTile, offsets, sizes, strides);
    memref::CopyOp::create(rewriter, loc, srcSubview, dstTile);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createGemminiTilingPass() {
  return std::make_unique<GemminiTilingPass>();
}

} // namespace onnx_mlir
