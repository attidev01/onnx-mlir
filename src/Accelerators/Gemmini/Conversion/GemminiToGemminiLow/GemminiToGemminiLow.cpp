/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ GemminiToGemminiLow.cpp - Lower Gemmini dialect ops ----------===//
//
// Rewrites scheduled high-level Gemmini operations into GemminiLow operations.
// GemminiLow makes memory movement, execute, and fence operations explicit so
// later passes can allocate scratchpad rows and lower directly to RoCC commands.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiOps.hpp"
#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.hpp"
#include "src/Accelerators/Gemmini/Conversion/GemminiToGemminiLow/GemminiToGemminiLow.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

template <typename SrcOp, typename DstOp>
static void copyAttrsExceptNamed(
    PatternRewriter &rewriter, SrcOp srcOp, DstOp dstOp,
    ArrayRef<StringRef> excluded = {}) {
  // TableGen builders copy operands and required attributes explicitly. This
  // helper preserves any auxiliary scheduling/debug attributes that should
  // survive the dialect boundary without duplicating the core operands.
  llvm::SmallDenseSet<StringRef> excludedSet(excluded.begin(), excluded.end());
  for (NamedAttribute attr : srcOp->getAttrs()) {
    if (excludedSet.contains(attr.getName().strref()))
      continue;
    dstOp->setAttr(attr.getName(), attr.getValue());
  }
}

struct GemminiToGemminiLowPass
    : public PassWrapper<GemminiToGemminiLowPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemminiToGemminiLowPass)

  StringRef getArgument() const override { return "convert-gemmini-to-low"; }
  StringRef getDescription() const override {
    return "Lower scheduled Gemmini instruction ops into GemminiLow IR";
  }
  void runOnOperation() override {
    // Collect first so replacements do not invalidate the function walk.
    SmallVector<Operation *> worklist;
    getOperation().walk([&](Operation *op) {
      if (isa<gemmini::GemminiConfigOp, gemmini::GemminiMvinOp,
              gemmini::GemminiMvoutOp, gemmini::GemminiFenceOp,
              gemmini::GemminiMatmulOp>(op))
        worklist.push_back(op);
    });

    PatternRewriter rewriter(&getContext());
    for (Operation *op : worklist) {
      rewriter.setInsertionPoint(op);

      if (auto configOp = dyn_cast<gemmini::GemminiConfigOp>(op)) {
        // High-level config stores the dataflow mode; the low-level op also
        // carries transpose flags, which default to false until scheduling
        // starts using them.
        auto lowOp = gemmini::GemminiLowConfigOp::create(
            rewriter, configOp.getLoc(), configOp.getDataflowAttr(),
            mlir::BoolAttr{}, mlir::BoolAttr{});
        copyAttrsExceptNamed(rewriter, configOp, lowOp, {"dataflow"});
        rewriter.eraseOp(configOp);
        continue;
      }
      if (auto mvinOp = dyn_cast<gemmini::GemminiMvinOp>(op)) {
        // Preserve the source memref and static tile shape while moving the op
        // into the instruction-level dialect.
        auto lowOp = gemmini::GemminiLowMvinOp::create(rewriter, mvinOp.getLoc(),
            mvinOp.getSource(), mvinOp.getSpadOffsetRowsAttr(),
            mvinOp.getTileRowsAttr(), mvinOp.getTileColsAttr());
        copyAttrsExceptNamed(rewriter, mvinOp, lowOp,
            {"source", "spad_offset_rows", "tile_rows", "tile_cols"});
        rewriter.eraseOp(mvinOp);
        continue;
      }
      if (auto mvoutOp = dyn_cast<gemmini::GemminiMvoutOp>(op)) {
        // Mvout is the mirror of mvin: it writes from a scratchpad row back to
        // the destination memref.
        auto lowOp = gemmini::GemminiLowMvoutOp::create(rewriter,
            mvoutOp.getLoc(), mvoutOp.getDest(), mvoutOp.getSpadOffsetRowsAttr(),
            mvoutOp.getTileRowsAttr(), mvoutOp.getTileColsAttr());
        copyAttrsExceptNamed(rewriter, mvoutOp, lowOp,
            {"dest", "spad_offset_rows", "tile_rows", "tile_cols"});
        rewriter.eraseOp(mvoutOp);
        continue;
      }
      if (auto fenceOp = dyn_cast<gemmini::GemminiFenceOp>(op)) {
        auto lowOp =
            gemmini::GemminiLowFenceOp::create(rewriter, fenceOp.getLoc());
        copyAttrsExceptNamed(rewriter, fenceOp, lowOp);
        rewriter.eraseOp(fenceOp);
        continue;
      }
      if (auto matmulOp = dyn_cast<gemmini::GemminiMatmulOp>(op)) {
        // The low-level matmul keeps the visible operands for MLIR dependence
        // tracking while later passes attach concrete scratchpad/acc offsets.
        auto lowOp = gemmini::GemminiLowMatmulOp::create(rewriter,
            matmulOp.getLoc(), matmulOp.getLhs(), matmulOp.getRhs(),
            matmulOp.getOut(), matmulOp.getModeAttr(),
            matmulOp.getDataflowAttr());
        copyAttrsExceptNamed(rewriter, matmulOp, lowOp,
            {"lhs", "rhs", "out", "mode", "dataflow"});
        rewriter.eraseOp(matmulOp);
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createGemminiToGemminiLowPass() {
  return std::make_unique<GemminiToGemminiLowPass>();
}

} // namespace onnx_mlir
