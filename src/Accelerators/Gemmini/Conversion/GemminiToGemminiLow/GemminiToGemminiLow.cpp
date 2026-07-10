/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
        auto lowOp = gemmini::GemminiLowConfigOp::create(
            rewriter, configOp.getLoc(), configOp.getDataflowAttr(),
            mlir::BoolAttr{}, mlir::BoolAttr{});
        copyAttrsExceptNamed(rewriter, configOp, lowOp, {"dataflow"});
        rewriter.eraseOp(configOp);
        continue;
      }
      if (auto mvinOp = dyn_cast<gemmini::GemminiMvinOp>(op)) {
        auto lowOp = gemmini::GemminiLowMvinOp::create(rewriter, mvinOp.getLoc(),
            mvinOp.getSource(), mvinOp.getSpadOffsetRowsAttr(),
            mvinOp.getTileRowsAttr(), mvinOp.getTileColsAttr());
        copyAttrsExceptNamed(rewriter, mvinOp, lowOp,
            {"source", "spad_offset_rows", "tile_rows", "tile_cols"});
        rewriter.eraseOp(mvinOp);
        continue;
      }
      if (auto mvoutOp = dyn_cast<gemmini::GemminiMvoutOp>(op)) {
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
