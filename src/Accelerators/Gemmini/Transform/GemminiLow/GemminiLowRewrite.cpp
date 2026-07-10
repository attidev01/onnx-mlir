/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.hpp"
#include "src/Accelerators/Gemmini/Transform/GemminiLow/GemminiLowRewrite.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

static bool haveSameI64Attr(Operation *lhs, Operation *rhs, StringRef name) {
  return lhs->getAttrOfType<IntegerAttr>(name) ==
         rhs->getAttrOfType<IntegerAttr>(name);
}

static bool haveSameTileAttrs(Operation *lhs, Operation *rhs) {
  return haveSameI64Attr(lhs, rhs, "spad_offset_rows") &&
         haveSameI64Attr(lhs, rhs, "tile_rows") &&
         haveSameI64Attr(lhs, rhs, "tile_cols");
}

static bool areRedundantConfigOps(
    gemmini::GemminiLowConfigOp previous, gemmini::GemminiLowConfigOp current) {
  return previous.getDataflow() == current.getDataflow();
}

static bool areRedundantMvinOps(
    gemmini::GemminiLowMvinOp previous, gemmini::GemminiLowMvinOp current) {
  return previous.getSource() == current.getSource() &&
         haveSameTileAttrs(previous.getOperation(), current.getOperation());
}

static bool areRedundantMvoutOps(
    gemmini::GemminiLowMvoutOp previous, gemmini::GemminiLowMvoutOp current) {
  return previous.getDest() == current.getDest() &&
         haveSameTileAttrs(previous.getOperation(), current.getOperation());
}

static bool isRedundantAfterPrevious(Operation *previous, Operation *current) {
  if (!previous)
    return false;
  if (auto currentConfig = dyn_cast<gemmini::GemminiLowConfigOp>(current)) {
    if (auto previousConfig = dyn_cast<gemmini::GemminiLowConfigOp>(previous))
      return areRedundantConfigOps(previousConfig, currentConfig);
    return false;
  }
  if (isa<gemmini::GemminiLowFenceOp>(current))
    return isa<gemmini::GemminiLowFenceOp>(previous);
  if (auto currentMvin = dyn_cast<gemmini::GemminiLowMvinOp>(current)) {
    if (auto previousMvin = dyn_cast<gemmini::GemminiLowMvinOp>(previous))
      return areRedundantMvinOps(previousMvin, currentMvin);
    return false;
  }
  if (auto currentMvout = dyn_cast<gemmini::GemminiLowMvoutOp>(current)) {
    if (auto previousMvout = dyn_cast<gemmini::GemminiLowMvoutOp>(previous))
      return areRedundantMvoutOps(previousMvout, currentMvout);
    return false;
  }
  return false;
}

static bool rewriteBlock(Block &block) {
  bool changed = false;
  SmallVector<Operation *> toErase;
  Operation *previous = nullptr;

  for (Operation &op : block) {
    if (isRedundantAfterPrevious(previous, &op)) {
      toErase.push_back(&op);
      changed = true;
      continue;
    }
    previous = &op;
  }

  for (Operation *op : toErase)
    op->erase();
  return changed;
}

static bool rewriteRegion(Region &region) {
  bool changed = false;
  for (Block &block : region) {
    changed |= rewriteBlock(block);
    for (Operation &op : block)
      for (Region &nestedRegion : op.getRegions())
        changed |= rewriteRegion(nestedRegion);
  }
  return changed;
}

struct GemminiLowRewritePass
    : public PassWrapper<GemminiLowRewritePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemminiLowRewritePass)

  StringRef getArgument() const override { return "gemmini-low-rewrite"; }
  StringRef getDescription() const override {
    return "Canonicalize GemminiLow instruction streams before LLVM lowering";
  }
  void runOnOperation() override {
    bool changed = false;
    for (Region &region : getOperation()->getRegions())
      changed |= rewriteRegion(region);
    (void)changed;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createGemminiLowRewritePass() {
  return std::make_unique<GemminiLowRewritePass>();
}

} // namespace onnx_mlir
