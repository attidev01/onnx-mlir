/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ StaticScratchpadAllocation.cpp - GemminiLow address pass ------===//
//
// Assigns fixed scratchpad rows to GemminiLow mvin/mvout operations. Keeping
// this policy in one pass prevents later lowering stages from duplicating
// hardware capacity calculations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.hpp"
#include "src/Accelerators/Gemmini/Support/GemminiTargetInfo.hpp"
#include "src/Accelerators/Gemmini/Transform/GemminiLow/StaticScratchpadAllocation.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

static IntegerAttr getI64Attr(Operation *op, StringRef name) {
  // GemminiLow address decisions are stored as i64 attributes for easy
  // propagation into LLVM constants.
  return op->getAttrOfType<IntegerAttr>(name);
}

static IntegerAttr getOrCreateI64Attr(Operation *op, StringRef name,
    int64_t fallback) {
  // Preserve explicit scheduler choices, otherwise use the simple static policy
  // in this pass.
  if (IntegerAttr attr = getI64Attr(op, name))
    return attr;
  return IntegerAttr::get(IntegerType::get(op->getContext(), 64), fallback);
}

struct StaticScratchpadAllocationPass : public PassWrapper<
                                            StaticScratchpadAllocationPass,
                                            OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticScratchpadAllocationPass)

  StringRef getArgument() const override {
    return "gemmini-static-scratchpad-allocation";
  }
  StringRef getDescription() const override {
    return "Assign compile-time Gemmini scratchpad and accumulator offsets";
  }

  void runOnOperation() override {
    constexpr int64_t tileRows = gemmini::GemminiTargetInfo::dim;
    constexpr int64_t spadRows = gemmini::GemminiTargetInfo::scratchpadRows();
    constexpr int64_t accRows = gemmini::GemminiTargetInfo::accRows;

    int64_t nextSpadA = 0;
    int64_t nextSpadB = tileRows;
    int64_t nextAcc = 0;
    bool assignLhs = true;
    // Use alternating scratchpad slots for A/B mvins. This is conservative but
    // gives the lowerer concrete addresses without needing global scheduling.
    // Track the accumulator row assigned to each group. A "group" spans from
    // one fence to the next; every matmul in the group must target the same
    // accumulator row because the final mvout reads exactly one row. This is
    // especially important when the scheduler emits scf.if branches: both
    // branches are present in the IR, but only one executes at runtime.
    int64_t groupFirstAcc = 0;
    bool inGroup = false;

    getOperation().walk([&](Operation *op) {
      if (auto mvin = dyn_cast<gemmini::GemminiLowMvinOp>(op)) {
        // Assign alternating A/B scratchpad rows to incoming tiles.
        int64_t fallback = assignLhs ? nextSpadA : nextSpadB;
        op->setAttr("spad_offset_rows",
            getOrCreateI64Attr(op, "spad_offset_rows", fallback));
        assignLhs = !assignLhs;
      } else if (auto matmul = dyn_cast<gemmini::GemminiLowMatmulOp>(op)) {
        // Attach the concrete scratchpad rows consumed by this matmul and the
        // accumulator row where the result is produced.
        op->setAttr("lhs_spad_offset_rows",
            getOrCreateI64Attr(op, "lhs_spad_offset_rows", nextSpadA));
        op->setAttr("rhs_spad_offset_rows",
            getOrCreateI64Attr(op, "rhs_spad_offset_rows", nextSpadB));
        if (!inGroup) {
          groupFirstAcc = nextAcc;
          inGroup = true;
          nextAcc += tileRows;
          if (nextAcc >= accRows)
            nextAcc = 0;
        }
        IntegerAttr accAttr =
            getOrCreateI64Attr(op, "acc_offset_rows", groupFirstAcc);
        op->setAttr("acc_offset_rows", accAttr);
      } else if (isa<gemmini::GemminiLowFenceOp>(op)) {
        inGroup = false;
      } else if (auto mvout = dyn_cast<gemmini::GemminiLowMvoutOp>(op)) {
        // The final store reads from the accumulator row assigned to its group.
        op->setAttr("spad_offset_rows",
            IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                groupFirstAcc));
      }
    });

    if (nextSpadA >= spadRows || nextSpadB >= spadRows)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createStaticScratchpadAllocationPass() {
  return std::make_unique<StaticScratchpadAllocationPass>();
}

} // namespace onnx_mlir
