/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- GemminiCompilerUtils.cpp ---------------------===//
//
// Compiler utilities for the Gemmini accelerator pipeline.
//
// This file introduces the NNPA-style pass-orchestration layer for Gemmini.
// The actual Gemmini lowerings remain intentionally minimal in this phase, but
// the backend now has a stable place to express a staged pipeline:
//
//   ONNX -> Gemmini -> GemminiLow -> LLVM
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "src/Accelerators/Gemmini/Compiler/GemminiCompilerUtils.hpp"
#include "src/Accelerators/Gemmini/Pass/GemminiPasses.hpp"
#include "src/Compiler/CompilerPasses.hpp"

using namespace mlir;

namespace onnx_mlir {

void configurePassesGemmini() {
  // The initial NNPA-style Gemmini pipeline does not need dynamic pass-option
  // wiring yet. Keep this hook so future tiling, dataflow, or target-profile
  // options have a stable integration point.
}

void addONNXToGemminiPasses(PassManager &pm) {
  // First select accelerator-compatible ONNX patterns, then canonicalize so
  // later Gemmini passes see simpler IR and fewer dead helper operations.
  pm.addNestedPass<func::FuncOp>(createONNXToGemminiPass());
  pm.addPass(createCanonicalizerPass());
}

void addGemminiToGemminiLowPasses(PassManager &pm) {
  // The high-level Gemmini dialect is still schedule-oriented. These passes
  // make tile boundaries, scratchpad rows, and low-level instruction cleanup
  // explicit before the LLVM inline-assembly lowering stage.
  pm.addNestedPass<func::FuncOp>(createGemminiTilingPass());
  pm.addNestedPass<func::FuncOp>(createGemminiToGemminiLowPass());
  pm.addNestedPass<func::FuncOp>(createStaticScratchpadAllocationPass());
  pm.addNestedPass<func::FuncOp>(createGemminiLowRewritePass());
  pm.addPass(createCanonicalizerPass());
}

void addPassesGemmini(mlir::OwningOpRef<mlir::ModuleOp> &module,
    PassManager &pm, EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) {
  configurePassesGemmini();

  // Keep the regular ONNX-MLIR pipeline as the outer driver, but splice the
  // Gemmini-local stages in ahead of generic lowering so the backend grows
  // toward an NNPA-style accelerator-controlled pipeline.
  addONNXToGemminiPasses(pm);
  addGemminiToGemminiLowPasses(pm);
  if (emissionTarget >= EmitLLVMIR)
    pm.addPass(createGemminiLowToLLVMPass());
  onnx_mlir::addPasses(module, pm, emissionTarget, outputNameNoExt);
}

} // namespace onnx_mlir
