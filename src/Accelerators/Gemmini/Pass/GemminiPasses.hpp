/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- GemminiPasses.hpp - Gemmini Pass Definitions ---------===//
//
// Entry points for Gemmini-specific compiler passes.
//
// This header mirrors the role of NNPA/Pass/NNPAPasses.hpp and centralizes the
// pass factories used by the Gemmini lowering pipeline.
//
// Expected staged lowering ladder:
//   - ONNX -> Gemmini
//   - Gemmini -> GemminiLow
//   - GemminiLow -> LLVM
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_PASSES_H
#define ONNX_MLIR_GEMMINI_PASSES_H

#include "mlir/Pass/Pass.h"
#include "src/Accelerators/Gemmini/Compiler/GemminiCompilerOptions.hpp"

namespace onnx_mlir {

// Gemmini pass factories.
std::unique_ptr<mlir::Pass> createONNXToGemminiPass();
std::unique_ptr<mlir::Pass> createGemminiTilingPass();
std::unique_ptr<mlir::Pass> createGemminiToGemminiLowPass();
std::unique_ptr<mlir::Pass> createStaticScratchpadAllocationPass();
std::unique_ptr<mlir::Pass> createGemminiLowRewritePass();
std::unique_ptr<mlir::Pass> createGemminiLowToLLVMPass();

} // namespace onnx_mlir

#endif
