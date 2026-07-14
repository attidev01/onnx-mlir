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

// Select supported ONNX operations for Gemmini dialect/runtime lowering.
std::unique_ptr<mlir::Pass> createONNXToGemminiPass();
// Tile high-level Gemmini matmul operations into DIM-sized loop nests.
std::unique_ptr<mlir::Pass> createGemminiTilingPass();
// Lower high-level Gemmini operations to explicit GemminiLow operations.
std::unique_ptr<mlir::Pass> createGemminiToGemminiLowPass();
// Assign static scratchpad and accumulator offsets to GemminiLow operations.
std::unique_ptr<mlir::Pass> createStaticScratchpadAllocationPass();
// Remove redundant adjacent GemminiLow commands before LLVM lowering.
std::unique_ptr<mlir::Pass> createGemminiLowRewritePass();
// Emit LLVM dialect with RISC-V RoCC inline assembly for GemminiLow ops.
std::unique_ptr<mlir::Pass> createGemminiLowToLLVMPass();

} // namespace onnx_mlir

#endif
