/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- GemminiLowToLLVM.hpp - GemminiLow→LLVM Pass --------------===//
//
// Factory declaration for the pass that lowers GemminiLow dialect ops into
// LLVM IR (CUSTOM_3 inline-asm RoCC instruction sequences).
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_LOW_TO_LLVM_H
#define ONNX_MLIR_GEMMINI_LOW_TO_LLVM_H

#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

// Create the pass that lowers GemminiLow ops to LLVM/RoCC inline assembly.
std::unique_ptr<mlir::Pass> createGemminiLowToLLVMPass();

} // namespace onnx_mlir

#endif
