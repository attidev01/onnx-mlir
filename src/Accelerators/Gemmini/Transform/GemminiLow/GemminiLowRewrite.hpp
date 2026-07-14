/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- GemminiLowRewrite.hpp - GemminiLow Rewrite Pass ----------===//
//
// Factory declaration for the GemminiLow rewrite pass that canonicalises
// adjacent mvin/mvout sequences and folds redundant fence operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_LOW_REWRITE_H
#define ONNX_MLIR_GEMMINI_LOW_REWRITE_H

#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

// Create the pass that removes redundant adjacent GemminiLow commands.
std::unique_ptr<mlir::Pass> createGemminiLowRewritePass();

} // namespace onnx_mlir

#endif
