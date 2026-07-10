/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ONNXToGemmini.hpp - ONNX to Gemmini Conversion ---------===//
//
// Entry points for ONNX-to-Gemmini conversion support.
//
// The current implementation lowers a narrow direct-matmul slice into the
// high-level Gemmini dialect and leaves broader cases on the generic path.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_TO_GEMMINI_H
#define ONNX_MLIR_ONNX_TO_GEMMINI_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace onnx_mlir {

// Create the ONNX-to-Gemmini conversion pass for the currently supported direct
// Gemmini matmul slice.
std::unique_ptr<mlir::Pass> createONNXToGemminiPass();

void populateONNXToKrnlForGemmini(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx);

} // namespace onnx_mlir

#endif
