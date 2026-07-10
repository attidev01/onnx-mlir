/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- GemminiToGemminiLow.hpp - Gemmini→GemminiLow Pass --------===//
//
// Factory declaration for the pass that lowers scheduled Gemmini dialect ops
// into GemminiLow IR (direct scratchpad-addressed instruction sequences).
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_TO_GEMMINI_LOW_H
#define ONNX_MLIR_GEMMINI_TO_GEMMINI_LOW_H

#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createGemminiToGemminiLowPass();

} // namespace onnx_mlir

#endif
