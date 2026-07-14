/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- GemminiTiling.hpp - Gemmini Transform Hooks ----------===//
//
// Declarations for Gemmini transform passes.
//
// The current tiling pass expands high-level Gemmini matmul ops into a tiled
// instruction sequence and now handles edge tiles by padding partial tiles up
// to the hardware `DIM x DIM` shape before lowering.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_TILING_H
#define ONNX_MLIR_GEMMINI_TILING_H

#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

// Create the Gemmini tiling/scheduling pass for high-level matmul ops.
std::unique_ptr<mlir::Pass> createGemminiTilingPass();

} // namespace onnx_mlir

#endif
