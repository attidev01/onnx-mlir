/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- StaticScratchpadAllocation.hpp - Scratchpad Alloc Pass ---===//
//
// Factory declaration for the static scratchpad allocation pass that assigns
// fixed scratchpad row offsets to each GemminiLow mvin/mvout tile operation.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_STATIC_SCRATCHPAD_ALLOCATION_H
#define ONNX_MLIR_GEMMINI_STATIC_SCRATCHPAD_ALLOCATION_H

#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createStaticScratchpadAllocationPass();

} // namespace onnx_mlir

#endif
