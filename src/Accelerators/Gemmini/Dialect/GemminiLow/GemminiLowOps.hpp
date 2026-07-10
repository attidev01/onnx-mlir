/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- GemminiLowOps.hpp -----------------------------===//
//
// Include wrapper for the TableGen-generated GemminiLow dialect op
// declarations. Pulls in the dialect and op class headers emitted by
// mlir-tblgen from GemminiLowOps.td.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_LOW_OPS_H
#define ONNX_MLIR_GEMMINI_LOW_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.hpp.inc"

#endif
