/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ GemminiOps.hpp ------------------------------===//
//
// Include wrapper for the TableGen-generated Gemmini dialect op declarations.
// Pulls in the dialect and op class headers emitted by mlir-tblgen from
// GemminiOps.td.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_OPS_H
#define ONNX_MLIR_GEMMINI_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiOps.hpp.inc"

#endif
