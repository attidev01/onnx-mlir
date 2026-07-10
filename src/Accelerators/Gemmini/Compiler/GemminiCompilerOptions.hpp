/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- GemminiCompilerOptions.hpp --------------------===//
//
// Gemmini-specific compiler-option extensions.
//
// This file follows the same purpose as NNPA's Compiler option files:
// it extends shared ONNX-MLIR option enums with backend-specific entries for
// instrumentation, IR dumping, and optimization reporting.
//
// These macros are consumed by the central compiler option definitions rather
// than instantiated directly in this file.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_COMPILER_OPTIONS_H
#define ONNX_MLIR_GEMMINI_COMPILER_OPTIONS_H

#include "llvm/Support/CommandLine.h"

// clang-format off

// Instrumentation stages exposed specifically for Gemmini execution events.
#define INSTRUMENTSTAGE_ENUM_Gemmini                                            \
    ,                                                                          \
    Gemmini,                                                                   \
    GemminiLoad,                                                               \
    GemminiStore,                                                              \
    GemminiCompute,                                                            \
    GemminiFlush

// Command-line spellings and help text for the Gemmini instrumentation stages.
#define INSTRUMENTSTAGE_CL_ENUM_Gemmini                                        \
  clEnumVal(Gemmini, "Profile Gemmini-targeted IR or execution stages."),     \
  clEnumVal(GemminiLoad, "Profile Gemmini load operations."),                 \
  clEnumVal(GemminiStore, "Profile Gemmini store operations."),               \
  clEnumVal(GemminiCompute, "Profile Gemmini compute operations."),           \
  clEnumVal(GemminiFlush, "Profile Gemmini flush operations.")

// Additional IR-dump category for Gemmini-specific IR once that IR exists.
#define PROFILEIR_CL_ENUM_Gemmini                                              \
  , clEnumVal(Gemmini, "Profile operations in GemminiIR.")

// Optimization-report category for missed Gemmini acceleration opportunities.
#define OPTREPORT_ENUM_Gemmini , GemminiUnsupportedOps

#define OPTREPORT_CL_ENUM_Gemmini                                              \
  , clEnumVal(GemminiUnsupportedOps,                                           \
        "Provide report on why ONNX ops did not run on Gemmini.")

namespace onnx_mlir {
// Backend-specific emission targets. This enum is intentionally small for now
// because the Gemmini pipeline is not fully implemented in this tree.
typedef enum {
  EmitGemminiNone,
  EmitGemminiIR,
} GemminiEmissionTargetType;

} // namespace onnx_mlir

#endif // ONNX_MLIR_GEMMINI_COMPILER_OPTIONS_H
