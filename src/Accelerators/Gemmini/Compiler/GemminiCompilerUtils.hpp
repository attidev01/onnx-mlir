/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- GemminiCompilerUtils.hpp ---------------------===//
//
// Compiler utilities for the Gemmini accelerator pipeline.
//
// This file establishes the same role for Gemmini that NNPACompilerUtils.hpp
// plays for NNPA: a central place for backend-local pass sequencing and pass
// configuration helpers.
//
//===----------------------------------------------------------------------===//

/// @file GemminiCompilerUtils.hpp
/// @brief Pass-orchestration helpers for the Gemmini compiler backend.
///
/// These functions are the primary integration surface between
/// `GemminiAccelerator::addPasses` and the individual Gemmini pass objects.
/// Callers build the full staged pipeline by calling them in order:
///
/// @code
///   configurePassesGemmini();
///   addONNXToGemminiPasses(pm);
///   addGemminiToGemminiLowPasses(pm);
///   // optionally: pm.addPass(createGemminiLowToLLVMPass());
///   addPasses(module, pm, emissionTarget, outputName); // generic lowering
/// @endcode

#ifndef ONNX_MLIR_GEMMINI_COMPILER_UTILS_H
#define ONNX_MLIR_GEMMINI_COMPILER_UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"

namespace onnx_mlir {

/// @brief Initialise Gemmini pass configuration callbacks.
///
/// This is a stable hook for future tiling-aggressiveness or target-profile
/// options.  Currently a no-op.
void configurePassesGemmini();

/// @brief Append the ONNX → Gemmini dialect conversion passes to @p pm.
///
/// The added passes lower selected ONNX ops (MatMul, Conv, Gemm, …) to
/// `gemmini` dialect ops and run a canonicaliser pass to clean up.
/// @param pm Pass manager to populate.
void addONNXToGemminiPasses(mlir::PassManager &pm);

/// @brief Append the Gemmini → GemminiLow lowering passes to @p pm.
///
/// The added passes perform tiling, scratchpad allocation, GemminiLow
/// rewriting, and a final canonicalisation.
/// @param pm Pass manager to populate.
void addGemminiToGemminiLowPasses(mlir::PassManager &pm);

/// @brief Build the complete Gemmini compilation pipeline in @p pm.
///
/// Calls `configurePassesGemmini`, `addONNXToGemminiPasses`,
/// `addGemminiToGemminiLowPasses`, conditionally appends the
/// `GemminiLowToLLVM` pass when @p emissionTarget requires LLVM IR, and
/// finally delegates to the generic `onnx_mlir::addPasses` for the remainder
/// of the standard lowering ladder.
///
/// @param module          Module being compiled.
/// @param pm              Pass manager to populate.
/// @param emissionTarget  Controls which stages are appended (e.g.
///                        `EmitLLVMIR` triggers the LLVM-lowering stage).
/// @param outputNameNoExt Base output file path without extension.
void addPassesGemmini(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt);

} // namespace onnx_mlir

#endif
