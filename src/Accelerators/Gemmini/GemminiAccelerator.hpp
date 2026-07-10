/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ GemminiAccelerator.hpp ----------------------===//
//
// Accelerator support for the Gemmini systolic-array coprocessor.
//
// This file mirrors the role of NNPAAccelerator.hpp:
// it declares the singleton accelerator object that plugs Gemmini into the
// ONNX-MLIR accelerator framework.
//
// The implementation now supports a staged accelerator-local pipeline for the
// direct matmul path:
//   ONNX -> Gemmini -> GemminiLow -> LLVM.
//
//===----------------------------------------------------------------------===//

/// @file GemminiAccelerator.hpp
/// @brief ONNX-MLIR accelerator plugin for the UC Berkeley Gemmini
///        systolic-array coprocessor.
///
/// The Gemmini backend follows the same structural contract as the NNPA
/// accelerator: a singleton object registers with the ONNX-MLIR accelerator
/// framework and contributes dialect, pass, and pattern hooks at well-defined
/// pipeline stages.
///
/// Compilation pipeline (direct matmul path):
/// @code
///   ONNX -> Gemmini -> GemminiLow -> LLVM
/// @endcode
///
/// Runtime path (float operators):
///   f32/f16 inputs are quantised to int8, forwarded to `tiled_matmul_auto`,
///   and the i32 accumulator is dequantised back to f32/f16.

#ifndef ONNX_MLIR_GEMMINI_ACCELERATOR_H
#define ONNX_MLIR_GEMMINI_ACCELERATOR_H

#include "src/Accelerators/Accelerator.hpp"

namespace mlir {
namespace gemmini {

/// @brief Singleton accelerator plugin that connects Gemmini to ONNX-MLIR.
///
/// This class implements the `onnx_mlir::accel::Accelerator` interface and is
/// responsible for three backend-facing tasks:
///   -# Register the `gemmini` and `gemmini_low` MLIR dialects.
///   -# Contribute Gemmini-specific passes to the compilation pipeline.
///   -# Provide ONNX→Krnl and Krnl→LLVM pattern hooks.
///
/// A process-wide singleton is created by the generated `InitAccelerators`
/// registration entry point and is accessible via `getInstance()`.
class GemminiAccelerator : public onnx_mlir::accel::Accelerator {
public:
  /// @brief Construct the Gemmini accelerator and register it with the global
  ///        accelerator target list used by ONNX-MLIR.
  GemminiAccelerator();

  /// @brief Return the backend plugin version number.
  ///
  /// The value is encoded as `(major << 16) | (minor << 8) | patch`.
  /// @return Plugin version encoded as a 64-bit integer.
  uint64_t getVersionNumber() const override { return 0x030000; }

  /// @brief Register all Gemmini passes with the global pass registry.
  ///
  /// After this call `onnx-mlir-opt --help` lists Gemmini-specific passes.
  /// @param optLevel Optimisation level passed by the driver (currently unused
  ///                 but forwarded for future tiling-aggressiveness tuning).
  void registerPasses(int optLevel) const override;

  /// @brief Wire pass configuration callbacks (currently a no-op placeholder).
  void configurePasses() const override;

  /// @brief Tensor-to-MemRef type conversion hook.
  ///
  /// Returns `nullptr` to indicate that the Gemmini backend defers to the
  /// default ONNX-MLIR type-conversion policy.
  /// @param tensorType The tensor type to convert.
  /// @return `nullptr` (uses default conversion).
  mlir::MemRefType convertTensorTypeToMemRefType(
      const mlir::TensorType tensorType) const override {
    (void)tensorType;
    return nullptr;
  }

  /// @brief Mark Gemmini-handled ops as legal on the ONNX→Krnl target.
  /// @param target The conversion target to update.
  void conversionTargetONNXToKrnl(
      mlir::ConversionTarget &target) const override;

  /// @brief Register ONNX→Krnl rewrite patterns for Gemmini ops.
  /// @param patterns    Pattern set to populate.
  /// @param typeConverter Type converter provided by the host pass.
  /// @param ctx         MLIR context.
  void rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
      mlir::TypeConverter &typeConverter,
      mlir::MLIRContext *ctx) const override;

  /// @brief Mark Gemmini-handled ops as legal on the Krnl→LLVM target.
  /// @param target The conversion target to update.
  void conversionTargetKrnlToLLVM(
      mlir::ConversionTarget &target) const override;

  /// @brief Register Krnl→LLVM rewrite patterns for Gemmini ops.
  /// @param patterns       Pattern set to populate.
  /// @param typeConverter  LLVM type converter provided by the host pass.
  /// @param ctx            MLIR context.
  void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
      mlir::LLVMTypeConverter &typeConverter,
      mlir::MLIRContext *ctx) const override;

  /// @brief Register `gemmini` and `gemmini_low` dialects with the registry.
  ///
  /// Called by the MLIR context during initialisation so that the IR parser
  /// and pass infrastructure can resolve `gemmini.*` and `gemmini_low.*` ops.
  /// @param registry The dialect registry to update.
  void registerDialects(mlir::DialectRegistry &registry) const override;

  /// @brief Add all Gemmini-specific passes to the compilation pass manager.
  ///
  /// The pass sequence implements the staged pipeline:
  ///   ONNX → Gemmini → GemminiLow → (LLVM if emitting LLVM IR)
  /// followed by the generic ONNX-MLIR lowering passes.
  ///
  /// @param module          The module being compiled.
  /// @param pm              The pass manager to populate.
  /// @param emissionTarget  Requested output form; controls whether the
  ///                        LLVM-lowering stage is appended.
  /// @param outputNameNoExt Base output file name (extension added by driver).
  void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
      std::string outputNameNoExt) const override;

  /// @brief Return the process-wide singleton instance.
  ///
  /// Called by the C registration entry point in `GemminiAccelerator.cpp`.
  /// @return Pointer to the singleton; never null after construction.
  static GemminiAccelerator *getInstance();

private:
  /// Process-wide singleton instance owned for the lifetime of the program.
  static GemminiAccelerator *instance;
};

} // namespace gemmini
} // namespace mlir

#endif // ONNX_MLIR_GEMMINI_ACCELERATOR_H
