/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ GemminiAccelerator.cpp ----------------------===//
//
// Implementation of the Gemmini accelerator integration layer.
//
// This file wires the staged Gemmini pipeline into the accelerator framework.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/Gemmini/GemminiAccelerator.hpp"
#include "src/Accelerators/Gemmini/Compiler/GemminiCompilerUtils.hpp"
#include "src/Accelerators/Gemmini/Conversion/GemminiLowToLLVM/GemminiLowToLLVM.hpp"
#include "src/Accelerators/Gemmini/Conversion/GemminiToGemminiLow/GemminiToGemminiLow.hpp"
#include "src/Accelerators/Gemmini/Conversion/ONNXToGemmini/ONNXToGemmini.hpp"
#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiOps.hpp"
#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.hpp"
#include "src/Accelerators/Gemmini/Pass/GemminiPasses.hpp"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gemmini {

// Singleton storage for the backend instance returned by createGemmini().
GemminiAccelerator *GemminiAccelerator::instance = nullptr;

GemminiAccelerator *GemminiAccelerator::getInstance() {
  // Create the accelerator lazily the first time ONNX-MLIR asks for it.
  if (instance == nullptr)
    instance = new GemminiAccelerator();
  return instance;
}

GemminiAccelerator::GemminiAccelerator()
    : onnx_mlir::accel::Accelerator(onnx_mlir::accel::Accelerator::Kind::Gemmini) {
  // Register this backend in the list of available accelerator targets.
  // This mirrors the pattern used by the other accelerator backends.
  acceleratorTargets.push_back(this);

#ifdef ONNX_MLIR_WITH_GEMMINI_RUNTIME
  // When the RISC-V Gemmini runtime shim is available, make generated models
  // link against it just like other accelerator backends do for their runtime.
  addCompilerConfig(onnx_mlir::CCM_SHARED_LIB_DEPS, {"RuntimeGemmini"}, true);
#endif
}

void GemminiAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
  registry.insert<onnx_mlir::gemmini::GemminiDialect>();
  registry.insert<onnx_mlir::gemmini::GemminiLowDialect>();
}

void GemminiAccelerator::registerPasses(int optLevel) const {
  (void)optLevel;

  // Register the current Gemmini placeholder passes so they are available as
  // explicit command-line passes in onnx-mlir-opt.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createONNXToGemminiPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createGemminiTilingPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createGemminiToGemminiLowPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createStaticScratchpadAllocationPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createGemminiLowRewritePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createGemminiLowToLLVMPass();
  });
}

void GemminiAccelerator::configurePasses() const {
  onnx_mlir::configurePassesGemmini();
}

void GemminiAccelerator::conversionTargetONNXToKrnl(
    mlir::ConversionTarget &target) const {
  target.addLegalDialect<onnx_mlir::gemmini::GemminiDialect>();
  target.addLegalDialect<onnx_mlir::gemmini::GemminiLowDialect>();
}

void GemminiAccelerator::rewritePatternONNXToKrnl(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx) const {
  onnx_mlir::populateONNXToKrnlForGemmini(patterns, typeConverter, ctx);
}

void GemminiAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
                                   mlir::PassManager &pm,
                                   onnx_mlir::EmissionTargetType &emissionTarget,
                                   std::string outputNameNoExt) const {
  onnx_mlir::addPassesGemmini(module, pm, emissionTarget, outputNameNoExt);
}

void GemminiAccelerator::conversionTargetKrnlToLLVM(
    mlir::ConversionTarget &target) const {
  target.addLegalDialect<onnx_mlir::gemmini::GemminiLowDialect>();
}

void GemminiAccelerator::rewritePatternKrnlToLLVM(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    mlir::MLIRContext *ctx) const {
  (void)patterns;
  (void)typeConverter;
  (void)ctx;
}

} // namespace gemmini
} // namespace mlir

namespace onnx_mlir {
namespace accel {

// Entry point used by the accelerator registration logic generated from
// Accelerators.inc. Keep C++ linkage here to match the declarations emitted by
// src/Accelerators/Accelerator.hpp.
Accelerator *createGemmini() {
  return mlir::gemmini::GemminiAccelerator::getInstance();
}

} // namespace accel
} // namespace onnx_mlir
