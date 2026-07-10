/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace gemmini {

void GemminiDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiOps.cpp.inc"
      >();
}

void GemminiConfigOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(
      MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
}

void GemminiMvinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
      SideEffects::DefaultResource::get());
  // Write models the DMA into the Gemmini scratchpad (hardware-external state).
  // Without this, MLIR's DCE removes the op because it has no result uses and
  // no visible Write effect — treating it as trivially dead.
  effects.emplace_back(MemoryEffects::Write::get(),
      SideEffects::DefaultResource::get());
}

void GemminiMvoutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable(),
      SideEffects::DefaultResource::get());
}

void GemminiFenceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(
      MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
}

void GemminiMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getOutMutable(),
      SideEffects::DefaultResource::get());
}

} // namespace gemmini
} // namespace onnx_mlir

#define GET_OP_CLASSES
#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiOps.cpp.inc"

#include "src/Accelerators/Gemmini/Dialect/Gemmini/GemminiDialect.cpp.inc"
