/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- GemminiLowOps.cpp -----------------------------===//
//
// Registers the GemminiLow dialect operations generated from GemminiLowOps.td
// and annotates their memory effects for MLIR analyses and canonicalization.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace gemmini {

void GemminiLowDialect::initialize() {
  // Operation classes are generated from GemminiLowOps.td by mlir-tblgen.
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.cpp.inc"
      >();
}

void GemminiLowConfigOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Configuration writes Gemmini hardware state and must not be removed as dead.
  effects.emplace_back(
      MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
}

void GemminiLowMvinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
      SideEffects::DefaultResource::get());
  // Write models the DMA into the Gemmini scratchpad (hardware-external state).
  // Without this, MLIR's DCE removes the op because it has no result uses and
  // no visible Write effect, so it would be treated as trivially dead.
  effects.emplace_back(MemoryEffects::Write::get(),
      SideEffects::DefaultResource::get());
}

void GemminiLowMvoutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Mvout writes the destination memref from a concrete scratchpad row.
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable(),
      SideEffects::DefaultResource::get());
}

void GemminiLowFenceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Fences serialize hardware commands and therefore carry an ordering effect.
  effects.emplace_back(
      MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
}

void GemminiLowMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Keep operand effects visible even though actual inputs are scratchpad rows.
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
#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowOps.cpp.inc"

#include "src/Accelerators/Gemmini/Dialect/GemminiLow/GemminiLowDialect.cpp.inc"
