/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ GemminiTargetInfo.hpp ----------------------===//
//
// Centralized target-parameter helpers for the Gemmini backend.
//
// The current backend previously duplicated or documented stale target values
// in several files. This support helper gives the refactored pipeline a single
// source of truth for the checked-in Gemmini profile used by this repository.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_GEMMINI_TARGET_INFO_H
#define ONNX_MLIR_GEMMINI_TARGET_INFO_H

#include <cstdint>

namespace onnx_mlir {
namespace gemmini {

struct GemminiTargetInfo {
  static constexpr int64_t dim = 16;
  static constexpr int64_t bankNum = 4;
  static constexpr int64_t bankRows = 4096;  // BANK_ROWS in gemmini_params.h
  static constexpr int64_t accRows  = 1024;  // ACC_ROWS  in gemmini_params.h
  static constexpr int64_t elemBits = 8;
  static constexpr int64_t accBits = 32;

  static constexpr int64_t scratchpadRows() { return bankNum * bankRows; }
  static constexpr int64_t scratchpadMatrices() {
    return scratchpadRows() / dim;
  }
  static constexpr int64_t accumulatorMatrices() { return accRows / dim; }

  static constexpr int64_t ceilDiv(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
  }

  struct MatmulTileCounts {
    int64_t mTiles;
    int64_t nTiles;
    int64_t kTiles;
  };

  static constexpr MatmulTileCounts getMatmulTileCounts(
      int64_t m, int64_t n, int64_t k) {
    return {ceilDiv(m, dim), ceilDiv(n, dim), ceilDiv(k, dim)};
  }
};

} // namespace gemmini
} // namespace onnx_mlir

#endif
