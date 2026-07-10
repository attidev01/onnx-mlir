/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Accelerators/Gemmini/Support/GemminiTargetInfo.hpp"

// Keep a translation unit for the support library even though the initial
// target information is constexpr-only. Future target-profile discovery and
// command-line overrides can grow here without changing users of the helper.

namespace onnx_mlir {
namespace gemmini {} // namespace gemmini
} // namespace onnx_mlir
