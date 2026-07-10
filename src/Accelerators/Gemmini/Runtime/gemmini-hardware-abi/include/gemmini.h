// See LICENSE for license details.
//
// Compatibility wrapper for ONNX-MLIR's vendored Gemmini hardware ABI.
//
// The Gemmini runtime and helper headers include "gemmini.h" through either
// GEMMINI_ROCC_INCLUDE_DIR or the Gemmini ABI root.  Keeping this forwarding
// header in the bundled ABI tree gives IDEs and compile_commands consumers a
// concrete local include target, while the full API implementation remains the
// upstream Gemmini header copied into the repository under gemmini/lib.

#ifndef ONNX_MLIR_GEMMINI_HARDWARE_ABI_GEMMINI_WRAPPER_H
#define ONNX_MLIR_GEMMINI_HARDWARE_ABI_GEMMINI_WRAPPER_H

#if __has_include("../../../../../../gemmini/lib/gemmini-rocc-tests/include/gemmini.h")
#include "../../../../../../gemmini/lib/gemmini-rocc-tests/include/gemmini.h"
#else
#error "Missing Gemmini API header. Run scripts/setup_gemmini_simulator.sh or set GEMMINI_ROCC_INCLUDE_DIR to a Gemmini rocc-tests include directory."
#endif

#endif // ONNX_MLIR_GEMMINI_HARDWARE_ABI_GEMMINI_WRAPPER_H
