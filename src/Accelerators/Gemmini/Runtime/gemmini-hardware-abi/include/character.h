// See LICENSE for license details.

#ifndef SRC_MAIN_C_CHARACTER_H
#define SRC_MAIN_C_CHARACTER_H

// RoCC custom-slot declaration for the standalone character-device example.
// This header is retained with the upstream ABI bundle for compatibility with
// low-level tests; ONNX-MLIR's Gemmini runtime does not call it directly.

#include "rocc-software/src/xcustom.h"

#define XCUSTOM_CHAR 2

#endif  // SRC_MAIN_C_CHARACTER_H
