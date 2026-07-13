// See LICENSE for license details.

#ifndef SRC_MAIN_C_TRANSLATOR_H
#define SRC_MAIN_C_TRANSLATOR_H

// Minimal RoCC wrapper for the standalone virtual-address translator helper.
// It is not called by ONNX-MLIR's Gemmini runtime, but is part of the bundled
// upstream ABI headers used by low-level Gemmini tests.

#include "rocc-software/src/xcustom.h"

// RoCC custom slot for the translator helper.
#define XCUSTOM_TRANS 1

// Ask the translator accelerator to translate vaddr and return the result in y.
#define doTranslate(y, vaddr)                                \
    ROCC_INSTRUCTION(XCUSTOM_TRANS, y, vaddr, 0, 0);

#endif  // SRC_MAIN_C_TRANSLATOR_H
