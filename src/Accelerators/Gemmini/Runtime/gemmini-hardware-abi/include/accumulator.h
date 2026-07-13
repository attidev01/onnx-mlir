// See LICENSE for license details.

#ifndef SRC_MAIN_C_ACCUMULATOR_H
#define SRC_MAIN_C_ACCUMULATOR_H

// Minimal RoCC wrapper for the standalone accumulator example accelerator.
// Gemmini itself uses the command definitions in gemmini.h; this header is
// kept with the ABI bundle because upstream Gemmini tests may include it.

#include "rocc-software/src/xcustom.h"

// Function IDs understood by the accumulator RoCC device.
#define k_DO_WRITE 0
#define k_DO_READ 1
#define k_DO_LOAD 2
#define k_DO_ACCUM 3

// RoCC custom slot used by this standalone accumulator device.
#define XCUSTOM_ACC 0

// Write a value to the accelerator state.
#define doWrite(y, rocc_rd, data)                                       \
  ROCC_INSTRUCTION(XCUSTOM_ACC, y, data, rocc_rd, k_DO_WRITE);
// Read a value from the accelerator state.
#define doRead(y, rocc_rd)                                              \
  ROCC_INSTRUCTION(XCUSTOM_ACC, y, 0, rocc_rd, k_DO_READ);
// Load data from memory through the accelerator.
#define doLoad(y, rocc_rd, mem_addr)                                    \
  ROCC_INSTRUCTION(XCUSTOM_ACC, y, mem_addr, rocc_rd, k_DO_LOAD);
// Accumulate one value into the accelerator state.
#define doAccum(y, rocc_rd, data) \
  ROCC_INSTRUCTION(XCUSTOM_ACC, y, data, rocc_rd, k_DO_ACCUM);

#endif  // SRC_MAIN_C_ACCUMULATOR_H
