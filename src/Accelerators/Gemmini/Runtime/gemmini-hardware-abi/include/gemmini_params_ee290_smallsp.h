#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

// Alternate EE290 Gemmini profile with a smaller scratchpad/accumulator.
// This is not the default ONNX-MLIR runtime profile; it is bundled for
// compatibility with upstream examples and experiments.

#include <stdint.h>
#include <limits.h>

// Systolic-array and memory-system dimensions for the small-SP profile.
#define DIM 32
#define ADDR_LEN 32
#define BANK_NUM 4
#define BANK_ROWS 1024
#define ACC_ROWS 256
#define MAX_BYTES 64
#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*1))
#define MAX_BLOCK_LEN_ACC 1

// Element and accumulator C types expected by this profile.
typedef int8_t elem_t;
elem_t elem_t_max = 127;
elem_t elem_t_min = -128;
typedef int32_t acc_t;
typedef int64_t full_t;

// Align static buffers to Gemmini row boundaries.
#define row_align(blocks) __attribute__((aligned(blocks*DIM*sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks*DIM*sizeof(acc_t))))

#endif // GEMMINI_PARAMS_H
