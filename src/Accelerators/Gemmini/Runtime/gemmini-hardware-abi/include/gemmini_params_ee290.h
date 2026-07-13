#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

// Alternate Gemmini hardware profile used by EE290 examples.
// This header is not included by the ONNX-MLIR runtime by default; it is kept
// in the ABI bundle so developers can compare or switch hardware profiles.
// Values here must match the target RTL if this profile is selected.

#include <stdint.h>
#include <limits.h>

// Systolic-array and memory-system dimensions for this profile.
#define DIM 32
#define ADDR_LEN 32
#define BANK_NUM 4
#define BANK_ROWS 2048
#define ACC_ROWS 512
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
