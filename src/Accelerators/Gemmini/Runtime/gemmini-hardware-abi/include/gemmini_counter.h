// See LICENSE for license details.

#ifndef COUNTER_H_
#define COUNTER_H_

// Numeric selectors for Gemmini's hardware performance counters.
// gemmini.h writes these values to the counter configuration register via the
// k_COUNTER RoCC command, then reads back snapshots for profiling.

#define DISABLE 0

// Counter IDs up to this value are incremental cycle/event counters.
#define INCREMENTAL_COUNTERS 44

// Top-level controller activity counters.

#define MAIN_LD_CYCLES 1
#define MAIN_ST_CYCLES 2
#define MAIN_EX_CYCLES 3
#define MAIN_LD_ST_CYCLES 4
#define MAIN_LD_EX_CYCLES 5
#define MAIN_ST_EX_CYCLES 6
#define MAIN_LD_ST_EX_CYCLES 7

// Load-controller counters.
#define LOAD_DMA_WAIT_CYCLE 8
#define LOAD_ACTIVE_CYCLE 9
#define LOAD_SCRATCHPAD_WAIT_CYCLE 10

// Store-controller counters.
#define STORE_DMA_WAIT_CYCLE 11
#define STORE_ACTIVE_CYCLE 12
#define STORE_POOLING_CYCLE 13
#define STORE_SCRATCHPAD_WAIT_CYCLE 14

// DMA translation lookaside buffer counters.
#define DMA_TLB_MISS_CYCLE 15
#define DMA_TLB_HIT_REQ 16
#define DMA_TLB_TOTAL_REQ 17

// Read-DMA and write-DMA counters.
#define RDMA_ACTIVE_CYCLE 18
#define RDMA_TLB_WAIT_CYCLES 19
#define RDMA_TL_WAIT_CYCLES 20

#define WDMA_ACTIVE_CYCLE 21
#define WDMA_TLB_WAIT_CYCLES 22
#define WDMA_TL_WAIT_CYCLES 23

// Execute-controller counters.
#define EXE_ACTIVE_CYCLE 24
#define EXE_FLUSH_CYCLE 25
#define EXE_CONTROL_Q_BLOCK_CYCLE 26
#define EXE_PRELOAD_HAZ_CYCLE 27
#define EXE_OVERLAP_HAZ_CYCLE 28

// Scratchpad and accumulator bank wait counters.
#define SCRATCHPAD_A_WAIT_CYCLE 29
#define SCRATCHPAD_B_WAIT_CYCLE 30
#define SCRATCHPAD_D_WAIT_CYCLE 31

#define ACC_A_WAIT_CYCLE 32
#define ACC_B_WAIT_CYCLE 33
#define ACC_D_WAIT_CYCLE 34

#define A_GARBAGE_CYCLES 35
#define B_GARBAGE_CYCLES 36
#define D_GARBAGE_CYCLES 37

// Im2col/transposer and loop-controller activity counters.
#define IM2COL_MEM_CYCLES 38
#define IM2COL_ACTIVE_CYCLES 39
#define IM2COL_TRANSPOSER_WAIT_CYCLE 40

#define RESERVATION_STATION_FULL_CYCLES 41
#define RESERVATION_STATION_ACTIVE_CYCLES 42

#define LOOP_MATMUL_ACTIVE_CYCLES 43
#define TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES 44

// Non-incremental counters start after INCREMENTAL_COUNTERS.
#define RESERVATION_STATION_LD_COUNT (INCREMENTAL_COUNTERS + 1)
#define RESERVATION_STATION_ST_COUNT (INCREMENTAL_COUNTERS + 2)
#define RESERVATION_STATION_EX_COUNT (INCREMENTAL_COUNTERS + 3)

#define RDMA_BYTES_REC (INCREMENTAL_COUNTERS + 4)
#define WDMA_BYTES_SENT (INCREMENTAL_COUNTERS + 5)

#define RDMA_TOTAL_LATENCY (INCREMENTAL_COUNTERS + 6)
#define WDMA_TOTAL_LATENCY (INCREMENTAL_COUNTERS + 7)

#endif
