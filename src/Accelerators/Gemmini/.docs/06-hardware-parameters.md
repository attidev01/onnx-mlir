# Hardware Parameters — Complete Reference

Two files define the physical constants of the Gemmini chip:

| File | Purpose |
|------|---------|
| `Runtime/gemmini_params.hpp` | Raw chip constants (used by runtime C code) |
| `Support/GemminiTargetInfo.hpp` | Compiler-visible wrapper (used by compiler passes) |

---

## Why Two Files?

`gemmini_params.hpp` is written in C-style (macros, typedefs) so it can
be included in the RISC-V runtime library compiled with a C compiler.

`GemminiTargetInfo.hpp` wraps the same numbers in a modern C++ struct with
helper methods, so the compiler passes can use them cleanly.

Both files describe **the same physical chip**. They must stay in sync.

---

## `gemmini_params.hpp` — Full Reference

### RoCC Instruction Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `XCUSTOM_ACC` | `3` | RISC-V custom function opcode for Gemmini. All Gemmini instructions use `CUSTOM_3`. |

**Simple analogy:** Just as ARM has opcode `0x0F` for SVC, RISC-V has
opcode `CUSTOM_3` for Gemmini. Every mvin/mvout/matmul encodes this value.

---

### Systolic Array Dimensions

| Constant | Value | Meaning |
|----------|-------|---------|
| `DIM` | `16` | The systolic array is a `16 × 16` grid. All tiles must be `16 × 16`. |
| `ADDR_LEN` | `32` | Internal address bus is 32 bits wide. |

**Simple example:**
```
A systolic array with DIM=16 multiplies a [16×16] matrix by a [16×16] matrix
in a single hardware pass. A [64×64] matrix requires 4×4×4 = 64 passes.
```

---

### Scratchpad Memory

| Constant | Value | Meaning |
|----------|-------|---------|
| `BANK_NUM` | `4` | Scratchpad has 4 independent banks. |
| `BANK_ROWS` | `4096` | Each bank holds 4096 rows. |
| Total rows | `4 × 4096 = 16384` | Total scratchpad capacity. |
| `MAX_BYTES` | `64` | Maximum bytes per single DMA transfer (one mvin/mvout transaction). |

**What is the scratchpad?**

The scratchpad is fast on-chip SRAM. It sits between DRAM (main memory) and
the systolic array. Loading data from DRAM is slow; operating on data in the
scratchpad is fast.

```
DRAM (slow, large) → [mvin] → Scratchpad (fast, small) → Systolic Array
Systolic Array → [mvout] → DRAM
```

**Capacity calculation:**
```
16384 rows × 16 values per row × 1 byte per value (i8) = 256 KB
```

---

### Accumulator Memory

| Constant | Value | Meaning |
|----------|-------|---------|
| `ACC_ROWS` | `1024` | Accumulator holds 1024 rows. |

**What is the accumulator?**

The accumulator is separate from the scratchpad. It stores partial sums
in `int32` format during matrix multiplication. At the end of a tiled
matmul, the final `int32` result is read back from the accumulator.

```
Accumulator capacity: 1024 rows × 16 values × 4 bytes = 64 KB
```

---

### Data Types

| Typedef | Underlying type | Range | Meaning |
|---------|----------------|-------|---------|
| `elem_t` | `int8_t` | `[-128, 127]` | Systolic array input element type. |
| `acc_t` | `int32_t` | `[-2^31, 2^31-1]` | Accumulator element type. |
| `full_t` | `int64_t` | `[-2^63, 2^63-1]` | Full-precision type for intermediate results. |
| `scale_t` | `float` | IEEE-754 f32 | Quantization scale factor. |
| `scale_t_bits` | `uint32_t` | — | `scale_t` reinterpreted as bits (for passing through integer ABI). |
| `acc_scale_t` | `float` | IEEE-754 f32 | Accumulator output scale. |
| `acc_scale_t_bits` | `uint32_t` | — | `acc_scale_t` reinterpreted as bits. |

**Why "bits" types?**

Some function signatures pass floats as integer bit patterns to avoid
platform-specific float-in-register calling convention issues:

```c
// Store the bits of 1.0f into an int64
int64_t alphaBits = *(int64_t*)&alpha;
// Function receives bits and reconstructs:
float alpha = *(float*)&alphaBits;
```

---

### Quantization Macros

| Macro | Purpose |
|-------|---------|
| `ROUNDING_RIGHT_SHIFT(x, shift)` | Right-shift `x` by `shift` bits with rounding. Used in fixed-point accumulator rescaling. |
| `MVIN_SCALE(elem, scale)` | Apply quantization scale to an input element during mvin. |
| `ACC_SCALE(elem, scale)` | Apply quantization scale to an accumulator element during mvout. |

**Simple example of rounding right shift:**
```
x = 7 (binary 0111), shift = 2
Without rounding: 7 >> 2 = 1  (0.75 lost)
With rounding:    (7 + 2) >> 2 = 2  (rounds 1.75 → 2)
```

---

### Memory Alignment Macros

| Macro | Purpose |
|-------|---------|
| `row_align(n)` | Attribute to align an array to the scratchpad row width. |
| `row_align_acc(n)` | Attribute to align to the accumulator row width. |

Alignment allows the DMA engine to transfer data in wide bursts,
improving bandwidth.

---

## `GemminiTargetInfo.hpp` — Full Reference

This struct wraps the same constants in a C++ interface for use in
the compiler passes.

```cpp
struct GemminiTargetInfo {
  static constexpr int64_t dim       = 16;
  static constexpr int64_t bankNum   = 4;
  static constexpr int64_t bankRows  = 4096;
  static constexpr int64_t accRows   = 1024;
  static constexpr int64_t elemBits  = 8;
  static constexpr int64_t accBits   = 32;
  ...
};
```

### Constants

| Field | Value | Where used in compiler |
|-------|-------|----------------------|
| `dim` | `16` | Tiling pass: loop step, tile dimension |
| `bankNum` | `4` | Capacity checks |
| `bankRows` | `4096` | Capacity checks |
| `accRows` | `1024` | Accumulator capacity checks |
| `elemBits` | `8` | Type size calculations |
| `accBits` | `32` | Accumulator type size |

### Helper Methods

#### `scratchpadRows()` → `16384`

Total scratchpad rows = `bankNum × bankRows`.

Used in the `StaticScratchpadAllocation` pass to check that all allocated
row ranges fit within the physical scratchpad.

#### `scratchpadMatrices()` → `1024`

How many `dim × dim` tiles fit in the scratchpad:

```
16384 rows / 16 rows per tile = 1024 tiles
```

#### `accumulatorMatrices()` → `64`

How many tiles fit in the accumulator:

```
1024 rows / 16 rows per tile = 64 tiles
```

#### `ceilDiv(value, divisor)` → `⌈value / divisor⌉`

Integer ceiling division. Used when computing how many tiles are needed:

```
ceilDiv(20, 16) = 2  → need 2 tiles for 20 rows
ceilDiv(16, 16) = 1  → need 1 tile for 16 rows
ceilDiv(1, 16)  = 1  → even 1 row needs a full tile
```

#### `getMatmulTileCounts(m, n, k)` → `{mTiles, nTiles, kTiles}`

Returns a struct with the number of tile loop iterations needed for a matmul
of shape `[m × k] × [k × n] → [m × n]`.

**Example:**
```cpp
auto tiles = GemminiTargetInfo::getMatmulTileCounts(64, 32, 48);
// tiles.mTiles = ceilDiv(64, 16) = 4
// tiles.nTiles = ceilDiv(32, 16) = 2
// tiles.kTiles = ceilDiv(48, 16) = 3
// Total hardware matmul calls = 4 × 2 × 3 = 24
```

---

## `gemmini.hpp` — Hardware Abstraction Layer

This file provides the C-level interface to issue actual RoCC instructions.
It defines the high-level `tiled_matmul_auto()` and related helpers that
the runtime uses internally.

### Key Type: `gemmini_state_t`

A 100+ field struct that tracks the complete hardware state machine.
The runtime updates fields in this struct before issuing instructions,
so it does not need to reconfigure the hardware for every operation.

Selected important fields:

| Field | Type | Meaning |
|-------|------|---------|
| `output_sp_addr` | `uint32_t` | Scratchpad address for the output tile |
| `preload_sp_addr` | `uint32_t` | Scratchpad address for the B (weight) tile |
| `preload_cols`, `preload_rows` | `size_t` | Tile dimensions for preload op |
| `output_cols`, `output_rows` | `size_t` | Tile dimensions for output op |
| `load_strides[3]` | `size_t` | DRAM strides for up to 3 mvin channels |
| `store_stride` | `size_t` | DRAM stride for mvout |
| `a_transpose`, `b_transpose` | `bool` | Whether to transpose A or B |
| `pool_stride` | `uint8_t` | Stride for integrated pooling after matmul |
| `pool_size` | `uint8_t` | Kernel size for integrated pooling |
| `counter_val[8]` | `uint64_t` | Performance counter values |
| `counter_config[8]` | `uint64_t` | Performance counter event selectors |

### Key Enums

#### `Dataflow`

```c
enum Dataflow {
  OS = 1,   // Output Stationary: partial sums stay in array
  WS = 2,   // Weight Stationary: weights stay in array
};
```

#### `Activation`

```c
enum Activation {
  NONE      = 0,   // No post-compute activation
  RELU      = 1,   // Apply ReLU after accumulation
  LAYERNORM = 2,   // Apply LayerNorm (transformer use)
  IGELU     = 3,   // Integer GeLU approximation
  SOFTMAX   = 4,   // Softmax over output
};
```

#### `NormCmd`

Commands for the hardware normalization unit (used in LayerNorm/IGELU):

```c
enum NormCmd {
  RESET      = 0,   // Reset accumulator
  SUM        = 1,   // Accumulate sum
  MEAN       = 2,   // Compute mean from sum
  VARIANCE   = 3,   // Compute variance
  INV_STDDEV = 4,   // Compute 1/stddev
  MAX        = 5,   // Find max (for softmax stability)
  SUM_EXP    = 6,   // Sum of exp(x - max)
  INV_SUM_EXP = 7,  // 1 / sum_exp (final softmax divisor)
};
```

---

## Memory Layout Summary

```
On-chip memory map (logical):

Row 0
Row 1
...                    ← Scratchpad (SRAM, 16384 rows, 256 KB total)
Row 16383

Row 0 (acc)
Row 1 (acc)
...                    ← Accumulator (SRAM, 1024 rows × 4 bytes, 64 KB total)
Row 1023 (acc)
```

The compiler passes encode scratchpad addresses as **row numbers** (0–16383)
and accumulator addresses as row numbers with a special high bit set (to
distinguish them from scratchpad rows).
