# Gemmini Dialects — Operations and Types

ONNX-MLIR uses MLIR's dialect system. A **dialect** is a named set of
operations, types, and attributes. The Gemmini backend defines two dialects:

| Dialect | C++ namespace | Purpose |
|---------|---------------|---------|
| `gemmini` | `onnx_mlir::gemmini` | High-level tiled operations |
| `gemmini_low` | `onnx_mlir::gemmini` | Low-level direct RoCC hardware ops |

---

## What is a Dialect?

Think of a dialect as a vocabulary layer:

```
ONNX dialect:        onnx.MatMul, onnx.Conv, ...
Gemmini dialect:     gemmini.matmul, gemmini.mvin, ...
GemminiLow dialect:  gemmini_low.matmul, gemmini_low.mvin, ...
LLVM dialect:        llvm.call, llvm.alloca, ...
```

Each pass lowers IR from a higher dialect to a lower one.

---

## The `gemmini` Dialect (High-Level)

**Defined in:**
- `Dialect/Gemmini/Gemmini.td` — dialect descriptor
- `Dialect/Gemmini/GemminiOps.td` — op definitions
- `Dialect/Gemmini/GemminiOps.hpp/cpp` — C++ classes

**Purpose:** Represent Gemmini operations at a level that the tiling pass
can still understand and manipulate. Operations here refer to `memref` values
(tensor memory regions), not raw hardware addresses.

---

### `gemmini.config`

**Role:** Send a configuration instruction to the systolic array.
Must be called before any compute to set the dataflow mode.

**Arguments:**
| Argument | Type | Meaning |
|----------|------|---------|
| `dataflow` | string attribute | `"OS"` (Output Stationary) or `"WS"` (Weight Stationary). See below. |
| `a_transpose` | optional bool | If true, treat A matrix as transposed. |
| `b_transpose` | optional bool | If true, treat B matrix as transposed. |

**What is OS vs WS?**

The systolic array can keep either the **output** or the **weights** stationary
in the grid. Choosing correctly improves data reuse.

```
Weight Stationary (WS):
  Load B (weights) into the array once.
  Stream many A (input) tiles through.
  Good for: inference with fixed weights (most neural networks).

Output Stationary (OS):
  Keep the partial sum C in the array.
  Stream both A and B through.
  Good for: when C is large and writing it often is expensive.
```

**Example:**
```mlir
gemmini.config {dataflow = "WS"}
```

---

### `gemmini.mvin` (Move In)

**Role:** Load a tile from main memory (DRAM) into the scratchpad (SRAM).

**Arguments:**
| Argument | Type | Meaning |
|----------|------|---------|
| `source` | memref | The DRAM tensor to read from. |
| `spad_offset_rows` | i64 | Which scratchpad row to write into (0 = first row). |
| `tile_rows` | i64 | Number of rows to transfer (≤ 16). |
| `tile_cols` | i64 | Number of columns to transfer (≤ 16). |

**Memory effects:** Reads `source` (DRAM), writes scratchpad (SRAM).

**Why we need it:**
The systolic array cannot read DRAM directly. Data must first be staged
into the scratchpad. This is called "double-buffering" when done in advance.

**Simple analogy:**
Moving food from a warehouse (DRAM) to a kitchen counter (scratchpad)
before the chef (systolic array) can use it.

**Example:**
```mlir
// Load a 16×16 tile of A into scratchpad rows 0–15
gemmini.mvin %A_tile, %c0_i64, %c16_i64, %c16_i64
```

---

### `gemmini.mvout` (Move Out)

**Role:** Store a computed tile from the accumulator or scratchpad back to
main memory (DRAM).

**Arguments:**
| Argument | Type | Meaning |
|----------|------|---------|
| `dest` | memref | The DRAM tensor to write into. |
| `spad_offset_rows` | i64 | Which scratchpad/accumulator row to read from. |
| `tile_rows` | i64 | Number of rows to transfer. |
| `tile_cols` | i64 | Number of columns to transfer. |

**Memory effects:** Reads scratchpad/accumulator, writes `dest` (DRAM).

**Example:**
```mlir
// Write the computed 16×16 result back to DRAM
gemmini.mvout %C_tile, %c0_i64, %c16_i64, %c16_i64
```

---

### `gemmini.matmul`

**Role:** Issue a matrix multiply using tiles already in the scratchpad.

**Arguments:**
| Argument | Type | Meaning |
|----------|------|---------|
| `lhs` | memref | Left-hand side matrix (A). |
| `rhs` | memref | Right-hand side matrix (B). |
| `out` | memref | Output matrix (C). |
| `mode` | string attribute | `"preload"` or `"accumulate"`. |
| `dataflow` | string attribute | `"OS"` or `"WS"`. |

**The `mode` attribute:**

```
"preload":     Start fresh. C = A × B.
"accumulate":  Add to existing. C += A × B.
```

Accumulate mode is used when K is tiled: each K-tile adds its partial
product to the same output tile.

**Memory effects:** Reads `lhs` and `rhs`, writes `out`.

**Example — full matmul with accumulation:**
```mlir
// First K-tile: start fresh
gemmini.matmul %A0, %B0, %C {mode = "preload", dataflow = "WS"}

// Second K-tile: accumulate
gemmini.matmul %A1, %B1, %C {mode = "accumulate", dataflow = "WS"}
```

---

## The `gemmini_low` Dialect (Low-Level)

**Defined in:**
- `Dialect/GemminiLow/GemminiLow.td`
- `Dialect/GemminiLow/GemminiLowOps.td`
- `Dialect/GemminiLow/GemminiLowOps.hpp/cpp`

**Purpose:** Represent Gemmini operations at the RoCC instruction level.
Each op in this dialect corresponds to a specific hardware instruction opcode.
Operations here use integer row addresses, not memref values.

The `gemmini_low` ops are the IR that Pass 6 (`GemminiLowToLLVM`) converts
directly to inline assembly.

---

### Mapping: High-Level → Low-Level

| `gemmini` op | `gemmini_low` op | Difference |
|--------------|-----------------|------------|
| `gemmini.config` | `gemmini_low.config` | Adds exact instruction bits for transpose flags |
| `gemmini.mvin` | `gemmini_low.mvin` | Uses concrete row numbers (from Pass 4) |
| `gemmini.mvout` | `gemmini_low.mvout` | Uses concrete row numbers |
| `gemmini.matmul` | `gemmini_low.matmul` | Uses concrete row numbers for pre-assigned scratchpad locations |

---

## How Dialects Are Registered

Both dialects are registered in `GemminiAccelerator.cpp`:

```cpp
void GemminiAccelerator::registerDialects(DialectRegistry &registry) {
  registry.insert<gemmini::GemminiDialect>();
  registry.insert<gemmini::GemminiLowDialect>();
}
```

After registration, MLIR's parser and printer know how to read and write
ops from these dialects in `.mlir` text files.

---

## Reading Gemmini Dialect IR in a File

When you dump MLIR with `--EmitMLIR`, you will see patterns like:

```mlir
func.func @main_graph(%arg0: memref<1x3x416x416xf32>) {

  // Allocate output buffer
  %out = memref.alloc() : memref<16x16xf32>

  // Move weight tile into scratchpad row 16
  gemmini.mvin %weights, %c16_i64, %c16_i64, %c16_i64

  // Move input tile into scratchpad row 0
  gemmini.mvin %input_tile, %c0_i64, %c16_i64, %c16_i64

  // Multiply (weight stationary, first K tile: preload)
  gemmini.matmul %input_tile, %weights, %out {mode="preload", dataflow="WS"}

  // Write result back to DRAM
  gemmini.mvout %out, %c0_i64, %c16_i64, %c16_i64

  return
}
```

Each `%cN_i64` is a compiler-generated integer constant (`N` is the value).
`%alloc_N` names are compiler-generated temporary buffers.

---

## TableGen: How Ops Are Defined

The `.td` files use TableGen syntax (a C++-like DSL) to define operations.
The MLIR build system auto-generates `.hpp` and `.cpp` boilerplate from them.

Simple example — a TableGen op definition:
```tablegen
def Gemmini_MvinOp : Gemmini_Op<"mvin"> {
  let summary = "Move tile from DRAM to scratchpad";

  let arguments = (ins
    AnyMemRef:$source,
    I64:$spad_offset_rows,
    I64:$tile_rows,
    I64:$tile_cols
  );

  let results = (outs);   // no return value

  let effects = [MemRead<DefaultResource, $source>,
                 MemWrite<DefaultResource>];
}
```

This is expanded by `mlir-tblgen` into the C++ `GemminiMvinOp` class
that compiler passes work with.
