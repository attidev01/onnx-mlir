# Gemmini Compiler Pipeline — Pass by Pass

The Gemmini backend compiles an ONNX model through **6 sequential passes**.
Each pass transforms the compiler's internal representation (MLIR IR) one
step closer to RISC-V machine code.

This document explains each pass in plain language with small examples.

---

## The Full Pipeline at a Glance

```
ONNX IR
  │
  ▼  Pass 1: ONNXToGemmini
Gemmini dialect (high-level tiled ops)
  │
  ▼  Pass 2: GemminiTiling
Gemmini dialect (explicit 16×16 tile loops)
  │
  ▼  Pass 3: GemminiToGemminiLow
GemminiLow dialect (RoCC instruction ops)
  │
  ▼  Pass 4: StaticScratchpadAllocation
GemminiLow dialect (concrete scratchpad row numbers)
  │
  ▼  Pass 5: GemminiLowRewrite
GemminiLow dialect (optimized / canonical form)
  │
  ▼  Pass 6: GemminiLowToLLVM
LLVM IR (CUSTOM_3 inline-assembly sequences)
  │
  ▼  (standard LLVM → RISC-V codegen)
RISC-V machine code
```

---

## Pass 1 — ONNXToGemmini

**File:** `Conversion/ONNXToGemmini/ONNXToGemmini.cpp`

**What it does:**
Scans the ONNX IR for matrix-multiply and convolution ops.
For each supported op it emits a `gemmini.*` high-level op.
Unsupported ops pass through unchanged and are later lowered by the
generic ONNX-MLIR Krnl path.

**Supported ONNX ops → Gemmini ops:**
| ONNX op | Condition | Gemmini replacement |
|---------|-----------|---------------------|
| `onnx.MatMul` | rank-2, static shape, `f32` | `gemmini.matmul` + runtime call to `om_gemmini_matmul_f32_hw` |
| `onnx.MatMulInteger` | `i8 × i8 → i32` | `gemmini.matmul` + runtime call to `om_gemmini_matmulinteger_i8i8acc32` |
| `onnx.Conv` | `f32` NCHW | runtime call to `om_gemmini_conv_f32_bias` |

**What "supported" means (simple rules):**
- Matrix must have exactly 2 dimensions.
- Dimensions must be known at compile time (no `?` shapes).
- Data type must be `f32` (float) or `i8` (quantized).

**Example — before and after this pass:**
```mlir
// Before (ONNX dialect):
%result = "onnx.MatMul"(%A, %B)

// After (Gemmini dialect):
%result = gemmini.matmul %A, %B, %out {mode = "preload", dataflow = "WS"}
```

---

## Pass 2 — GemminiTiling

**File:** `Transform/Gemmini/GemminiTiling.cpp`

**What it does:**
Takes a `gemmini.matmul` over a large matrix and rewrites it as
nested loops. Each loop body processes exactly one `16 × 16` tile
using `gemmini.mvin`, `gemmini.matmul`, and `gemmini.mvout` ops.

**Why it is needed:**
The Gemmini hardware can only compute a `16 × 16` block per step.
Larger matrices must be broken into tiles.

**Simple tiling example:**

```
Input:  A[32×32]  ×  B[32×32]  →  C[32×32]
Tile size: 16×16
Tiles needed: 4 along M, 4 along N, 4 along K → 64 hardware calls

Tile loop:
  for m_tile = 0, 1:          (32 rows / 16 = 2 tiles)
    for n_tile = 0, 1:        (32 cols / 16 = 2 tiles)
      for k_tile = 0, 1:
        gemmini.mvin A[m_tile*16 : +16, k_tile*16 : +16]
        gemmini.mvin B[k_tile*16 : +16, n_tile*16 : +16]
        gemmini.matmul (accumulate into C tile)
      gemmini.mvout C[m_tile*16 : +16, n_tile*16 : +16]
```

**Edge tile handling:**
When the matrix dimension is not a multiple of 16, the last tile is
smaller. The tiling pass uses `min(remaining, 16)` for the tile
dimensions, so no out-of-bounds access occurs.

Example: a `20 × 16` matrix produces:
- Tile 0: `min(16, 20) = 16` rows → full tile
- Tile 1: `min(16, 20-16) = 4` rows → partial tile (padded to 16×16 on HW)

---

## Pass 3 — GemminiToGemminiLow

**File:** `Conversion/GemminiToGemminiLow/GemminiToGemminiLow.cpp`

**What it does:**
Converts each `gemmini.*` op into one or more `gemmini_low.*` ops
that encode exactly what hardware instruction to issue.

The high-level ops use abstract offsets; the low-level ops use
the concrete scratchpad addresses that will be resolved in Pass 4.

**Conversion table:**
| High-level op | Low-level op |
|---------------|--------------|
| `gemmini.config` | `gemmini_low.config` |
| `gemmini.mvin` | `gemmini_low.mvin` |
| `gemmini.mvout` | `gemmini_low.mvout` |
| `gemmini.matmul` | `gemmini_low.matmul` |

The `copyAttrsExceptNamed()` helper in `GemminiLowRewrite.cpp` is used
to copy attributes from old ops to new ops, excluding a named subset.

---

## Pass 4 — StaticScratchpadAllocation

**File:** `Transform/GemminiLow/StaticScratchpadAllocation.cpp`

**What it does:**
Assigns a fixed compile-time scratchpad row number to every
`gemmini_low.mvin` and `gemmini_low.mvout` in the function.

**Why it is needed:**
The scratchpad is fast on-chip SRAM (256 KB, 16384 rows).
If two tiles are loaded into the same rows, they overwrite each other.
This pass allocates non-overlapping row ranges so tiles live peacefully
side by side.

**Simple analogy:**
Think of the scratchpad as a whiteboard with 16384 rows. The pass
assigns "row 0–15 for tile A", "row 16–31 for tile B", etc.

**Example — before and after:**
```
Before:  gemmini_low.mvin ... spad_row = %unknown
         gemmini_low.mvin ... spad_row = %unknown

After:   gemmini_low.mvin ... spad_row = 0
         gemmini_low.mvin ... spad_row = 16
```

**Capacity check:**
Total scratchpad = 16384 rows. Maximum simultaneous tiles:
- A tile: 16 rows
- B tile: 16 rows
- C accumulator tile: 16 rows (in accumulator, not scratchpad)
→ Fits 1024 tiles in scratchpad, 64 tiles in accumulator.

---

## Pass 5 — GemminiLowRewrite

**File:** `Transform/GemminiLow/GemminiLowRewrite.cpp`

**What it does:**
Canonicalizes and optimizes the GemminiLow IR:

1. **Fold redundant fences.** If two fence ops appear back-to-back
   with no operations between them, the second fence is redundant.
   ```
   Before:  gemmini_low.fence + gemmini_low.fence
   After:   gemmini_low.fence
   ```

2. **Merge adjacent mvin sequences.** When two consecutive mvin ops
   load from the same source matrix at consecutive offsets, they may
   be merged into one wider transfer (subject to hardware constraints).

3. **Attribute cleanup.** Removes internal compiler-only attributes
   (`gemmini.high_level`, etc.) that should not appear in the final IR.

---

## Pass 6 — GemminiLowToLLVM

**File:** `Conversion/GemminiLowToLLVM/GemminiLowToLLVM.cpp`

**What it does:**
Converts every `gemmini_low.*` op into LLVM IR using **CUSTOM_3**
inline-assembly sequences (the RoCC custom function opcode 3).

This is the final step before standard LLVM compiles everything to
RISC-V machine code.

**RoCC (Rocket Custom Coprocessor) instruction format:**

Each Gemmini instruction is encoded as a RISC-V custom instruction
using opcode `CUSTOM_3` (opcode bits `0b0001011`).

Simple example — `mvin` in LLVM inline-asm:
```llvm
; Move data into scratchpad
call void asm sideeffect
  ".insn r CUSTOM_3, 0, 0, x0, $0, $1",
  "r,r"(i64 %spad_addr, i64 %dram_ptr)
```

The LLVM assembler turns this text into the correct RISC-V bit pattern.

---

## How Passes Are Registered

Each pass is registered inside `GemminiAccelerator.cpp`:

```cpp
void GemminiAccelerator::registerPasses() {
  gemmini::registerONNXToGemminiPass();
  gemmini::registerGemminiTilingPass();
  gemmini::registerGemminiToGemminiLowPass();
  gemmini::registerStaticScratchpadAllocationPass();
  gemmini::registerGemminiLowRewritePass();
  gemmini::registerGemminiLowToLLVMPass();
}
```

Registration means the pass is given a name (e.g. `"--convert-onnx-to-gemmini"`)
so it can be called by name from the command line or from a pipeline string.

---

## Debugging Tips

| What you want to see | Command |
|---------------------|---------|
| IR after Pass 1 | `onnx-mlir --maccel=Gemmini --EmitGemminiIR model.onnx` |
| IR after all passes (MLIR) | `onnx-mlir --maccel=Gemmini --EmitMLIR model.onnx` |
| IR at LLVM level | `onnx-mlir --maccel=Gemmini --EmitLLVMIR model.onnx` |

Each output file will contain the IR at that stage so you can inspect
which ops were lowered and how scratchpad rows were assigned.
