# Gemmini Source Files — Complete Guide

Every source file in `src/Accelerators/Gemmini/` is listed here with its
role, key functions, and a plain-English explanation.

---

## Top Level

### `GemminiAccelerator.hpp` / `GemminiAccelerator.cpp`

**Role:** The single entry point for the Gemmini backend plugin.
ONNX-MLIR calls this class to register dialects, passes, and pipeline steps.

Think of it as the "front door" of the entire backend.

**Class:** `GemminiAccelerator : public Accelerator`

| Method | What it does |
|--------|-------------|
| `getVersionNumber()` | Returns `0x030000` (version 3.0.0). Used to check ABI compatibility at runtime. |
| `registerDialects(registry)` | Tells MLIR that `gemmini` and `gemmini_low` dialects exist. |
| `registerPasses()` | Registers the 6 Gemmini compiler passes so they can be looked up by name. |
| `conversionTargetONNXToKrnl(target)` | Marks Gemmini dialect ops as "legal" so the lowering framework does not try to remove them. |
| `rewritePatternONNXToKrnl(patterns, ctx, options)` | Provides the pattern list that converts ONNX ops into Gemmini ops. |
| `addPasses(pm, target, options)` | Adds all 6 passes to the compiler's pass manager in the right order. |

Simple example — what happens when the user runs `onnx-mlir --maccel=Gemmini`:
```
1. ONNX-MLIR finds GemminiAccelerator plugin
2. Calls registerDialects()  → now "gemmini" ops are known
3. Calls registerPasses()    → all 6 passes are available
4. Calls addPasses()         → builds the pipeline
5. Pipeline runs on the ONNX model
```

---

## `Compiler/`

### `GemminiCompilerOptions.hpp`

**Role:** Defines enums and macros for command-line compiler options.

| Enum / Macro | Meaning |
|--------------|---------|
| `GemminiInstrumentationStage::Gemmini` | Profile the overall Gemmini path. |
| `GemminiInstrumentationStage::GemminiLoad` | Profile mvin (load to scratchpad) operations. |
| `GemminiInstrumentationStage::GemminiStore` | Profile mvout (store from scratchpad) operations. |
| `GemminiInstrumentationStage::GemminiCompute` | Profile the compute (matmul) operations. |
| `GemminiInstrumentationStage::GemminiFlush` | Profile flush/fence operations. |
| `ProfileIR_Gemmini` | IR profiling category for Gemmini ops. |
| `GemminiUnsupportedOps` | Optimization report: which ONNX ops could not map to Gemmini. |
| `EmitGemminiNone` | Do not emit Gemmini IR (default). |
| `EmitGemminiIR` | Stop after emitting Gemmini dialect IR (for debugging). |

### `GemminiCompilerUtils.hpp` / `GemminiCompilerUtils.cpp`

**Role:** Orchestrates which passes run and in what order. This is the
"recipe book" for the compilation pipeline.

| Function | What it does |
|----------|-------------|
| `configurePassesGemmini()` | Called once at startup. Sets initial pass configuration state. |
| `addONNXToGemminiPasses(pm)` | Adds `ONNXToGemmini` pass + a canonicalize step to `pm`. |
| `addGemminiToGemminiLowPasses(pm)` | Adds 4 passes: Tiling → GemminiToGemminiLow → StaticScratchpadAlloc → GemminiLowRewrite. |
| `addPassesGemmini(pm, options)` | Builds the full Gemmini pipeline by calling both helpers above. |

Simple example — what `addPassesGemmini` does to the pass manager:
```
pm.addPass(createONNXToGemminiPass())
pm.addPass(canonicalize)
pm.addPass(createGemminiTilingPass())
pm.addPass(createGemminiToGemminiLowPass())
pm.addPass(createStaticScratchpadAllocationPass())
pm.addPass(createGemminiLowRewritePass())
pm.addPass(createGemminiLowToLLVMPass())
```

---

## `Pass/`

### `GemminiPasses.hpp`

**Role:** Declares the factory functions that create each pass object.
Other files `#include` this to get access to the passes.

| Factory function | Creates |
|-----------------|---------|
| `createONNXToGemminiPass()` | Pass 1: ONNX → Gemmini dialect |
| `createGemminiTilingPass()` | Pass 2: Tile ops into 16×16 loops |
| `createGemminiToGemminiLowPass()` | Pass 3: Gemmini → GemminiLow |
| `createStaticScratchpadAllocationPass()` | Pass 4: Assign scratchpad row numbers |
| `createGemminiLowRewritePass()` | Pass 5: Canonicalize GemminiLow |
| `createGemminiLowToLLVMPass()` | Pass 6: GemminiLow → LLVM IR |

---

## `Dialect/Gemmini/`

### `Gemmini.td`

**Role:** TableGen dialect definition. Defines the name, C++ namespace,
and assembly format for the high-level `gemmini` dialect.

### `GemminiOps.td`

**Role:** TableGen operation definitions. Each ONNX op that maps to Gemmini
becomes one or more of these higher-level operations.

### `GemminiOps.hpp` / `GemminiOps.cpp`

**Role:** C++ implementation of the 4 high-level Gemmini dialect operations.

| Op | Arguments | What it represents |
|----|-----------|-------------------|
| `gemmini.config` | `dataflow` (string), optional `a_transpose`, `b_transpose` | Send `CONFIG_EX` instruction to set systolic array mode. |
| `gemmini.mvin` | `source` (memref), `spad_offset_rows`, `tile_rows`, `tile_cols` | Move a tile from main memory **into** the scratchpad. |
| `gemmini.mvout` | `dest` (memref), `spad_offset_rows`, `tile_rows`, `tile_cols` | Move a tile **out** of the scratchpad/accumulator to main memory. |
| `gemmini.matmul` | `lhs`, `rhs`, `out` (memrefs), `mode`, `dataflow` | Issue a matrix multiply using scratchpad-resident tiles. |

Simple example — a 16×16 matmul in Gemmini dialect:
```mlir
// Move A tile into scratchpad at row 0
gemmini.mvin %A_tile, %c0, %c16, %c16

// Move B tile into scratchpad at row 16
gemmini.mvin %B_tile, %c16, %c16, %c16

// Multiply: result goes to accumulator
gemmini.matmul %A_tile, %B_tile, %out, "preload", "WS"

// Move result out of accumulator to memory
gemmini.mvout %out_tile, %c0, %c16, %c16
```

### `GemminiTypes.td`

**Role:** TableGen type definitions for any custom types used by the dialect.

---

## `Dialect/GemminiLow/`

### `GemminiLow.td` / `GemminiLowOps.td`

**Role:** Same structure as the high-level dialect but for low-level RoCC
instructions. "Low" means these ops map 1-to-1 to hardware instruction opcodes.

### `GemminiLowOps.hpp` / `GemminiLowOps.cpp`

**Role:** The 4 low-level ops mirror the high-level ops but encode exact
hardware addresses and flags.

| Op | Description |
|----|-------------|
| `gemmini_low.config` | Direct `CONFIG_EX` with transposed flags |
| `gemmini_low.mvin` | Direct scratchpad row address form of mvin |
| `gemmini_low.mvout` | Direct output address form of mvout |
| `gemmini_low.matmul` | Against pre-assigned scratchpad row numbers |

---

## `Conversion/ONNXToGemmini/`

### `ONNXToGemmini.hpp` / `ONNXToGemmini.cpp`

**Role:** Pass 1. Detects ONNX ops that Gemmini can accelerate and rewrites
them as `gemmini.*` dialect ops.

| Key function | What it does |
|-------------|-------------|
| `isSupportedGemminiMatMul(op)` | Returns true if the MatMul has: rank-2, static shape, `i32` element type. |
| `isSupportedGemminiMatMulInteger(op)` | Returns true if the MatMulInteger uses `i8 × i8 → i32`. |
| `isSupportedMatMulZeroPoint(zp)` | Checks that zero-point tensors are scalars or 1-D. |
| `populateONNXToGemminiPatterns(patterns)` | Adds all ONNX→Gemmini rewrite patterns to the pattern set. |

Simple example — what happens to `onnx.MatMul`:
```
Before:  onnx.MatMul(%A_f32, %B_f32) -> tensor<16x16xf32>
After:   gemmini.matmul(%A, %B, %out, "preload", "WS")
         (with krnl.call to om_gemmini_matmul_f32_hw for runtime path)
```

---

## `Conversion/GemminiToGemminiLow/`

### `GemminiToGemminiLow.hpp` / `GemminiToGemminiLow.cpp`

**Role:** Pass 3. Converts each high-level `gemmini.*` op into one or more
`gemmini_low.*` ops with concrete hardware addresses.

This pass resolves the scratchpad row assignments that were left as abstract
offsets in the high-level dialect.

---

## `Conversion/GemminiLowToLLVM/`

### `GemminiLowToLLVM.hpp` / `GemminiLowToLLVM.cpp`

**Role:** Pass 6. Converts `gemmini_low.*` ops into LLVM IR inline-assembly
sequences that directly encode **CUSTOM_3** RoCC instructions.

Simple example — what a `gemmini_low.mvin` becomes in LLVM IR:
```llvm
; Encode mvin as RISC-V CUSTOM_3 instruction
call void asm sideeffect "...rocc encoding...", ""(i64 %spad_addr, i64 %dram_addr)
```

---

## `Transform/Gemmini/`

### `GemminiTiling.hpp` / `GemminiTiling.cpp`

**Role:** Pass 2. Takes a whole-matrix `gemmini.matmul` op (any size) and
replaces it with nested loops that process one `16 × 16` tile per iteration.

This is necessary because the systolic array can only compute `16 × 16` at a time.

Simple example — tiling a `64 × 64` matrix:
```
Before: gemmini.matmul A[64×64], B[64×64] → C[64×64]

After:
  for m = 0 to 4:         ← 4 tiles along M
    for n = 0 to 4:       ← 4 tiles along N
      for k = 0 to 4:     ← 4 tiles along K
        gemmini.mvin A_tile[16×16] at scratchpad row (m*16)
        gemmini.mvin B_tile[16×16] at scratchpad row (n*16)
        gemmini.matmul (accumulate)
  gemmini.mvout C[64×64]
```

| Key variable / function | Role |
|------------------------|------|
| `GemminiTilingPass` | The pass class, applied once per `func::FuncOp`. |
| `dim` (from `GemminiTargetInfo`) | Tile size = 16. Used as loop step. |
| `minIndex(a, b)` | Helper: returns the smaller of two SSA values. Used for edge tiles. |

---

## `Transform/GemminiLow/`

### `StaticScratchpadAllocation.hpp` / `StaticScratchpadAllocation.cpp`

**Role:** Pass 4. Assigns fixed (compile-time) scratchpad row numbers to
every `gemmini_low.mvin` and `gemmini_low.mvout` in the function.

Why this is needed: the scratchpad is a fast on-chip memory. To avoid
data conflicts, different tiles need different row ranges. This pass
allocates non-overlapping row ranges at compile time.

Simple example — before and after:
```
Before:  gemmini_low.mvin ... spad_row = ???

After:   gemmini_low.mvin ... spad_row = 0
         gemmini_low.mvin ... spad_row = 16
         gemmini_low.mvin ... spad_row = 32
```

### `GemminiLowRewrite.hpp` / `GemminiLowRewrite.cpp`

**Role:** Pass 5. Cleans up the GemminiLow IR: folds redundant fences,
merges adjacent operations, and removes no-op sequences.

| Key function | Role |
|-------------|------|
| `copyAttrsExceptNamed(op, names)` | Helper: copies all attributes from an op except the ones in `names`. Used when converting Gemmini ops to GemminiLow ops. |

---

## `Support/`

### `GemminiTargetInfo.hpp.in` / `GemminiTargetInfo.cpp`

**Role:** Compiler-side view of the hardware constants used during
compilation. `GemminiTargetInfo.hpp.in` is the source-tree template;
CMake reads `Runtime/gemmini-hardware-abi/include/gemmini_params.h` and
generates `GemminiTargetInfo.hpp` in the build tree. All passes that need to
know the tile size or memory layout read from the generated struct.

| Constant | Value | Meaning |
|----------|-------|---------|
| `dim` | 16 | Systolic array is 16×16. All tiles must be 16×16. |
| `bankNum` | 4 | Scratchpad has 4 memory banks. |
| `bankRows` | 4096 | Each bank holds 4096 rows. |
| `accRows` | 1024 | Accumulator holds 1024 rows. |
| `elemBits` | 8 | Each input element is 8 bits (int8). |
| `accBits` | 32 | Accumulator values are 32 bits (int32). |

| Helper method | Returns | Example |
|--------------|---------|---------|
| `scratchpadRows()` | `bankNum × bankRows` = 16384 | Total scratchpad rows |
| `scratchpadMatrices()` | `16384 / 16` = 1024 | How many 16×16 tiles fit in scratchpad |
| `accumulatorMatrices()` | `1024 / 16` = 64 | How many 16×16 tiles fit in accumulator |
| `getMatmulTileCounts(m,n,k)` | `{⌈m/16⌉, ⌈n/16⌉, ⌈k/16⌉}` | Number of tile loop iterations |

Simple example — `getMatmulTileCounts(64, 64, 64)` returns `{4, 4, 4}`.
That means 4×4×4 = 64 hardware matmul calls.

---

## `Runtime/`

### `OMRuntimeGemmini.hpp` / `OMRuntimeGemmini.cpp`

**Role:** Declares (and implements) all `om_gemmini_*` runtime functions.
These are the functions that the generated model code calls at runtime.

The `.cpp` file is compiled **only for RISC-V** (`#error` guard prevents
accidental compilation on x86).

See `05-runtime-functions.md` for a full function-by-function reference.

### `gemmini.hpp`

**Role:** Hardware abstraction layer. Provides C macros and structs that
wrap the raw RISC-V custom instruction encoding.

Key type: `gemmini_state_t` — a 100+ field struct tracking the full hardware
state including strides, pooling parameters, loop unrolling state, and counters.

Key enums:

| Enum | Values | Meaning |
|------|--------|---------|
| `Dataflow` | `OS`, `WS` | Output Stationary vs Weight Stationary |
| `Activation` | `NONE`, `RELU`, `LAYERNORM`, `IGELU`, `SOFTMAX` | Post-compute activation |
| `NormCmd` | `RESET`, `SUM`, `MEAN`, `VARIANCE`, ... | Normalization sequence commands |

### `gemmini_params.hpp`

**Role:** Chip-level constant definitions. Mirrors what the RISC-V Gemmini
hardware generator (`gemmini_scala/`) was configured with.

See `06-hardware-parameters.md` for a full constant-by-constant reference.

---

## Build Files

### `CMakeLists.txt` (top-level)

Sets `GEMMINI_ENABLED=1`, `GEMMINI_VERSION="3.0.0"`, and creates the
`OMGemminiAccel` library that links all sub-libraries together.

Conditionally defines:
- `ONNX_MLIR_WITH_GEMMINI` — backend is compiled in.
- `ONNX_MLIR_WITH_GEMMINI_RUNTIME` — runtime functions are compiled (RISC-V only).

Sub-folder CMakeLists follow the same pattern: define a library target,
list sources, and declare dependencies.
