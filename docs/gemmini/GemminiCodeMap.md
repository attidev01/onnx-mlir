# Gemmini Accelerator — Source Code Map

Complete guide to every folder and file in `src/Accelerators/Gemmini/`:
what it does, how it fits the compilation pipeline, and how folders relate to each other.

---

## Directory Tree

```
src/Accelerators/Gemmini/
│
├── GemminiAccelerator.hpp / .cpp       ← backend entry point (plugin singleton)
├── CMakeLists.txt                      ← top-level CMake, subdirs added here
│
├── Compiler/                           ← CLI options + pass-orchestration helpers
│   ├── GemminiCompilerOptions.hpp
│   ├── GemminiCompilerUtils.hpp / .cpp
│   └── CMakeLists.txt
│
├── Conversion/                         ← dialect-lowering conversion passes
│   ├── CMakeLists.txt
│   ├── ONNXToGemmini/                  ← ONNX ops → krnl.call om_gemmini_* (runtime path)
│   │   ├── ONNXToGemmini.hpp / .cpp
│   │   └── CMakeLists.txt
│   ├── GemminiToGemminiLow/            ← Gemmini dialect → GemminiLow dialect
│   │   ├── GemminiToGemminiLow.hpp / .cpp
│   │   └── CMakeLists.txt
│   └── GemminiLowToLLVM/              ← GemminiLow dialect → LLVM IR (direct-RoCC path)
│       ├── GemminiLowToLLVM.hpp / .cpp
│       └── CMakeLists.txt
│
├── Dialect/                            ← MLIR dialect definitions (TableGen → C++)
│   ├── CMakeLists.txt
│   ├── Gemmini/                        ← high-level Gemmini dialect
│   │   ├── Gemmini.td                  ← dialect root
│   │   ├── GemminiOps.td               ← operation definitions
│   │   ├── GemminiTypes.td             ← type definitions
│   │   ├── GemminiOps.hpp / .cpp       ← generated-inc consumers
│   │   └── CMakeLists.txt
│   └── GemminiLow/                     ← low-level RoCC-instruction dialect
│       ├── GemminiLow.td               ← dialect root
│       ├── GemminiLowOps.td            ← low-level op definitions
│       ├── GemminiLowOps.hpp / .cpp    ← generated-inc consumers
│       └── CMakeLists.txt
│
├── Pass/                               ← pass factory declarations
│   └── GemminiPasses.hpp
│
├── Runtime/                            ← RISC-V runtime delegate library
│   ├── gemmini_params.hpp              ← hardware constants (DIM, BANK_ROWS, …)
│   ├── gemmini.hpp                     ← C++ simulator header (RoCC wrappers)
│   ├── OMRuntimeGemmini.hpp            ← public C ABI — 33 om_gemmini_* symbols
│   ├── OMRuntimeGemmini.cpp            ← implementations (RISC-V only)
│   └── CMakeLists.txt
│
├── Support/                            ← shared hardware-parameter helper
│   ├── GemminiTargetInfo.hpp / .cpp
│   └── CMakeLists.txt
│
└── Transform/                          ← in-dialect transformation passes
    ├── CMakeLists.txt
    ├── Gemmini/                        ← tiling on the high-level dialect
    │   ├── GemminiTiling.hpp / .cpp
    │   └── CMakeLists.txt
    └── GemminiLow/                     ← rewriting + scratchpad allocation
        ├── GemminiLowRewrite.hpp / .cpp
        ├── StaticScratchpadAllocation.hpp / .cpp
        └── CMakeLists.txt
```

---

## Compilation Pipeline

The backend has two execution paths that share the entry stages:

```
ONNX IR
   │
   ▼
[ONNXToGemmini pass]  ─────────────────────────────────────────────────────┐
   │  lowers 23 ONNX ops to krnl.call "om_gemmini_*" nodes                 │ runtime-delegate path
   ▼                                                                        │ (all float + quantized ops)
Krnl IR                                                                     │
   │                                                                        │
   ▼                                                                        │
[standard ONNX-MLIR lowering]                                              │
   │                                                                        ▼
LLVM IR  ─────────────────────────────────────────>  compiled .so calls om_gemmini_* at runtime
                                                      (RuntimeGemmini library on RISC-V)

─────── direct-RoCC path (integer / tiled-matmul) ────────────────────────

ONNX IR
   │
   ▼
[ONNXToGemmini pass]   ← emits gemmini.* ops (not krnl.call) for direct path
   │
   ▼
Gemmini dialect IR
   │
   ▼
[GemminiTiling pass]   ← tiles matmul into DIM×DIM chunks, pads edge tiles
   │
   ▼
[GemminiToGemminiLow pass]  ← converts gemmini.mvin/compute/mvout → gemmini_low.*
   │
   ▼
GemminiLow dialect IR
   │
   ▼
[StaticScratchpadAllocation pass]  ← assigns static scratchpad addresses
   │
   ▼
[GemminiLowRewrite pass]  ← peephole cleans + final canonicalization
   │
   ▼
[GemminiLowToLLVM pass]  ← inline RISC-V RoCC intrinsics via LLVM IR
   │
   ▼
LLVM IR  →  riscv64 object  →  linked binary
```

---

## Folder-by-Folder Reference

### Root: `src/Accelerators/Gemmini/`

**What it is:** The Gemmini accelerator package. All subdirectories are registered
from the top-level `CMakeLists.txt`. The two files at the root are the plugin
entry point consumed by ONNX-MLIR's `Accelerator` registry.

| File | Role |
|---|---|
| `GemminiAccelerator.hpp` | Declares `class GemminiAccelerator` — the singleton that implements the `onnx_mlir::accel::Accelerator` interface. Exposes hooks: `registerDialects`, `addPasses`, `rewritePatternONNXToKrnl`, `rewritePatternKrnlToLLVM`, `conversionTargetONNXToKrnl`, `conversionTargetKrnlToLLVM`. |
| `GemminiAccelerator.cpp` | Implements the singleton. Registers the `gemmini` and `gemmini_low` dialects, calls the Compiler utilities to build the pass pipeline, and provides a C entry point (`createGemminiAccelerator`) so the TableGen-generated `InitAccelerators` can register it. |
| `CMakeLists.txt` | Top-level CMake. Adds all sub-directories: `Compiler`, `Conversion`, `Dialect`, `Pass`, `Runtime`, `Support`, `Transform`. Defines the `GemminiAccelerator` CMake library target. |

---

### `Compiler/`

**What it is:** Two files that sit between the ONNX-MLIR driver and the Gemmini
pass objects. One defines CLI flag macros; the other provides the three functions
that `GemminiAccelerator::addPasses` calls to build the full pipeline.

| File | Role |
|---|---|
| `GemminiCompilerOptions.hpp` | Macro expansions plugged into the ONNX-MLIR `INSTRUMENTSTAGE_ENUM`, `PROFILEIR_CL_ENUM`, and `OPTREPORT_ENUM` extension points. Adds `Gemmini`, `GemminiLoad`, `GemminiStore`, `GemminiCompute`, `GemminiFlush` instrumentation stages and a `GemminiUnsupportedOps` optimization-report category. Also defines `GemminiEmissionTargetType`. |
| `GemminiCompilerUtils.hpp` | Declares `configurePassesGemmini()`, `addONNXToGemminiPasses(pm)`, `addGemminiToGemminiLowPasses(pm)`, and `addPassesGemmini(module, pm, emissionTarget, outputName)`. |
| `GemminiCompilerUtils.cpp` | Implements the four pass-orchestration functions. `addPassesGemmini` is the single call site in `GemminiAccelerator::addPasses`. |

**Relation to other folders:** Compiler/ depends on Pass/ for `createONNXToGemminiPass` etc., and
on Conversion/ and Transform/ for the concrete pass factories it adds to the pipeline.

---

### `Conversion/`

**What it is:** Three conversion passes — one per lowering stage in the direct-RoCC path plus
the runtime-delegate entry stage. Each lives in its own sub-directory with its own CMake target.

#### `Conversion/ONNXToGemmini/`

| File | Role |
|---|---|
| `ONNXToGemmini.hpp` | Declares `createONNXToGemminiPass()` and `populateONNXToKrnlForGemmini(patterns, typeConverter, ctx)`. |
| `ONNXToGemmini.cpp` | The largest file in the backend (~1800 lines). Contains one rewrite pattern per supported ONNX op (23 total). Each pattern either (a) emits `krnl.call "om_gemmini_*"` nodes for the runtime-delegate path, or (b) emits `gemmini.*` ops for the direct-RoCC path. Reads `GemminiTargetInfo` for hardware constants. Uses string array `kFuncNames[numOutputs]` for multi-output ops like Split. |

Supported ops lowered here: MatMul, MatMulInteger, QLinearConv, Conv, ConvTranspose,
Gemm, ReLU, BatchNormalization, Add, Sigmoid, Mul, Softmax,
GlobalAveragePool, MaxPool, AveragePool, Resize, Pad, Slice,
Concat, Transpose, Split (2/3/4 outputs), MatMul (f16).

#### `Conversion/GemminiToGemminiLow/`

| File | Role |
|---|---|
| `GemminiToGemminiLow.hpp` | Declares `createGemminiToGemminiLowPass()`. |
| `GemminiToGemminiLow.cpp` | Converts each `gemmini.*` op to the equivalent `gemmini_low.*` op. Essentially a structural 1-to-1 lowering that replaces the high-level dialect with the instruction-level dialect without changing tile shapes. |

#### `Conversion/GemminiLowToLLVM/`

| File | Role |
|---|---|
| `GemminiLowToLLVM.hpp` | Declares `createGemminiLowToLLVMPass()`. |
| `GemminiLowToLLVM.cpp` | Converts `gemmini_low.*` ops to LLVM IR by inlining the RoCC instruction encoding (CUSTOM_3 opcode, rs1/rs2 register packing). This is the final stage of the direct-RoCC compilation path; the output is native RISC-V machine code when assembled. |

**Relation to other folders:** Conversion/ reads dialect headers from Dialect/ and
uses Support/GemminiTargetInfo for hardware constants. The result feeds Transform/
(Gemmini→GemminiLow direction) or the standard LLVM pipeline.

---

### `Dialect/`

**What it is:** TableGen source files (`.td`) that define two MLIR dialects, plus
the `.hpp`/`.cpp` files that `#include` the generated `.inc` files produced by `mlir-tblgen`.

#### `Dialect/Gemmini/` — high-level Gemmini dialect (`gemmini.*`)

| File | Role |
|---|---|
| `Gemmini.td` | Dialect root: sets `name = "gemmini"`, C++ namespace `::onnx_mlir::gemmini`, enables default type printer/parser. Includes `GemminiTypes.td`. |
| `GemminiOps.td` | Defines all high-level ops: `gemmini.config`, `gemmini.mvin`, `gemmini.preload`, `gemmini.compute`, `gemmini.mvout`, and higher-level tiled-matmul ops that the tiling pass decomposes. |
| `GemminiTypes.td` | Custom scalar/memref type definitions used by `gemmini.*` ops (e.g. the accumulator element type). |
| `GemminiOps.hpp` | Pulls in `GemminiDialect.hpp.inc`, `GemminiTypes.hpp.inc`, and `GemminiOps.hpp.inc` generated by CMake. Include this header to use `gemmini.*` ops in C++ passes. |
| `GemminiOps.cpp` | Implements op verifiers and custom assembly format methods declared in the `.td`; includes `GemminiOps.cpp.inc`. |

#### `Dialect/GemminiLow/` — instruction-level dialect (`gemmini_low.*`)

| File | Role |
|---|---|
| `GemminiLow.td` | Dialect root: `name = "gemmini_low"`, C++ namespace `::onnx_mlir::gemmini`. |
| `GemminiLowOps.td` | Defines one op per RoCC instruction: `gemmini_low.config`, `gemmini_low.mvin`, `gemmini_low.preload`, `gemmini_low.compute`, `gemmini_low.mvout`. Each op carries exactly the attributes needed to encode the RoCC instruction fields. |
| `GemminiLowOps.hpp` | Pulls in `GemminiLowDialect.hpp.inc` and `GemminiLowOps.hpp.inc`. Include this to use `gemmini_low.*` ops. |
| `GemminiLowOps.cpp` | Op verifiers and any custom builders for `gemmini_low.*` ops. |

**How TableGen generates code:**  
CMake runs `mlir-tblgen` on the `.td` files and writes `.hpp.inc` / `.cpp.inc`
files into the build directory (`gemmini_toolchain_build/src/Accelerators/Gemmini/Dialect/…`).
The `.hpp` and `.cpp` files are thin wrappers that `#include` those generated files.

---

### `Pass/`

**What it is:** A single header that declares all six pass factory functions.
There is no `.cpp` here — the factories are defined in Conversion/ and Transform/.

| File | Role |
|---|---|
| `GemminiPasses.hpp` | Declares: `createONNXToGemminiPass`, `createGemminiTilingPass`, `createGemminiToGemminiLowPass`, `createStaticScratchpadAllocationPass`, `createGemminiLowRewritePass`, `createGemminiLowToLLVMPass`. Also includes `GemminiCompilerOptions.hpp`. This header is the single include that Compiler/ needs to build the full pipeline. |

**Why a separate Pass/ folder?** The same pattern used by NNPA and other ONNX-MLIR accelerators:
one central header aggregates all pass names so there are no circular includes between
Compiler/, Conversion/, and Transform/.

---

### `Runtime/`

**What it is:** The RISC-V runtime delegate library. Compiled models call into this
library via `krnl.call "om_gemmini_*"` nodes emitted by `ONNXToGemmini.cpp`. The
implementations use Berkeley's `gemmini.hpp` RoCC wrappers and are guarded by
`#if !defined(__riscv) / #error` so they can only be built for RISC-V targets.

| File | Role |
|---|---|
| `gemmini_params.hpp` | Hardware constants: `DIM=16`, `BANK_NUM=4`, `BANK_ROWS=4096`, `ACC_ROWS=1024`, `XCUSTOM_ACC=3`, `elem_t=int8_t`, `acc_t=int32_t`, `ADDR_LEN`. Included by `gemmini.hpp`. |
| `gemmini.hpp` | UC Berkeley Gemmini C++ simulator/driver header. Includes `<riscv/rocc.h>`, `<random>`, `<limits>`, and `gemmini_params.hpp`. Provides `tiled_matmul_auto`, `tiled_conv_auto`, `gemmini_flush`, scratchpad load/store macros, and `struct gemmini_state_t`. Included by `OMRuntimeGemmini.cpp`. |
| `OMRuntimeGemmini.hpp` | **Public C ABI.** All 33 `om_gemmini_*` symbols declared with `extern "C"` linkage. This is the contract between compiled ONNX models and the runtime library. Divided into five groups: integer/quantized (4), float HW path (7), elementwise/norm (6), pooling (3), linear-algebra (2), spatial (11). |
| `OMRuntimeGemmini.cpp` | **Implementations.** ~1500 lines. Each function unpacks `OMTensor` arguments, calls into `gemmini.hpp` macros or scalar CPU loops, and writes results back through OMTensor. Float ops quantize inputs to `int8_t`, call `tiled_matmul_auto`, then dequantize the `int32_t` accumulator. Guarded by `#if !defined(__riscv)`. |
| `CMakeLists.txt` | Defines the `RuntimeGemmini` STATIC library. The `if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")` guard ensures it is only built for RISC-V. Uses `LANGUAGE CXX` and `POSITION_INDEPENDENT_CODE TRUE`. |

**Include chain inside Runtime:**
```
OMRuntimeGemmini.cpp
  ├── include/onnx-mlir/Runtime/OMTensor.h   ← OMTensor struct / accessors
  ├── OMRuntimeGemmini.hpp                   ← own public ABI
  ├── src/Support/SmallFPConversion.h        ← f16 ↔ f32 conversion helpers
  └── gemmini.hpp
        └── gemmini_params.hpp              ← hardware constants
```

**33 public symbols by group:**

| Group | Count | Example symbols |
|---|---|---|
| Integer / quantized | 4 | `om_gemmini_matmulinteger_i8i8acc32`, `om_gemmini_qlinearconv_i8_bias` |
| Float HW path | 7 | `om_gemmini_matmul_f32_hw`, `om_gemmini_conv_f32_bias`, `om_gemmini_gemm_f32_bias` |
| Elementwise / norm | 6 | `om_gemmini_relu_f32`, `om_gemmini_batchnorm_f32`, `om_gemmini_sigmoid_f32` |
| Pooling | 3 | `om_gemmini_globalavgpool_f32`, `om_gemmini_maxpool_f32`, `om_gemmini_avgpool_f32` |
| Spatial | 11 | `om_gemmini_resize_nearest_f32`, `om_gemmini_pad_constant_f32`, `om_gemmini_split_2_f32` |
| Linear algebra | 2 | `om_gemmini_gemm_f32`, `om_gemmini_gemm_f32_bias` |

---

### `Support/`

**What it is:** A single compile-time struct that centralises all Gemmini hardware
parameters used by the compiler side (passes, tiling, scratchpad allocation).
Eliminates duplicated magic numbers across Conversion/ and Transform/.

| File | Role |
|---|---|
| `GemminiTargetInfo.hpp` | Defines `struct GemminiTargetInfo` with `static constexpr` fields: `dim=16`, `bankNum=4`, `bankRows=4096`, `accRows=1024`, `elemBits=8`, `accBits=32`. Provides helper functions `scratchpadRows()`, `scratchpadMatrices()`, `accumulatorMatrices()`, `getMatmulTileCounts(m,n,k)`. |
| `GemminiTargetInfo.cpp` | Thin translation unit; out-of-line definitions if required by the linker (most are `constexpr`). |

**Used by:** `ONNXToGemmini.cpp`, `GemminiTiling.cpp`, `StaticScratchpadAllocation.cpp`.
**Mirrors:** `Runtime/gemmini_params.hpp` — both encode the same hardware numbers
but for two different contexts (compiler vs. runtime).

---

### `Transform/`

**What it is:** Two sub-directories of in-dialect transformation passes that operate
on the Gemmini and GemminiLow dialects respectively. These are distinct from Conversion/
because they do not change the dialect — they reshape the IR within it.

#### `Transform/Gemmini/` — tiling on the high-level dialect

| File | Role |
|---|---|
| `GemminiTiling.hpp` | Declares `createGemminiTilingPass()`. |
| `GemminiTiling.cpp` | Implements the tiling pass. Walks `gemmini.*` matmul ops and decomposes them into sequences of `gemmini.mvin` / `gemmini.compute` / `gemmini.mvout` ops covering `DIM×DIM` tiles. Handles edge tiles by inserting zero-padding so every tile is exactly `16×16`. Reads `GemminiTargetInfo::dim` for tile size. |

#### `Transform/GemminiLow/` — rewriting and scratchpad allocation on the low-level dialect

| File | Role |
|---|---|
| `GemminiLowRewrite.hpp` | Declares `createGemminiLowRewritePass()`. |
| `GemminiLowRewrite.cpp` | Peephole rewrites on `gemmini_low.*` IR: removes redundant config ops, merges consecutive mvin/mvout sequences, and runs a canonicalisation. |
| `StaticScratchpadAllocation.hpp` | Declares `createStaticScratchpadAllocationPass()`. |
| `StaticScratchpadAllocation.cpp` | Assigns static `spad_offset_rows` values to each `gemmini_low.mvin` and `gemmini_low.preload` op. Performs a linear scan over the instruction sequence, tracking which scratchpad rows are live, and writes the `I64Attr` offset directly into the op attributes. |

---

## Cross-Folder Dependency Map

```
GemminiAccelerator.hpp / .cpp
  ├── Compiler/GemminiCompilerUtils       (builds the pass pipeline)
  │     └── Pass/GemminiPasses.hpp        (pass factory declarations)
  │           ├── Conversion/ONNXToGemmini
  │           ├── Conversion/GemminiToGemminiLow
  │           ├── Conversion/GemminiLowToLLVM
  │           ├── Transform/Gemmini/GemminiTiling
  │           ├── Transform/GemminiLow/GemminiLowRewrite
  │           └── Transform/GemminiLow/StaticScratchpadAllocation
  │
  ├── Dialect/Gemmini/GemminiOps.hpp      (gemmini.* op classes)
  │     └── [generated] GemminiOps.hpp.inc  (from GemminiOps.td via mlir-tblgen)
  │
  └── Dialect/GemminiLow/GemminiLowOps.hpp  (gemmini_low.* op classes)
        └── [generated] GemminiLowOps.hpp.inc

Conversion/ONNXToGemmini.cpp
  ├── Dialect/Gemmini/GemminiOps.hpp
  └── Support/GemminiTargetInfo.hpp

Transform/Gemmini/GemminiTiling.cpp
  ├── Dialect/Gemmini/GemminiOps.hpp
  └── Support/GemminiTargetInfo.hpp

Transform/GemminiLow/StaticScratchpadAllocation.cpp
  ├── Dialect/GemminiLow/GemminiLowOps.hpp
  └── Support/GemminiTargetInfo.hpp

Conversion/GemminiLowToLLVM.cpp
  └── Dialect/GemminiLow/GemminiLowOps.hpp

Runtime/OMRuntimeGemmini.cpp               (RISC-V only — not linked into host tools)
  ├── Runtime/OMRuntimeGemmini.hpp
  ├── Runtime/gemmini.hpp
  │     └── Runtime/gemmini_params.hpp
  └── src/Support/SmallFPConversion.h
```

---

## CMake Library Targets

| CMake Target | Source files | Linked by |
|---|---|---|
| `GemminiAccelerator` | `GemminiAccelerator.cpp` | `onnx-mlir` driver |
| `GemminiCompilerUtils` | `Compiler/GemminiCompilerUtils.cpp` | `GemminiAccelerator` |
| `ONNXToGemmini` | `Conversion/ONNXToGemmini/ONNXToGemmini.cpp` | `GemminiCompilerUtils` |
| `GemminiToGemminiLow` | `Conversion/GemminiToGemminiLow/GemminiToGemminiLow.cpp` | `GemminiCompilerUtils` |
| `GemminiLowToLLVM` | `Conversion/GemminiLowToLLVM/GemminiLowToLLVM.cpp` | `GemminiCompilerUtils` |
| `GemminiDialect` | `Dialect/Gemmini/GemminiOps.cpp` | `ONNXToGemmini`, `GemminiTiling`, `GemminiToGemminiLow` |
| `GemminiLowDialect` | `Dialect/GemminiLow/GemminiLowOps.cpp` | `GemminiToGemminiLow`, `StaticScratchpadAllocation`, `GemminiLowRewrite`, `GemminiLowToLLVM` |
| `GemminiTiling` | `Transform/Gemmini/GemminiTiling.cpp` | `GemminiCompilerUtils` |
| `GemminiLowRewrite` | `Transform/GemminiLow/GemminiLowRewrite.cpp` | `GemminiCompilerUtils` |
| `StaticScratchpadAllocation` | `Transform/GemminiLow/StaticScratchpadAllocation.cpp` | `GemminiCompilerUtils` |
| `GemminiTargetInfo` | `Support/GemminiTargetInfo.cpp` | `ONNXToGemmini`, `GemminiTiling`, `StaticScratchpadAllocation` |
| `RuntimeGemmini` *(RISC-V only)* | `Runtime/OMRuntimeGemmini.cpp` | compiled model `.so` at link time |

---

## File Count Summary

| Folder | `.td` | `.hpp` | `.cpp` | `CMakeLists.txt` | Total |
|---|---|---|---|---|---|
| Root | 0 | 1 | 1 | 1 | 3 |
| Compiler/ | 0 | 2 | 1 | 1 | 4 |
| Conversion/ | 0 | 3 | 3 | 4 | 10 |
| Dialect/ | 4 | 2 | 2 | 3 | 11 |
| Pass/ | 0 | 1 | 0 | 0 | 1 |
| Runtime/ | 0 | 3 | 1 | 1 | 5 |
| Support/ | 0 | 1 | 1 | 1 | 3 |
| Transform/ | 0 | 3 | 3 | 3 | 9 |
| **Total** | **4** | **16** | **12** | **14** | **46** |
