# Gemmini Accelerator — Overview

This document explains what the Gemmini backend is, how it fits inside
ONNX-MLIR, and who is responsible for each part of the system.

---

## What is Gemmini?

Gemmini is a **systolic-array accelerator** developed at UC Berkeley.
It runs on RISC-V chips and is very good at matrix multiplication,
which is the most expensive part of neural network inference.

Simple analogy:

```
CPU:     a general worker who can do many things slowly
Gemmini: a specialist who only multiplies matrices, but does it very fast
```

The Gemmini systolic array is a grid of small multiplier units arranged in a
`16 × 16` pattern. Each cycle, data flows through the grid and partial
products accumulate, producing a `16 × 16` output tile per step.

---

## How Gemmini fits in ONNX-MLIR

ONNX-MLIR is a compiler. It takes a trained neural network (ONNX format) and
produces machine code that runs on a target device.

The Gemmini backend is a **plugin** (called an Accelerator) that ONNX-MLIR
loads when you pass `--maccel=Gemmini`.

```
ONNX model
    │
    ▼
ONNX-MLIR compiler  (--maccel=Gemmini)
    │
    ├─ detect MatMul / Conv / Gemm ops
    ├─ lower them through the Gemmini dialect pipeline
    │
    ▼
RISC-V object file  (.o)
    │
    ▼  (link with Gemmini runtime library)
RISC-V executable  runs on board / Spike simulator
```

---

## The Three Parts of the Gemmini Backend

### 1. Compiler passes  (`src/Accelerators/Gemmini/`)

These are pieces of code that transform MLIR (compiler IR) step by step.
They live in the `Conversion/`, `Transform/`, `Dialect/`, and `Pass/` folders.

Think of each pass as a translator:
- Pass 1 translates ONNX ops into Gemmini-dialect ops.
- Pass 2 tiles the big matrix into `16 × 16` chunks.
- Pass 3 translates into low-level RoCC hardware instructions.
- ...and so on.

### 2. Runtime functions  (`src/Accelerators/Gemmini/Runtime/`)

These are C functions with names starting `om_gemmini_*`.

The compiled model code calls these functions at runtime on the RISC-V board.
They contain the actual hardware-driving code (or soft emulations for ops
that run on the CPU scalar core).

### 3. Hardware parameters  (`src/Accelerators/Gemmini/Runtime/gemmini_params.hpp`)

These are constants that describe the physical chip:
- Systolic array size: `16 × 16`
- Scratchpad memory: 256 KB  (16 384 rows × 16 bytes, 4 banks × 4096 rows)
- Accumulator memory: 64 KB  (1 024 rows × 16 values × 4 bytes)
- Data type: `int8`

---

## Supported ONNX Operators (22 total)

| Path | Operators |
|------|-----------|
| Systolic array (Gemmini HW) | Conv, Gemm, MatMul, MatMulInteger, QLinearConv |
| Systolic array — fused activation | Relu (fused into Conv/MatMul/Gemm as a hardware post-compute step) |
| Systolic array — fused transpose | Transpose of A or B matrix fused into MatMul/Gemm via `a_transpose` / `b_transpose` hardware flags |
| Scalar RISC-V CPU loops | Relu (standalone), Transpose (standalone), Sigmoid, Mul, Add, BatchNormalization, Tanh, AveragePool, MaxPool, GlobalAveragePool, Reshape, Flatten, Concat, Pad, Resize, Slice, Split, Softmax |

> **Relu note:** ReLU has two execution paths. When it immediately follows a
> Conv/MatMul/Gemm, the compiler fuses it via `Activation::RELU` in the config
> instruction — zero extra cycles. Standalone ReLU falls back to
> `om_gemmini_relu_f32` on the scalar CPU core.

> **Transpose note:** Transpose has two execution paths. When the transposed
> tensor is directly consumed as the A or B operand of a MatMul/Gemm, the
> compiler sets `a_transpose=true` or `b_transpose=true` in `gemmini.config`
> and the hardware reads the matrix in transposed order during mvin — zero
> extra cycles. Standalone Transpose (any other permutation or non-matmul
> consumer) falls back to `om_gemmini_transpose_f32_hw` on the scalar CPU core.

---

## Directory Map

```
src/Accelerators/Gemmini/
├── GemminiAccelerator.hpp/cpp      ← plugin entry point
├── Compiler/                       ← pass orchestration
├── Dialect/
│   ├── Gemmini/                    ← high-level dialect ops
│   └── GemminiLow/                 ← low-level RoCC dialect ops
├── Conversion/
│   ├── ONNXToGemmini/              ← ONNX → Gemmini
│   ├── GemminiToGemminiLow/        ← Gemmini → GemminiLow
│   └── GemminiLowToLLVM/           ← GemminiLow → LLVM IR
├── Transform/
│   ├── Gemmini/                    ← tiling pass
│   └── GemminiLow/                 ← low-level rewrites + scratchpad alloc
├── Support/
│   └── GemminiTargetInfo.hpp/cpp   ← hardware constants helper
├── Pass/
│   └── GemminiPasses.hpp           ← pass factory declarations
└── Runtime/
    ├── OMRuntimeGemmini.hpp/cpp    ← om_gemmini_* function declarations/impl
    ├── gemmini.hpp                 ← hardware abstraction layer
    └── gemmini_params.hpp          ← chip constant definitions
```

---

## Compilation Pipeline Summary

```
ONNX IR
  ↓  [ONNXToGemminiPass]       detect MatMul/Conv, emit gemmini.matmul ops
  ↓  [GemminiTilingPass]       break big matrices into 16×16 tiles
  ↓  [GemminiToGemminiLowPass] translate to direct RoCC instruction ops
  ↓  [StaticScratchpadAlloc]   assign fixed scratchpad row addresses
  ↓  [GemminiLowRewritePass]   canonicalize and fold redundant instructions
  ↓  [GemminiLowToLLVMPass]    emit LLVM inline-asm CUSTOM_3 instructions
RISC-V machine code
```

See `03-compiler-pipeline.md` for a full explanation of each pass.
