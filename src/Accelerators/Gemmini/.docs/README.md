# Gemmini Accelerator — Documentation Index

This folder contains plain-language documentation for the entire
Gemmini backend in `src/Accelerators/Gemmini/`.

---

## Documents

| File | What it covers |
|------|----------------|
| [01-overview.md](01-overview.md) | What Gemmini is, how it fits in ONNX-MLIR, the three-part architecture, and directory map. Start here. |
| [02-source-files.md](02-source-files.md) | Every `.hpp/.cpp/.td` file listed with its role, key functions, and short examples. |
| [03-compiler-pipeline.md](03-compiler-pipeline.md) | The 6 compiler passes explained step by step with before/after IR examples. |
| [04-dialects.md](04-dialects.md) | The `gemmini` and `gemmini_low` MLIR dialects: all 4 ops each, their arguments and side effects. |
| [05-runtime-functions.md](05-runtime-functions.md) | All 33 `om_gemmini_*` runtime functions: signature, role, simple example, and quick-reference table. |
| [06-hardware-parameters.md](06-hardware-parameters.md) | Every constant in `gemmini_params.hpp` and `GemminiTargetInfo.hpp` with meaning and simple examples. |

---

## Quick Orientation

### "How does a MatMul get compiled?"

```
onnx.MatMul
  Pass 1 → gemmini.matmul
  Pass 2 → tiled loop of gemmini.mvin / gemmini.matmul / gemmini.mvout
  Pass 3 → gemmini_low.mvin / gemmini_low.matmul / gemmini_low.mvout
  Pass 4 → row numbers assigned (e.g. spad_row=0, spad_row=16)
  Pass 5 → canonicalized
  Pass 6 → LLVM CUSTOM_3 inline-asm
```

### "Where is function X defined?"

| Function / type | File |
|----------------|------|
| `om_gemmini_conv_f32_bias` | `Runtime/OMRuntimeGemmini.hpp` (decl), `Runtime/OMRuntimeGemmini.cpp` (impl) |
| `GemminiTargetInfo::dim` | `Support/GemminiTargetInfo.hpp` |
| `DIM`, `BANK_ROWS`, `elem_t` | `Runtime/gemmini_params.hpp` |
| `gemmini.mvin` op | `Dialect/Gemmini/GemminiOps.td` + `GemminiOps.hpp/cpp` |
| `gemmini_low.mvin` op | `Dialect/GemminiLow/GemminiLowOps.td` + `GemminiLowOps.hpp/cpp` |
| Pass factories | `Pass/GemminiPasses.hpp` |
| Plugin entry point | `GemminiAccelerator.hpp/cpp` |

### "What does `--maccel=Gemmini` do?"

1. Loads `GemminiAccelerator` plugin.
2. Registers `gemmini` and `gemmini_low` dialects.
3. Adds the 6-pass pipeline to the compiler.
4. Pass 1 rewrites MatMul/Conv/Gemm ONNX ops as Gemmini ops.
5. Later passes lower all the way to RISC-V inline assembly.
