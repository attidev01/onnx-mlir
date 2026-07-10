# include

This folder contains the C header files used by the Gemmini test programs.

## What Is Here

- `gemmini.h`: main Gemmini API and instruction helpers
- `gemmini_nn.h`: neural-network helper functions built on top of Gemmini
- `gemmini_params*.h`: hardware parameter definitions for Gemmini configurations
- `gemmini_testutils.h`: test helpers
- `gemmini_counter.h`: performance counter helpers

## Why It Matters

This folder defines the software interface that Gemmini test code uses.

For ONNX-MLIR accelerator work, these headers help you understand:

- what low-level Gemmini operations exist
- what data types and memory assumptions the software stack uses
- what hardware parameters can affect code generation

## Mapping Used By ONNX-MLIR Gemmini Backend

The ONNX-MLIR Gemmini backend uses this folder as the bridge to Gemmini software APIs.

The mapping is:

1. ONNX operations such as `MatMulInteger` and `QLinearConv`
2. ONNX-MLIR runtime entry points such as `om_gemmini_matmulinteger_i8i8acc32`
3. Gemmini helper APIs from `gemmini.h` such as `tiled_matmul_auto`, `tiled_conv_auto`, and `gemmini_fence`
4. Low-level Gemmini instruction wrappers such as `gemmini_extended_mvin`, `gemmini_extended_mvout`, and `gemmini_extended_compute_*`
5. Raw RoCC instruction macros such as `ROCC_INSTRUCTION_*`

So ONNX-MLIR does not directly emit these raw Gemmini instruction macros in its frontend matching code.

Instead, the current backend flow is:

- match and legalize ONNX ops in `src/Accelerators/Gemmini/Conversion/ONNXToGemmini/`
- call runtime functions from `src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp`
- use `gemmini.h` helper functions
- let those helpers issue the final Gemmini instruction macros

This is why `include/` is important: it defines the API contract between the ONNX-MLIR Gemmini runtime layer and the actual Gemmini instruction layer.

## When You Will Need It

Use this folder when:

- reading `bareMetalC` examples
- matching generated code against known Gemmini helper calls
- checking hardware limits such as tile sizes or data types
- debugging Spike or bare-metal Gemmini execution

If `bareMetalC/` shows how Gemmini is used, `include/` shows the API contract behind that usage.
