# Gemmini Runtime Files Explained

This document explains every file under `src/Accelerators/Gemmini/Runtime`.
The directory is the runtime bridge between ONNX-MLIR generated calls and the
Gemmini RISC-V accelerator software ABI.

## Folder Overview And Relationships

### `Runtime/`

Top-level ONNX-MLIR Gemmini runtime folder.

This folder contains the runtime source code that compiled ONNX-MLIR models
link against. The compiler lowers supported ONNX operations into calls such as
`om_gemmini_matmul_f32_hw` or `om_gemmini_qlinearconv_i8`; those symbols are
declared in `OMRuntimeGemmini.hpp` and implemented in `OMRuntimeGemmini.cpp`.

`CMakeLists.txt` decides whether the runtime can be built. Because Gemmini uses
RISC-V RoCC custom instructions, this runtime is only enabled for RISC-V
targets.

### `Runtime/gemmini-hardware-abi/`

Bundled Gemmini hardware ABI folder.

This folder is not ONNX-MLIR compiler logic. It is the small set of Gemmini and
RoCC headers needed to compile the runtime without requiring a separate local
Gemmini checkout. It defines the software contract for the exact Gemmini
hardware configuration that the runtime is compiled against.

The top-level runtime code depends on this folder through:

```c++
#include "gemmini.h"
```

`OMRuntimeGemmini.cpp` calls helper APIs from `gemmini.h`, such as
`tiled_matmul_auto`, `tiled_conv_auto`, and `gemmini_fence`.

### `Runtime/gemmini-hardware-abi/include/`

Gemmini API and parameter header folder.

This folder contains the higher-level Gemmini C API. The most important files
are `gemmini.h`, which exposes the matmul/conv/data-movement helper functions,
and `gemmini_params.h`, which defines the hardware shape and data types:
array dimension, scratchpad size, accumulator size, `elem_t`, `acc_t`, scaling
types, and feature macros.

Relationship to the runtime:

1. `OMRuntimeGemmini.cpp` calls Gemmini helper functions.
2. Those helper functions are declared/defined in `include/gemmini.h`.
3. `gemmini.h` uses `include/gemmini_params.h` to know what hardware profile it
   is targeting.
4. `gemmini.h` uses RoCC macros from `rocc-software/src/xcustom.h` to emit the
   final custom RISC-V instructions.

### `Runtime/gemmini-hardware-abi/rocc-software/`

Vendored Rocket Custom Coprocessor support folder.

RoCC is the RISC-V coprocessor interface used by Gemmini. This folder contains
upstream documentation, license text, contribution metadata, and the `src`
subfolder with instruction-emission headers.

The ONNX-MLIR runtime does not call files in this folder directly. Instead,
`gemmini.h` includes the RoCC macros it needs.

### `Runtime/gemmini-hardware-abi/rocc-software/src/`

Lowest-level RoCC instruction macro folder.

This folder contains `xcustom.h`, which defines `ROCC_INSTRUCTION*` macros.
Those macros expand to inline assembly or raw RISC-V custom instructions.
It also contains `riscv_test_rocc.h`, which is useful for low-level RoCC tests.

The dependency direction is:

```text
compiled ONNX-MLIR model
  -> OMRuntimeGemmini.hpp / OMRuntimeGemmini.cpp
  -> gemmini-hardware-abi/include/gemmini.h
  -> gemmini-hardware-abi/include/gemmini_params.h
  -> gemmini-hardware-abi/rocc-software/src/xcustom.h
  -> RISC-V RoCC custom instructions
  -> Gemmini hardware or simulator
```

So the top-level `Runtime/` folder is the ONNX-MLIR-facing layer, `include/` is
the Gemmini-facing C helper layer, and `rocc-software/src/` is the raw
RISC-V-custom-instruction layer.

## Top-Level Runtime Files

### `CMakeLists.txt`

Build configuration for the Gemmini runtime library.

The file starts by disabling the runtime by default with
`GEMMINI_RUNTIME_ENABLED 0`. It then locates the Gemmini hardware ABI headers.
The search order is:

1. The bundled `gemmini-hardware-abi` directory in this runtime folder.
2. `${GEMMINI_ROOT}/software/gemmini-rocc-tests`.
3. `${ONNX_MLIR_SRC_ROOT}/gemmini/lib/gemmini-rocc-tests`, when available.
4. `$HOME/toolchains/gemmini-rocc-tests`.

`GEMMINI_ROOT` defaults to `$GEMMINI_SIM_PATH` when that environment variable is
set, otherwise to `$HOME/toolchains/gemmini`.

The important build rule is guarded by:

```cmake
if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
```

That is because `gemmini.h` expands into raw RoCC RISC-V custom instructions.
The runtime is only compiled for RISC-V targets. On RISC-V, CMake verifies that
`gemmini.h` and `rocc-software/src/xcustom.h` exist, enables the runtime, and
adds a static `RuntimeGemmini` library from `OMRuntimeGemmini.cpp`.

CMake variable reference:

- `GEMMINI_RUNTIME_ENABLED`: parent-scope flag indicating whether the runtime
  library was added for the current target.
- `GEMMINI_ROOT`: Chipyard/Gemmini repository root used as a fallback source for
  Gemmini RoCC headers.
- `GEMMINI_SIM_PATH`: environment variable used to initialize `GEMMINI_ROOT`
  when available.
- `GEMMINI_ROCC_TESTS_CANDIDATES`: ordered list of possible ABI roots.
- `GEMMINI_ROCC_TESTS_DIR`: selected Gemmini ABI root containing both
  `include/gemmini.h` and `rocc-software/src/xcustom.h`.
- `GEMMINI_ROCC_INCLUDE_DIR`: selected include directory containing
  `gemmini.h`.
- `ONNX_MLIR_SRC_ROOT`: optional ONNX-MLIR source root used to add another ABI
  search candidate.
- `candidate`: loop variable for each ABI directory probe.

### `FILES_EXPLAINED.md`

Human-readable inventory of the Gemmini runtime directory.

This is the file you are reading. It summarizes the purpose and contents of
each runtime source file, build file, documentation file, and vendored ABI
header under `src/Accelerators/Gemmini/Runtime`.

### `OMRuntimeGemmini.hpp`

Public C ABI header for the Gemmini runtime entry points.

Compiled ONNX-MLIR models call these functions through lowered `krnl.call`
symbols, so the header exposes all functions with `extern "C"` to avoid C++
name mangling. It includes `OMTensor.h`, because every runtime entry point
receives preallocated ONNX-MLIR runtime tensors as inputs and outputs.

The declarations are grouped by operation family:

- Integer and quantized hardware functions:
  `om_gemmini_matmulinteger_i8i8acc32`,
  `om_gemmini_matmulinteger_i8i8acc32_zp`,
  `om_gemmini_qlinearconv_i8`, and
  `om_gemmini_qlinearconv_i8_bias`.
- Float matrix and convolution functions that use the int8 Gemmini path by
  quantizing, running hardware, then converting back.
- CPU scalar elementwise and normalization helpers such as ReLU, Add, Mul,
  Sigmoid, Softmax, and BatchNormalization.
- CPU scalar pooling helpers.
- GEMM wrappers.
- Spatial/layout helpers such as Resize, Pad, Slice, Concat, Transpose, and
  Split.

The header is the contract between compiler lowering and the runtime
implementation. If the compiler emits a call to an `om_gemmini_*` symbol, the
symbol should appear here and be implemented in `OMRuntimeGemmini.cpp`.

### `OMRuntimeGemmini.cpp`

Main Gemmini runtime implementation.

This file is intentionally RISC-V-only:

```c++
#if !defined(__riscv)
#error "OMRuntimeGemmini.cpp must only be compiled for RISC-V targets"
#endif
```

It includes ONNX-MLIR tensor APIs, small float conversion helpers, and
`gemmini.h`. The public functions are the symbols declared in
`OMRuntimeGemmini.hpp`; private `static` helpers perform shape checks,
quantization, packing, indexing, and scalar fallback work.

Main content by section:

- Helpers:
  validate optional zero-point tensors, load scalar values from `OMTensor`,
  decode float bit patterns passed through integer attributes, and compute
  indices for flattened leading dimensions.
- Integer / quantized path:
  wraps Gemmini `tiled_matmul_auto` in weight-stationary mode for
  int8-by-int8 matrix multiplication with int32 accumulation.
  The zero-point variant first runs hardware matmul and then applies the ONNX
  zero-point correction in scalar code.
- Quantized convolution path:
  implements `QLinearConv` on NCHW int8 tensors. Weights are repacked from
  OIHW to the layout expected by Gemmini helpers, bias and input zero-point
  correction are folded into an adjusted bias buffer, `tiled_conv_auto` is
  called, and the temporary NHWC result is copied back to NCHW.
- Float matmul path:
  because this Gemmini configuration has an int8 systolic path, f32 and f16
  matmul are dynamically quantized to int8, executed through Gemmini, and then
  dequantized. For f32, a scalar residual correction is added so the final
  output preserves ONNX float semantics while still exercising Gemmini.
- Float convolution and transposed convolution:
  implemented as scalar NCHW loops for f32. The functions cover optional bias,
  stride, padding, and output padding for transpose convolution.
- Elementwise and normalization:
  scalar implementations of ReLU, BatchNormalization, Add, Sigmoid, Mul, and
  Softmax over `OMTensor` buffers.
- Pooling:
  scalar implementations of GlobalAveragePool, MaxPool, and AveragePool.
- GEMM:
  handles transposed inputs, `alpha` and `beta` decoded from f32 bit patterns,
  and optional C bias matrix.
- Spatial and layout operations:
  scalar implementations of nearest and linear resize, constant/reflect/edge
  padding, rank-4 slicing, two-input concat, rank-4 transpose, and split into
  2, 3, or 4 outputs.

In short, this file is both the real Gemmini hardware dispatch layer for the
supported int8 operations and the compatibility layer for surrounding ONNX ops
that compiled models still need at runtime.

#### Function Reference

Trace helpers:

- `gemminiTraceEnabled`: checks `GEMMINI_TRACE`; tracing is disabled by default
  and enabled only when the variable is set to a non-empty value other than `0`.
- `traceTensorDTypeName`: converts ONNX-MLIR dtype ids to short names such as
  `f32`, `f16`, `i8`, and `i32`.
- `traceTensorShape`: prints a tensor's dtype, shape, and element count.
- `traceTensorStats`: prints min, max, mean, nonzero count, NaN count, and Inf
  count for supported tensor types.
- `traceTensorShapeStats`: prints both shape and scalar statistics.
- `traceF32BufferStats`: prints scalar statistics for a raw float buffer.
- `traceConvF32Shape`: prints the compact NCHW convolution shape summary.

Scalar decode and zero-point helpers:

- `isNoneScalarOrVectorI8Tensor`: validates optional int8 zero-point tensors.
- `loadOptionalScalarI8`: reads an optional scalar int8 zero-point, defaulting
  to zero when absent.
- `loadScalarF32`: reads a required scalar float32 tensor.
- `getZeroPointSpan`: distinguishes scalar zero-points from vector zero-points.
- `getZeroPointValue`: reads the scalar or indexed vector zero-point.
- `decodeF32Bits`: reconstructs a float32 value passed as raw bits in an int64.

Gemmini matmul and quantization helpers:

- `om_gemmini_tiled_matmul_i8i8acc32_ws`: calls Gemmini `tiled_matmul_auto` in
  weight-stationary mode for int8-by-int8 accumulation into int32.
- `applyZeroPointCorrection`: applies ONNX MatMulInteger zero-point correction
  after raw hardware matmul.
- `quantizeSymmetricI8`: maps a float value to signed int8 using symmetric
  scaling and saturation.
- `dotProductF32`: computes a strided scalar float dot product.
- `dequantizedAccF32`: rescales an int32 Gemmini accumulator back to float32.
- `flattenedLeadingOffset`: maps a flattened leading-index back to a physical
  tensor memory offset.
- `om_gemmini_quantized_matmul_f32_ws`: quantizes f32 matrices, runs Gemmini
  int8 matmul, and returns the accumulator plus combined output scale.
- `maxAbsF16`: computes the maximum absolute value of f16 data after conversion
  to f32.

Layout and convolution helpers:

- `packOIHWtoHWOI`: repacks quantized convolution weights from ONNX OIHW layout
  to Gemmini's HWOI layout.
- `copyNHWCtoNCHW`: converts Gemmini NHWC convolution output back to ONNX NCHW.
- `buildAdjustedConvBias`: folds input zero-point correction into int32
  per-output-channel bias.
- `om_gemmini_qlinearconv_i8_impl`: shared QLinearConv implementation for bias
  and no-bias wrappers.
- `om_gemmini_conv_f32_impl`: shared f32 Conv implementation for bias and
  no-bias wrappers.
- `om_gemmini_convtranspose_f32_impl`: shared f32 ConvTranspose implementation
  for bias and no-bias wrappers.
- `om_gemmini_gemm_f32_impl`: shared f32 Gemm implementation for bias and
  no-bias wrappers.

Resize, pad, transpose, and split helpers:

- `resize_input_coord`: maps an output coordinate to an input coordinate using
  ONNX Resize coordinate transformation modes.
- `resize_nearest_coord`: rounds a floating coordinate according to ONNX
  nearest-neighbor modes.
- `resize_clamp`: clamps integer coordinates into a valid range.
- `pad_clamp_index`: implements edge padding index selection.
- `pad_reflect_index`: implements reflect padding index selection.
- `om_gemmini_transpose_f32_scalar_impl`: scalar rank-4 transpose fallback.
- `om_gemmini_transpose_perm_has_hw_route`: recognizes permutations that could
  use a future standalone Gemmini transposer route.
- `om_gemmini_transposer_hw_available`: detects whether that standalone
  transposer API is exposed by the ABI headers.
- `om_gemmini_split_copy_slice_f32`: copies one split slice from a rank-4 input
  tensor into an output tensor.

Public runtime entry points:

- `om_gemmini_matmulinteger_i8i8acc32`: MatMulInteger int8 x int8 to int32,
  no zero-points.
- `om_gemmini_matmulinteger_i8i8acc32_zp`: MatMulInteger with optional
  zero-point correction.
- `om_gemmini_qlinearconv_i8`: QLinearConv int8, no bias.
- `om_gemmini_qlinearconv_i8_bias`: QLinearConv int8 with int32 bias.
- `om_gemmini_matmul_f32_hw`: rank-2 f32 MatMul using Gemmini int8 hardware
  plus f32 residual correction.
- `om_gemmini_matmul_f32_nd_hw`: batched rank-N x rank-2 f32 MatMul.
- `om_gemmini_matmul_f16_hw`: rank-2 f16 MatMul through the quantized Gemmini
  path.
- `om_gemmini_conv_f32` and `om_gemmini_conv_f32_bias`: f32 Conv wrappers.
- `om_gemmini_convtranspose_f32` and `om_gemmini_convtranspose_f32_bias`: f32
  ConvTranspose wrappers.
- `om_gemmini_relu_f32`: element-wise ReLU.
- `om_gemmini_batchnorm_f32`: inference-mode BatchNormalization.
- `om_gemmini_add_f32`: flattened element-wise Add.
- `om_gemmini_globalavgpool_f32`: NCHW GlobalAveragePool.
- `om_gemmini_maxpool_f32`: NCHW MaxPool.
- `om_gemmini_avgpool_f32`: NCHW AveragePool.
- `om_gemmini_softmax_f32`: stable softmax over `batch x classes`.
- `om_gemmini_gemm_f32` and `om_gemmini_gemm_f32_bias`: ONNX Gemm wrappers.
- `om_gemmini_resize_nearest_f32` and `om_gemmini_resize_linear_f32`: NCHW
  nearest and bilinear resize.
- `om_gemmini_pad_constant_f32`, `om_gemmini_pad_reflect_f32`, and
  `om_gemmini_pad_edge_f32`: NCHW Pad modes.
- `om_gemmini_slice_f32`: rank-4 strided Slice.
- `om_gemmini_concat_f32`: two-input rank-4 Concat.
- `om_gemmini_transpose_f32_hw`: HW-preferred rank-4 Transpose with scalar
  fallback.
- `om_gemmini_split_2_f32`, `om_gemmini_split_3_f32`, and
  `om_gemmini_split_4_f32`: rank-4 Split wrappers.
- `om_gemmini_sigmoid_f32`: element-wise Sigmoid.
- `om_gemmini_mul_f32`: flattened element-wise Multiply.

#### Variable Name Reference

The implementation uses short names that mirror tensor math notation:

- `x`, `input`: input activation tensor.
- `w`, `weights`: weight/filter tensor.
- `b`, `bias`, `c`: optional bias or Gemm C tensor.
- `out`, `output`, `y`: pre-allocated output tensor.
- `a`, `lhs`, `A`: left operand of MatMul/Gemm.
- `b`, `rhs`, `B`: right operand of MatMul/Gemm; when the function also has a
  bias, the code uses names such as `biasData` to avoid ambiguity.
- `N`: batch size in convolution/pooling code; row count in GEMM-style code
  only when following `[M, K] x [K, N]`.
- `C`, `inChannels`, `ic`: input channel count and loop index.
- `OC`, `M`, `outChannels`, `oc`: output channel count and loop index.
- `H`, `W`: input height and width.
- `OH`, `OW`: output height and width.
- `KH`, `KW`, `kernelDim`: kernel height, width, or square kernel size.
- `M`, `K`, `N` in matmul helpers: matrix dimensions for
  `[M, K] x [K, N] -> [M, N]`.
- `stride`, `pad`, `padding`, `output_pad`: spatial operator attributes.
- `shape`, `strides`: metadata returned by `OMTensor` accessors.
- `src`, `dst`, `din`, `dout`: source and destination memory buffers.
- `idx`, `in_idx`, `out_idx`: flattened row-major memory indices.
- `n`, `c`, `h`, `w`: NCHW loop variables.
- `oh`, `ow`, `ih`, `iw`: output/input spatial loop variables.
- `kh`, `kw`: kernel spatial loop variables.
- `scaleA`, `scaleB`, `invScaleA`, `invScaleB`, `outputScale`: dynamic
  quantization scales for float-to-int8 Gemmini matmul.
- `aQ`, `bQ`: temporary quantized int8 matrix buffers.
- `acc`: int32 accumulator buffer produced by Gemmini.
- `im2col`, `input_flat`, `weight_flat`: temporary packed matrices used to
  express convolution as matrix multiplication.
- `packedWeights`, `tempOutput`, `adjustedBias`: temporary quantized
  convolution buffers for Gemmini layout and bias correction.
- `rowSumsA`, `colSumsB`: sums used by MatMulInteger zero-point correction.
- `perm`, `outCoord`, `inCoord`: Transpose permutation and coordinates.
- `off0`, `off1`, `off2`, `off3`: split/concat offsets along the selected
  axis.

## Bundled Gemmini Hardware ABI

### `gemmini-hardware-abi/README.md`

Explains why this runtime vendors a small Gemmini ABI bundle.

The key idea is that ONNX-MLIR only needs the headers required to compile the
Gemmini runtime, not a complete checkout of `gemmini-rocc-tests`. The most
important bundled generated header is `include/gemmini_params.h`, which fixes
the Gemmini hardware configuration used by the runtime build.

It also documents how developers can override the bundled ABI with
`-DGEMMINI_ROCC_TESTS_DIR=...` or `-DGEMMINI_ROCC_INCLUDE_DIR=...` when testing
against a different Gemmini hardware configuration.

### `gemmini-hardware-abi/include/README.md`

Documentation for the vendored `include` directory.

It explains the relationship between ONNX operations, ONNX-MLIR runtime
functions, Gemmini helper APIs such as `tiled_matmul_auto` and
`tiled_conv_auto`, and the lower-level RoCC instruction macros. This is useful
when tracing a compiled model call all the way down to the custom RISC-V
instruction layer.

### `gemmini-hardware-abi/include/accumulator.h`

Small helper header for an accumulator custom coprocessor interface.

It defines operation codes such as `k_DO_WRITE`, `k_DO_READ`, `k_DO_LOAD`, and
`k_DO_ACCUM`, then maps convenience macros like `doWrite` and `doAccum` to
`ROCC_INSTRUCTION`. This file is part of the wider RoCC/Gemmini software ABI;
the ONNX-MLIR runtime does not call these macros directly.

### `gemmini-hardware-abi/include/character.h`

Minimal header defining the custom instruction slot for a character device:
`XCUSTOM_CHAR 2`.

It includes `xcustom.h` so code can emit RoCC custom instructions. It is
vendored as part of the ABI header set, but it is not a central part of the
current ONNX-MLIR Gemmini runtime.

### `gemmini-hardware-abi/include/gemmini.h`

Main Gemmini software API header.

This is the largest vendored header. It includes the generated hardware
parameters from `gemmini_params.h`, imports RoCC instruction macros from
`xcustom.h`, defines Gemmini command/function codes, and provides helper macros
and functions for moving data, configuring Gemmini, launching matmul/conv
loops, reading counters, and running CPU reference fallbacks.

Important APIs used by `OMRuntimeGemmini.cpp` include:

- `tiled_matmul_auto`, used for int8 matmul and the quantized f32/f16 matmul
  path.
- `tiled_conv_auto`, used for quantized convolution.
- `gemmini_fence`, used after Gemmini work to enforce completion ordering.
- type names such as `elem_t`, `acc_t`, and `acc_scale_t`, which come through
  `gemmini_params.h`.

Most of this file is vendor logic from Gemmini. ONNX-MLIR treats it as the
hardware ABI layer rather than modifying it for each operation.

### `gemmini-hardware-abi/include/gemmini_counter.h`

Defines numeric IDs for Gemmini performance counters.

The names cover load/store/execute cycles, DMA activity, TLB events,
scratchpad and accumulator waits, im2col activity, reservation station events,
loop matmul activity, and data movement byte/latency counters. `gemmini.h`
uses these IDs in its counter access helpers.

### `gemmini-hardware-abi/include/gemmini_nn.h`

Neural-network helper routines built on top of `gemmini.h`.

It defines parameter structs such as `ConvParams` and `FcParams`, histogram
macros, tiled matmul convenience wrappers, depthwise convolution helpers,
im2col helpers, vector add, residual add, and pooling helpers. The current
ONNX-MLIR runtime primarily calls lower-level `gemmini.h` APIs directly, but
this file documents and provides higher-level Gemmini NN building blocks.

### `gemmini-hardware-abi/include/gemmini_params.h`

Generated Gemmini hardware configuration used by the bundled ABI.

This is one of the most important files in the bundle because it defines the
compile-time contract between software and a particular Gemmini hardware
configuration:

- `DIM 16`
- scratchpad and accumulator sizes such as `BANK_NUM`, `BANK_ROWS`, and
  `ACC_ROWS`
- maximum DMA block lengths
- element, accumulator, and full-precision types:
  `elem_t` is `int8_t`, `acc_t` is `int32_t`, and `full_t` is `int64_t`
- scaling types such as `scale_t` and `acc_scale_t`
- rounding and scaling macros
- feature macros such as `HAS_MVIN_SCALE`, `ACC_READ_SMALL_WIDTH`,
  `ACC_READ_FULL_WIDTH`, and `HAS_FIRST_LAYER_OPTIMIZATIONS`

If the Gemmini hardware changes, this header must match the generated headers
for that hardware.

### `gemmini-hardware-abi/include/gemmini_params_ee290.h`

Alternate Gemmini parameter header for an EE290-style configuration.

Here, `EE290` refers to UC Berkeley's EE290 hardware accelerator / advanced
computer architecture coursework context where Gemmini configurations are often
used for class projects and experiments. In this file name, it marks the header
as an alternate educational/experimental Gemmini hardware profile rather than
the default bundled runtime profile.

It defines a 32-wide array (`DIM 32`), smaller scratchpad and accumulator
dimensions than the main bundled `gemmini_params.h`, and the same basic int8
element / int32 accumulator type model. Because it uses the same include guard
name as `gemmini_params.h`, it is meant as an alternate selected header, not a
header to include together with the main one.

### `gemmini-hardware-abi/include/gemmini_params_ee290_smallsp.h`

Another alternate EE290-style parameter header.

It is similar to `gemmini_params_ee290.h`, but with an even smaller scratchpad
and accumulator configuration: `BANK_ROWS 1024` and `ACC_ROWS 256`. Like the
other alternate params file, it represents a different hardware profile and is
not included alongside the default `gemmini_params.h`.

### `gemmini-hardware-abi/include/gemmini_testutils.h`

Test utility header for Gemmini programs.

It provides CPU-side reference helpers for matmul variants, matrix addition,
matrix scaling, ReLU, transpose, printing, equality checks, and cycle reads.
It also replaces `assert` in bare-metal builds with a printf-and-exit version.

This file is mainly useful for Gemmini tests and debugging. It is vendored with
the ABI because other Gemmini helper headers may include or expect it.

### `gemmini-hardware-abi/include/translator.h`

Small helper header for a translator custom coprocessor interface.

It defines `XCUSTOM_TRANS 1` and a `doTranslate` macro that emits a RoCC
instruction through `ROCC_INSTRUCTION`. It is part of the broader RoCC support
bundle, not a direct ONNX-MLIR runtime dependency.

### `gemmini-hardware-abi/rocc-software/CONTRIBUTING.md`

Contribution policy text for the vendored RoCC software.

It records the Developer Certificate of Origin sign-off requirement used by
the upstream RoCC software project. It is documentation/legal metadata rather
than runtime code.

### `gemmini-hardware-abi/rocc-software/LICENSE`

Apache License 2.0 text for the vendored RoCC software.

This file gives the license terms for the RoCC software headers included in the
ABI bundle.

### `gemmini-hardware-abi/rocc-software/README.md`

Short upstream README for Rocket Custom Coprocessor software.

It states that the directory contains C and RISC-V assembly macros for emitting
custom RISC-V instructions used to communicate with RoCC accelerators.

### `gemmini-hardware-abi/rocc-software/src/riscv_test_rocc.h`

RISC-V test macro header for enabling RoCC state in low-level tests.

It defines `RVTEST_XS_ENABLE` and `RVTEST_WITH_ROCC`, which set the XS field in
`mstatus` during RISC-V test initialization. This is useful for bare-metal or
assembly tests that need RoCC enabled before issuing custom instructions.

### `gemmini-hardware-abi/rocc-software/src/xcustom.h`

Core RoCC custom instruction macro header.

This file defines the macros that ultimately emit custom RISC-V instructions.
It supports raw assembly macros and C/C++ inline assembly forms, including
instructions with destination registers, source registers, and immediate-style
operands.

Higher-level Gemmini macros in `gemmini.h` eventually bottom out in these
`ROCC_INSTRUCTION*` macros. In the runtime stack, this is the lowest-level
software layer before the generated RISC-V custom instruction reaches the
Gemmini hardware or simulator.

## Runtime Call Flow Summary

The practical path for a compiled ONNX operation is:

1. The Gemmini compiler lowering chooses an `om_gemmini_*` runtime symbol.
2. The generated model passes `OMTensor` pointers to that runtime symbol.
3. `OMRuntimeGemmini.cpp` checks shapes and data types, prepares layouts or
   quantized staging buffers, and either calls Gemmini or runs scalar loops.
4. Hardware-backed functions call `tiled_matmul_auto`, `tiled_conv_auto`, or
   related helpers from `gemmini.h`.
5. `gemmini.h` emits RoCC custom instructions through `xcustom.h`.
6. The output `OMTensor` buffer is filled in the layout expected by ONNX-MLIR.
