#
# SPDX-License-Identifier: Apache-2.0
#

# Build GemminiLow cleanup and allocation passes. These passes operate after
# tiling and before LLVM lowering, when scratchpad operations are explicit but
# still easy to rewrite as MLIR operations.
add_onnx_mlir_library(OMGemminiLowRewrite
  StaticScratchpadAllocation.cpp
  GemminiLowRewrite.cpp

  LINK_LIBS PUBLIC
    OMGemminiLowOps
    OMGemminiSupport
    MLIRIR
    MLIRPass
    MLIRFuncDialect
)
