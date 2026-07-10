#
# SPDX-License-Identifier: Apache-2.0
#

#===------------------------------------------------------------------------===#
#
# Build file for Gemmini-specific transform passes.
#
# This nested directory mirrors NNPA/Transform/ZHigh and NNPA/Transform/ZLow.
# It is intended to hold IR-to-IR passes that optimize Gemmini-targeted code
# after selection but before final LLVM lowering.
#
# Responsibilities:
#   - tiling and loop-nest construction
#   - scratchpad-aware load/store scheduling
#   - double-buffering and fence-aware scheduling support
#
#===------------------------------------------------------------------------===#

add_onnx_mlir_library(OMGemminiTiling
  GemminiTiling.cpp

  LINK_LIBS PUBLIC
    OMGemminiOps
    OMGemminiSupport
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRArithDialect
)
