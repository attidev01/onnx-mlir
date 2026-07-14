#
# SPDX-License-Identifier: Apache-2.0
#

# Build the lowering pass from scheduled Gemmini dialect ops to explicit
# GemminiLow instruction ops. This is where high-level accelerator work becomes
# visible memory movement, execute, and fence operations.
add_onnx_mlir_library(OMGemminiToGemminiLow
  GemminiToGemminiLow.cpp

  LINK_LIBS PUBLIC
    OMGemminiOps
    OMGemminiLowOps
    MLIRIR
    MLIRPass
    MLIRFuncDialect
)
