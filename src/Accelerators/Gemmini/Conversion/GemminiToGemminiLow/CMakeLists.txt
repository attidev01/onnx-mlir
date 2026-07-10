#
# SPDX-License-Identifier: Apache-2.0
#

add_onnx_mlir_library(OMGemminiToGemminiLow
  GemminiToGemminiLow.cpp

  LINK_LIBS PUBLIC
    OMGemminiOps
    OMGemminiLowOps
    MLIRIR
    MLIRPass
    MLIRFuncDialect
)
