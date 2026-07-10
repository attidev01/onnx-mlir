#
# SPDX-License-Identifier: Apache-2.0
#

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
