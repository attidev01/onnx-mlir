#
# SPDX-License-Identifier: Apache-2.0
#

add_onnx_mlir_library(OMGemminiLowToLLVM
  GemminiLowToLLVM.cpp

  LINK_LIBS PUBLIC
    OMGemminiLowOps
    OMKrnlToLLVM
    MLIRIR
    MLIRPass
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
)
