#
# SPDX-License-Identifier: Apache-2.0
#

# Build the final Gemmini lowering pass. This target depends on GemminiLow ops,
# compiler-visible hardware constants, and shared Krnl-to-LLVM helpers because
# it emits LLVM dialect plus RoCC inline-assembly call sequences.
add_onnx_mlir_library(OMGemminiLowToLLVM
  GemminiLowToLLVM.cpp

  LINK_LIBS PUBLIC
    OMGemminiLowOps
    OMGemminiSupport
    OMKrnlToLLVM
    MLIRIR
    MLIRPass
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
)
