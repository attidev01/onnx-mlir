#
# SPDX-License-Identifier: Apache-2.0
#

#===------------------------------------------------------------------------===#
#
# Build file for Gemmini compiler-side utilities and options.
#
# This begins mirroring NNPA/Compiler/CMakeLists.txt more directly by exposing
# a backend-local utility library for pass sequencing and configuration.
#
#===------------------------------------------------------------------------===#

add_onnx_mlir_library(OMGemminiCompilerUtils
  GemminiCompilerUtils.cpp

  INCLUDE_DIRS PUBLIC
    ${GEMMINI_SRC_ROOT}
    ${GEMMINI_BIN_ROOT}
    ${ONNX_MLIR_SRC_ROOT}
    ${ONNX_MLIR_BIN_ROOT}

  LINK_LIBS PUBLIC
    OMONNXToGemmini
    OMGemminiTiling
    OMGemminiToGemminiLow
    OMGemminiLowRewrite
    OMGemminiLowToLLVM
    MLIRIR
    MLIRPass
)
