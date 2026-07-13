#
# SPDX-License-Identifier: Apache-2.0
#

#===------------------------------------------------------------------------===#
#
# Build file for ONNX-to-Gemmini conversion.
#
# This subdirectory mirrors NNPA/Conversion/ONNXToZHigh/: it is intended to
# hold the first accelerator-specific interception point where ONNX or
# ONNX-lowered patterns are recognized as Gemmini-compatible computations.
#
# Responsibilities:
#   - legality checks for Gemmini-supported ONNX ops
#   - runtime-delegate lowering to om_gemmini_* calls
#   - direct Gemmini dialect emission for supported hardware paths
#
#===------------------------------------------------------------------------===#

add_onnx_mlir_library(OMONNXToGemmini
  ONNXToGemmini.cpp

  LINK_LIBS PUBLIC
    OMGemminiOps
    OMGemminiSupport
    OMONNXOps
    OMONNXToKrnl
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRMemRefDialect
)
