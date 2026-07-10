#
# SPDX-License-Identifier: Apache-2.0
#

#===------------------------------------------------------------------------===#
#
# Build file for Gemmini support utilities.
#
# This directory plays the same structural role as NNPA/Support/. It is the
# home for non-pass, non-dialect helper code shared across the backend, such as:
#   - tile-shape helpers
#   - scratchpad layout helpers
#   - libgemmini/runtime ABI helpers
#
# The support library keeps target constants and scheduling/layout helpers in a
# stable shared location.
#
#===------------------------------------------------------------------------===#

add_onnx_mlir_library(OMGemminiSupport
  GemminiTargetInfo.cpp

  INCLUDE_DIRS PUBLIC
    ${GEMMINI_SRC_ROOT}
    ${GEMMINI_BIN_ROOT}
    ${ONNX_MLIR_SRC_ROOT}

  LINK_LIBS PUBLIC
    MLIRIR
)
