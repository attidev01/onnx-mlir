#
# SPDX-License-Identifier: Apache-2.0
#

add_onnx_mlir_dialect(Gemmini gemmini)
add_onnx_mlir_dialect_doc(gemmini GemminiOps.td)

add_onnx_mlir_library(OMGemminiOps
  GemminiOps.cpp

  DEPENDS
    OMGemminiIncGen

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}
    ${MLIR_INCLUDE_DIRS}

  LINK_LIBS PUBLIC
    MLIRIR
)
