#
# SPDX-License-Identifier: Apache-2.0
#

# Generate and build the GemminiLow dialect. These ops model explicit
# hardware-facing mvin, mvout, execute, and fence steps before LLVM lowering.
add_onnx_mlir_dialect(GemminiLow gemmini_low)
add_onnx_mlir_dialect_doc(gemmini_low GemminiLowOps.td)

add_onnx_mlir_library(OMGemminiLowOps
  GemminiLowOps.cpp

  DEPENDS
    OMGemminiLowIncGen

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}
    ${MLIR_INCLUDE_DIRS}

  LINK_LIBS PUBLIC
    MLIRIR
)
