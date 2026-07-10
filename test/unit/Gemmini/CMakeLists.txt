# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestGemminiTiling
  NO_INSTALL
  TestGemminiTiling.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}

  LINK_LIBS PRIVATE
  llvm_gtest_main
  llvm_gtest
  OMGemminiSupport
)

add_dependencies(unittest TestGemminiTiling)
set_property(TARGET TestGemminiTiling PROPERTY FOLDER "Tests")

add_test(NAME TestGemminiTiling COMMAND TestGemminiTiling)
set_tests_properties(TestGemminiTiling PROPERTIES LABELS unittest)
