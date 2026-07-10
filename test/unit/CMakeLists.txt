# SPDX-License-Identifier: Apache-2.0

if (TARGET llvm_gtest_main)
  add_custom_target(unittest)
  set_target_properties(unittest PROPERTIES FOLDER "Tests")

  add_custom_target(check-unittest
    COMMENT "Running the ONNX-MLIR unit tests"
    COMMAND "${CMAKE_CTEST_COMMAND}" -L unittest --output-on-failure -C $<CONFIG> --force-new-ctest-process
    USES_TERMINAL
    DEPENDS unittest
  )
  set_target_properties(check-unittest PROPERTIES FOLDER "Tests")
  set_target_properties(check-unittest PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)

  add_subdirectory(Gemmini)
else()
  message(STATUS "Gemmini unit tests skipped: llvm_gtest_main is unavailable in this build")
endif()
