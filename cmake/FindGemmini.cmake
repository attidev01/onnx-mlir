# SPDX-License-Identifier: Apache-2.0
#
# Locate the Gemmini headers and optional simulator/hardware tools used by the
# ONNX-MLIR Gemmini backend.
#
# Inputs:
#   GEMMINI_ROOT          Gemmini checkout root. Defaults to <repo>/gemmini.
#   RISCV_INSTALL         Spike/pk install prefix for simulator mode.
#
# Outputs:
#   GEMMINI_FOUND
#   GEMMINI_ROOT
#   GEMMINI_ROCC_INCLUDE_DIR
#   GEMMINI_SPIKE_EXECUTABLE
#   GEMMINI_PK_EXECUTABLE
#   GEMMINI_LIBGEMMINI
#   GEMMINI_RISCV_GCC
#   GEMMINI_HAS_ROCC_HEADERS
#   GEMMINI_HAS_SIMULATOR
#   GEMMINI_HAS_HARDWARE_TOOLCHAIN

include(FindPackageHandleStandardArgs)

if (Gemmini_ROOT AND NOT GEMMINI_ROOT)
  set(GEMMINI_ROOT "${Gemmini_ROOT}" CACHE PATH
    "Path to the Gemmini checkout used by ONNX-MLIR.")
endif()

if (NOT GEMMINI_ROOT)
  set(GEMMINI_ROOT "${CMAKE_CURRENT_LIST_DIR}/../gemmini" CACHE PATH
    "Path to the Gemmini checkout used by ONNX-MLIR.")
endif()

get_filename_component(GEMMINI_ROOT "${GEMMINI_ROOT}" ABSOLUTE)

find_path(GEMMINI_ROCC_INCLUDE_DIR
  NAMES gemmini.h gemmini_params.h
  PATHS
    "${GEMMINI_ROOT}/lib/gemmini-rocc-tests/include"
    "$ENV{GEMMINI_ROOT}/lib/gemmini-rocc-tests/include"
  NO_DEFAULT_PATH)

find_program(GEMMINI_RISCV_GCC
  NAMES riscv64-linux-gnu-gcc riscv64-unknown-elf-gcc
  PATHS
    "$ENV{RISCV}/bin"
    "$ENV{RISCV_INSTALL}/bin"
    "$ENV{PATH}")

find_program(GEMMINI_SPIKE_EXECUTABLE
  NAMES spike
  PATHS
    "$ENV{RISCV_INSTALL}/bin"
    "$ENV{RISCV}/bin")

find_program(GEMMINI_PK_EXECUTABLE
  NAMES pk
  PATHS
    "$ENV{RISCV_INSTALL}/bin"
    "$ENV{RISCV}/bin")

find_library(GEMMINI_LIBGEMMINI
  NAMES gemmini libgemmini
  PATHS
    "$ENV{RISCV_INSTALL}/lib"
    "$ENV{RISCV}/lib")

set(GEMMINI_HAS_ROCC_HEADERS FALSE)
if (GEMMINI_ROCC_INCLUDE_DIR)
  set(GEMMINI_HAS_ROCC_HEADERS TRUE)
endif()

set(GEMMINI_HAS_HARDWARE_TOOLCHAIN FALSE)
if (GEMMINI_HAS_ROCC_HEADERS AND GEMMINI_RISCV_GCC)
  set(GEMMINI_HAS_HARDWARE_TOOLCHAIN TRUE)
endif()

set(GEMMINI_HAS_SIMULATOR FALSE)
if (GEMMINI_HAS_ROCC_HEADERS AND GEMMINI_SPIKE_EXECUTABLE
    AND GEMMINI_PK_EXECUTABLE AND GEMMINI_LIBGEMMINI)
  set(GEMMINI_HAS_SIMULATOR TRUE)
endif()

find_package_handle_standard_args(Gemmini
  FOUND_VAR GEMMINI_FOUND
  REQUIRED_VARS GEMMINI_ROOT GEMMINI_ROCC_INCLUDE_DIR)

mark_as_advanced(
  GEMMINI_ROCC_INCLUDE_DIR
  GEMMINI_RISCV_GCC
  GEMMINI_SPIKE_EXECUTABLE
  GEMMINI_PK_EXECUTABLE
  GEMMINI_LIBGEMMINI)
