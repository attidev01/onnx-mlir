# SPDX-License-Identifier: Apache-2.0

# Build rules for the optional Gemmini runtime library. The runtime is only
# enabled for RISC-V builds because its ABI headers expand to raw RoCC
# instructions that host compilers cannot assemble.
set(GEMMINI_RUNTIME_ENABLED 0 PARENT_SCOPE)

# Path to the Chipyard/Gemmini repository root.
# Default: GEMMINI_SIM_PATH env var (set by setup_gemmini_simulator.sh) or
# ~/toolchains/gemmini as a last resort.  Override with -DGEMMINI_ROOT=<path>.
if (NOT GEMMINI_ROOT)
  if (DEFINED ENV{GEMMINI_SIM_PATH})
    set(GEMMINI_ROOT "$ENV{GEMMINI_SIM_PATH}" CACHE PATH
        "Gemmini repository root (Chipyard clone)")
  else()
    set(GEMMINI_ROOT "$ENV{HOME}/toolchains/gemmini" CACHE PATH
        "Gemmini repository root (Chipyard clone)")
  endif()
endif()

# Path to the Gemmini hardware ABI root/include directory.
# Prefer the bundled headers so the runtime can build from a standalone
# checkout. External paths are fallbacks for developers testing a different
# Gemmini hardware configuration.
if (NOT GEMMINI_ROCC_TESTS_DIR)
  set(GEMMINI_ROCC_TESTS_CANDIDATES
      "${CMAKE_CURRENT_SOURCE_DIR}/gemmini-hardware-abi"
      "${GEMMINI_ROOT}/software/gemmini-rocc-tests")
  if (ONNX_MLIR_SRC_ROOT)
    list(APPEND GEMMINI_ROCC_TESTS_CANDIDATES
        "${ONNX_MLIR_SRC_ROOT}/gemmini/lib/gemmini-rocc-tests")
  endif()
  list(APPEND GEMMINI_ROCC_TESTS_CANDIDATES
      "$ENV{HOME}/toolchains/gemmini-rocc-tests")

  foreach(candidate ${GEMMINI_ROCC_TESTS_CANDIDATES})
    if (EXISTS "${candidate}/include/gemmini.h" AND
        EXISTS "${candidate}/rocc-software/src/xcustom.h")
      set(GEMMINI_ROCC_TESTS_DIR "${candidate}" CACHE PATH
          "Gemmini hardware ABI root directory")
      break()
    endif()
  endforeach()
endif()
if (NOT GEMMINI_ROCC_TESTS_DIR AND GEMMINI_ROCC_INCLUDE_DIR)
  get_filename_component(GEMMINI_ROCC_TESTS_DIR
      "${GEMMINI_ROCC_INCLUDE_DIR}/.." ABSOLUTE)
endif()
if (NOT GEMMINI_ROCC_INCLUDE_DIR)
  set(GEMMINI_ROCC_INCLUDE_DIR
      "${GEMMINI_ROCC_TESTS_DIR}/include"
      CACHE PATH "Gemmini RoCC tests include directory")
endif()

# The Gemmini helper headers expand to raw RoCC instructions, so they can only
# be compiled when targeting RISC-V.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
  if (NOT EXISTS "${GEMMINI_ROCC_INCLUDE_DIR}/gemmini.h")
    message(FATAL_ERROR
      "Gemmini runtime build requires GEMMINI_ROCC_INCLUDE_DIR/gemmini.h to exist.\n"
      "  Tried: ${GEMMINI_ROCC_INCLUDE_DIR}\n"
      "  Set -DGEMMINI_ROOT=<chipyard-clone> or "
      "-DGEMMINI_ROCC_INCLUDE_DIR=<path> on the cmake command line.")
  endif()
  if (NOT EXISTS "${GEMMINI_ROCC_TESTS_DIR}/rocc-software/src/xcustom.h")
    message(FATAL_ERROR
      "Gemmini runtime build requires rocc-software/src/xcustom.h.\n"
      "  Tried: ${GEMMINI_ROCC_TESTS_DIR}\n"
      "  Set -DGEMMINI_ROCC_TESTS_DIR=<gemmini-hardware-abi-or-gemmini-rocc-tests> "
      "on the cmake command line.")
  endif()

  set(GEMMINI_RUNTIME_ENABLED 1 PARENT_SCOPE)

  add_onnx_mlir_library(RuntimeGemmini STATIC
    OMRuntimeGemmini.cpp

    EXCLUDE_FROM_OM_LIBS

    INCLUDE_DIRS PRIVATE
      ${GEMMINI_ROCC_TESTS_DIR}
      ${GEMMINI_ROCC_INCLUDE_DIR}
    )

  set_target_properties(RuntimeGemmini
    PROPERTIES
    LANGUAGE CXX
    POSITION_INDEPENDENT_CODE TRUE
    )
endif()
