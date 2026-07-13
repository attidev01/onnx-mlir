#
# SPDX-License-Identifier: Apache-2.0
#

#===------------------------------------------------------------------------===#
#
# Top-level build file for the Gemmini accelerator backend.
#
# This file follows the same role as NNPA's top-level accelerator CMake:
# it marks the accelerator as enabled, builds the backend-local libraries,
# and exposes the resulting accelerator library to the rest of ONNX-MLIR.
#
# Current Gemmini status:
#   - The accelerator registration layer is enabled.
#   - High-level and low-level Gemmini dialects are built.
#   - ONNX-to-Gemmini, tiling, low-level rewrite, and LLVM lowering passes are
#     wired into the backend.
#   - The optional runtime library is built only for RISC-V targets.
#
#===------------------------------------------------------------------------===#

# This cache variable is the feature flag the top-level build checks to know
# whether Gemmini support is available.
set(GEMMINI_ENABLED 1 PARENT_SCOPE)
set(GEMMINI_VERSION "3.0.0" CACHE STRING "Gemmini backend version" FORCE)
set(GEMMINI_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(GEMMINI_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(GEMMINI_BUILD_MODE "${ONNX_MLIR_GEMMINI_MODE}")

add_subdirectory(Compiler)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Runtime)

# Build the accelerator integration library. Use the ONNX-MLIR wrapper so the
# target is exported, installed, and included like the existing accelerator
# backends.
add_onnx_mlir_library(OMGemminiAccel
  GemminiAccelerator.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include
    ${ONNX_MLIR_SRC_ROOT}
    ${ONNX_MLIR_BIN_ROOT}
    ${GEMMINI_SRC_ROOT}
    ${GEMMINI_BIN_ROOT}

  LINK_LIBS PUBLIC
    OMGemminiCompilerUtils
    OMGemminiOps
    OMGemminiLowOps
    OMONNXToGemmini
    OMGemminiTiling
    OMGemminiToGemminiLow
    OMGemminiLowRewrite
    OMGemminiLowToLLVM
    OMGemminiSupport
    MLIRIR
    MLIRPass
    OMAccelerator
)

# Convenience target used by some local builds/scripts.
add_custom_target(GemminiAccel DEPENDS OMGemminiAccel)

# Compile-time feature define used by the rest of ONNX-MLIR to detect that
# this backend is part of the current build.
add_definitions(-DONNX_MLIR_WITH_GEMMINI)
if (GEMMINI_RUNTIME_ENABLED)
  add_definitions(-DONNX_MLIR_WITH_GEMMINI_RUNTIME)
endif()
if (GEMMINI_BUILD_MODE STREQUAL "hardware")
  add_definitions(-DONNX_MLIR_GEMMINI_HARDWARE_MODE)
elseif (GEMMINI_BUILD_MODE STREQUAL "simulator")
  add_definitions(-DONNX_MLIR_GEMMINI_SIMULATOR_MODE)
else()
  add_definitions(-DONNX_MLIR_GEMMINI_COMPILE_ONLY_MODE)
endif()
