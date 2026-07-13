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

set(GEMMINI_PARAMS_HEADER
  "${GEMMINI_SRC_ROOT}/Runtime/gemmini-hardware-abi/include/gemmini_params.h")
set(GEMMINI_GENERATED_SUPPORT_DIR
  "${ONNX_MLIR_BIN_ROOT}/src/Accelerators/Gemmini/Support")

function(gemmini_read_define output_name define_name)
  file(STRINGS "${GEMMINI_PARAMS_HEADER}" _gemmini_define_line
    REGEX "^#define[ \t]+${define_name}[ \t]+[0-9]+")
  if (NOT _gemmini_define_line)
    message(FATAL_ERROR
      "Could not find #define ${define_name} in ${GEMMINI_PARAMS_HEADER}")
  endif()
  list(GET _gemmini_define_line 0 _gemmini_define_line)
  string(REGEX REPLACE
    "^#define[ \t]+${define_name}[ \t]+([0-9]+).*$" "\\1"
    _gemmini_define_value "${_gemmini_define_line}")
  set(${output_name} "${_gemmini_define_value}" PARENT_SCOPE)
endfunction()

function(gemmini_read_int_typedef_bits output_name typedef_name)
  file(STRINGS "${GEMMINI_PARAMS_HEADER}" _gemmini_typedef_line
    REGEX "^typedef[ \t]+int[0-9]+_t[ \t]+${typedef_name};")
  if (NOT _gemmini_typedef_line)
    message(FATAL_ERROR
      "Could not find intN_t typedef for ${typedef_name} in "
      "${GEMMINI_PARAMS_HEADER}")
  endif()
  list(GET _gemmini_typedef_line 0 _gemmini_typedef_line)
  string(REGEX REPLACE
    "^typedef[ \t]+int([0-9]+)_t[ \t]+${typedef_name};.*$" "\\1"
    _gemmini_typedef_bits "${_gemmini_typedef_line}")
  set(${output_name} "${_gemmini_typedef_bits}" PARENT_SCOPE)
endfunction()

gemmini_read_define(GEMMINI_TARGET_DIM DIM)
gemmini_read_define(GEMMINI_TARGET_BANK_NUM BANK_NUM)
gemmini_read_define(GEMMINI_TARGET_BANK_ROWS BANK_ROWS)
gemmini_read_define(GEMMINI_TARGET_ACC_ROWS ACC_ROWS)
gemmini_read_int_typedef_bits(GEMMINI_TARGET_ELEM_BITS elem_t)
gemmini_read_int_typedef_bits(GEMMINI_TARGET_ACC_BITS acc_t)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/GemminiTargetInfo.hpp.in"
  "${GEMMINI_GENERATED_SUPPORT_DIR}/GemminiTargetInfo.hpp"
  @ONLY)

add_onnx_mlir_library(OMGemminiSupport
  GemminiTargetInfo.cpp

  INCLUDE_DIRS PUBLIC
    ${GEMMINI_SRC_ROOT}
    ${ONNX_MLIR_BIN_ROOT}
    ${GEMMINI_BIN_ROOT}
    ${ONNX_MLIR_SRC_ROOT}

  LINK_LIBS PUBLIC
    MLIRIR
)
