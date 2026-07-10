#
# SPDX-License-Identifier: Apache-2.0
#

#===------------------------------------------------------------------------===#
#
# Top-level build file for Gemmini conversion passes.
#
# This directory hosts the staged lowerings for the refactored Gemmini
# pipeline:
#   - ONNXToGemmini/
#   - GemminiToGemminiLow/
#   - GemminiLowToLLVM/
#
#===------------------------------------------------------------------------===#

add_subdirectory(ONNXToGemmini)
add_subdirectory(GemminiToGemminiLow)
add_subdirectory(GemminiLowToLLVM)
