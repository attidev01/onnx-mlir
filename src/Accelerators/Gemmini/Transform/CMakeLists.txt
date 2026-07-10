#
# SPDX-License-Identifier: Apache-2.0
#

#===------------------------------------------------------------------------===#
#
# Top-level build file for Gemmini transform passes.
#
# This directory is reserved for Gemmini-specific IR-to-IR rewrites that do
# not change the dialect boundary but improve the schedule or memory layout,
# similar to NNPA's Transform/ subtree.
#
# Expected future passes include:
#   - tiling and loop restructuring
#   - scratchpad-aware scheduling
#   - load/store pipelining
#
# The backend now uses separate high-level and low-level transform stages,
# mirroring NNPA's ZHigh/ZLow split.
#
#===------------------------------------------------------------------------===#

add_subdirectory(Gemmini)
add_subdirectory(GemminiLow)
