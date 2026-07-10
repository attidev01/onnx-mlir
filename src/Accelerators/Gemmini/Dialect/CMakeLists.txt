#
# SPDX-License-Identifier: Apache-2.0
#

#===------------------------------------------------------------------------===#
#
# Build file for the Gemmini dialect layers.
#
# The refactored backend exposes separate high-level and low-level dialect
# shells, following the same structural split as NNPA's ZHigh/ZLow layers.
#
#===------------------------------------------------------------------------===#

add_subdirectory(Gemmini)
add_subdirectory(GemminiLow)
