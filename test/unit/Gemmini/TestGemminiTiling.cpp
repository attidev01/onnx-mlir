/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtest/gtest.h"

#include "src/Accelerators/Gemmini/Support/GemminiTargetInfo.hpp"

using namespace onnx_mlir::gemmini;

TEST(GemminiTiling, ExactTileCountsStayUnit) {
  auto counts = GemminiTargetInfo::getMatmulTileCounts(16, 16, 16);
  EXPECT_EQ(counts.mTiles, 1);
  EXPECT_EQ(counts.nTiles, 1);
  EXPECT_EQ(counts.kTiles, 1);
}

TEST(GemminiTiling, NonMultipleDimensionsRoundUp) {
  auto counts = GemminiTargetInfo::getMatmulTileCounts(17, 31, 33);
  EXPECT_EQ(counts.mTiles, 2);
  EXPECT_EQ(counts.nTiles, 2);
  EXPECT_EQ(counts.kTiles, 3);
}

TEST(GemminiTiling, ScratchpadProfileMatchesCompilerModel) {
  EXPECT_EQ(GemminiTargetInfo::dim, 16);
  EXPECT_EQ(GemminiTargetInfo::scratchpadRows(), 256);
  EXPECT_EQ(GemminiTargetInfo::accRows, 64);
  EXPECT_EQ(GemminiTargetInfo::scratchpadMatrices(), 16);
  EXPECT_EQ(GemminiTargetInfo::accumulatorMatrices(), 4);
}
