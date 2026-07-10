# RUN: %python %s --tools-dir %gemmini_tools

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import sys
from pathlib import Path


def assert_close_matrix(actual, expected, tolerance, label):
    for i, row in enumerate(actual):
        for j, value in enumerate(row):
            diff = abs(value - expected[i][j])
            if diff > tolerance:
                raise SystemExit(
                    f"{label} mismatch at [{i},{j}]: actual={value} "
                    f"expected={expected[i][j]} diff={diff}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-dir", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.tools_dir)))

    from gemmini_emulator import (
        matmul_f32_compensated,
        matmul_f32_reference,
        matmul_f32_via_i8,
        quantize_symmetric_f32,
    )

    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    quantized, scale = quantize_symmetric_f32(values)
    dequantized = [q * scale for q in quantized]
    for actual, expected in zip(dequantized, values):
        if abs(actual - expected) > (scale / 2.0 + 1e-7):
            raise SystemExit(
                f"round-trip error exceeds half-LSB: actual={actual} "
                f"expected={expected} scale={scale}"
            )

    lhs = [[0.013, -0.027, 0.041], [0.055, -0.069, 0.083]]
    rhs = [[0.11, -0.12], [0.13, 0.14], [-0.15, 0.16]]
    exact = matmul_f32_reference(lhs, rhs)
    quantized_result = matmul_f32_via_i8(lhs, rhs)
    compensated = matmul_f32_compensated(lhs, rhs)

    max_quantized_error = max(
        abs(quantized_result[i][j] - exact[i][j])
        for i in range(len(exact))
        for j in range(len(exact[0]))
    )
    if not math.isfinite(max_quantized_error) or max_quantized_error == 0.0:
        raise SystemExit("expected non-zero finite quantized f32 matmul error")

    assert_close_matrix(compensated, exact, 1e-7, "compensated matmul")
    print("Gemmini quantization/dequantization round-trip tests passed")


if __name__ == "__main__":
    main()

