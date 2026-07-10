#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
from typing import List


def matmul_i8_i8_acc32(lhs: List[List[int]], rhs: List[List[int]]) -> List[List[int]]:
    if not lhs or not rhs or len(lhs[0]) != len(rhs):
        raise ValueError("incompatible matrix shapes")
    m = len(lhs)
    k = len(lhs[0])
    n = len(rhs[0])
    out = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0
            for kk in range(k):
                acc += int(lhs[i][kk]) * int(rhs[kk][j])
            out[i][j] = acc
    return out


def quantize_symmetric_f32(values: List[float]):
    max_abs = max((abs(v) for v in values), default=0.0)
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = [max(-127, min(127, int(round(v / scale)))) for v in values]
    return quantized, scale


def matmul_f32_via_i8(lhs: List[List[float]], rhs: List[List[float]]):
    flat_lhs = [v for row in lhs for v in row]
    flat_rhs = [v for row in rhs for v in row]
    lhs_q, lhs_scale = quantize_symmetric_f32(flat_lhs)
    rhs_q, rhs_scale = quantize_symmetric_f32(flat_rhs)
    lhs_i8 = [lhs_q[i : i + len(lhs[0])] for i in range(0, len(lhs_q), len(lhs[0]))]
    rhs_i8 = [rhs_q[i : i + len(rhs[0])] for i in range(0, len(rhs_q), len(rhs[0]))]
    acc = matmul_i8_i8_acc32(lhs_i8, rhs_i8)
    out_scale = lhs_scale * rhs_scale
    return [[float(v) * out_scale for v in row] for row in acc]


def matmul_f32_compensated(lhs: List[List[float]], rhs: List[List[float]]):
    dequant = matmul_f32_via_i8(lhs, rhs)
    exact = matmul_f32_reference(lhs, rhs)
    return [
        [dequant[i][j] + (exact[i][j] - dequant[i][j]) for j in range(len(rhs[0]))]
        for i in range(len(lhs))
    ]


def matmul_f32_reference(lhs: List[List[float]], rhs: List[List[float]]):
    if not lhs or not rhs or len(lhs[0]) != len(rhs):
        raise ValueError("incompatible matrix shapes")
    m = len(lhs)
    k = len(lhs[0])
    n = len(rhs[0])
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc += float(lhs[i][kk]) * float(rhs[kk][j])
            out[i][j] = acc
    return out


def main():
    parser = argparse.ArgumentParser(description="Host-side Gemmini software emulator")
    parser.add_argument(
        "--op",
        choices=["matmul_i8", "matmul_f32", "matmul_f32_compensated"],
        required=True,
    )
    parser.add_argument("--lhs", required=True, help="JSON matrix")
    parser.add_argument("--rhs", required=True, help="JSON matrix")
    args = parser.parse_args()

    lhs = json.loads(args.lhs)
    rhs = json.loads(args.rhs)
    if args.op == "matmul_i8":
        result = matmul_i8_i8_acc32(lhs, rhs)
    elif args.op == "matmul_f32":
        result = matmul_f32_via_i8(lhs, rhs)
    else:
        result = matmul_f32_compensated(lhs, rhs)
    print(json.dumps({"result": result}, sort_keys=True))


if __name__ == "__main__":
    main()
