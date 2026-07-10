#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re


COUNTS = {
    "gemmini_mvin": re.compile(r"gemmini(?:_low)?\.mvin"),
    "gemmini_mvout": re.compile(r"gemmini(?:_low)?\.mvout"),
    "gemmini_matmul": re.compile(r"gemmini(?:_low)?\.matmul|om_gemmini_.*matmul"),
    "gemmini_runtime_calls": re.compile(r'funcName = "om_gemmini_[^"]+"'),
}


def main():
    parser = argparse.ArgumentParser(description="Estimate Gemmini operation counts and cycles")
    parser.add_argument("mlir")
    args = parser.parse_args()
    text = open(args.mlir, encoding="utf-8").read()
    counts = {name: len(regex.findall(text)) for name, regex in COUNTS.items()}
    estimated_cycles = (
        counts["gemmini_mvin"] * 64
        + counts["gemmini_mvout"] * 64
        + counts["gemmini_matmul"] * 256
        + counts["gemmini_runtime_calls"] * 512
    )
    print(json.dumps({"counts": counts, "estimated_cycles": estimated_cycles}, sort_keys=True))


if __name__ == "__main__":
    main()
