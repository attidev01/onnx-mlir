#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re


MVIN_RE = re.compile(r"gemmini(?:_low)?\.mvin.* to ([0-9]+) rows tile\(([0-9]+) x ([0-9]+)\)")
MVOUT_RE = re.compile(r"gemmini(?:_low)?\.mvout.* from ([0-9]+) rows tile\(([0-9]+) x ([0-9]+)\)")
CALL_RE = re.compile(r'funcName = "(om_gemmini_[^"]+)"')
ACCEL_RE = re.compile(r'"onnx-mlir\.accels"\s*=\s*\[([^\]]*)\]')


def scan(path):
    text = open(path, encoding="utf-8").read()
    transfers = []
    violations = []
    for kind, regex in (("mvin", MVIN_RE), ("mvout", MVOUT_RE)):
        for match in regex.finditer(text):
            offset, rows, cols = map(int, match.groups())
            transfers.append({"kind": kind, "offset_rows": offset, "rows": rows, "cols": cols})
            if offset < 0 or offset + rows > 16384:
                violations.append(
                    {
                        "kind": kind,
                        "offset_rows": offset,
                        "rows": rows,
                        "reason": "scratchpad row range exceeds 16384 rows",
                    }
                )
    calls = CALL_RE.findall(text)
    accelerator_tags = []
    for match in ACCEL_RE.finditer(text):
        accelerator_tags.extend(
            tag.strip().strip('"') for tag in match.group(1).split(",") if tag.strip()
        )
    gemmini_ops = len(transfers) + len(calls)
    return {
        "file": path,
        "accelerator_enabled": any(tag.startswith("Gemmini") for tag in accelerator_tags),
        "accelerator_tags": accelerator_tags,
        "gemmini_ops": gemmini_ops,
        "runtime_calls": calls,
        "transfers": transfers,
        "violations": violations,
        "passed": not violations,
    }


def main():
    parser = argparse.ArgumentParser(description="Check Gemmini memory access patterns")
    parser.add_argument("mlir")
    args = parser.parse_args()
    report = scan(args.mlir)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
