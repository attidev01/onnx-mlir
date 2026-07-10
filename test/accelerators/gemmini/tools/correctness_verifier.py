#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
from pathlib import Path


def compare_json(candidate, reference, tolerance):
    cand = json.loads(Path(candidate).read_text(encoding="utf-8"))
    ref = json.loads(Path(reference).read_text(encoding="utf-8"))
    flat_cand = flatten(cand)
    flat_ref = flatten(ref)
    if len(flat_cand) != len(flat_ref):
        return False, {"reason": "length mismatch", "candidate": len(flat_cand), "reference": len(flat_ref)}
    max_abs = 0.0
    for c, r in zip(flat_cand, flat_ref):
        diff = abs(float(c) - float(r))
        max_abs = max(max_abs, diff)
        if diff > tolerance:
            return False, {"reason": "tolerance exceeded", "max_abs_error": max_abs}
    return True, {"max_abs_error": max_abs}


def flatten(value):
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out.extend(flatten(value[key]))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten(item))
        return out
    return [value]


def main():
    parser = argparse.ArgumentParser(description="Gemmini correctness verifier")
    parser.add_argument("--mode", choices=["metadata", "json"], default="json")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    if args.mode == "metadata":
        candidate = Path(args.candidate)
        reference = Path(args.reference)
        passed = candidate.exists() and reference.exists()
        details = {"candidate_size": candidate.stat().st_size if candidate.exists() else 0}
    else:
        passed, details = compare_json(args.candidate, args.reference, args.tolerance)

    print(json.dumps({"passed": passed, "tolerance": args.tolerance, "details": details}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
