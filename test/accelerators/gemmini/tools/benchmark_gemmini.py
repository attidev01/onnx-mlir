#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


def timed_compile(onnx_mlir, model, accelerator):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / ("model_gemmini" if accelerator else "model_cpu")
        cmd = [onnx_mlir, "--EmitMLIR", model, "-o", str(out)]
        if accelerator:
            cmd.insert(1, "--maccel=Gemmini")
        start = time.perf_counter()
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - start
        return {"ok": completed.returncode == 0, "seconds": elapsed, "log_lines": len(completed.stdout.splitlines())}


def main():
    parser = argparse.ArgumentParser(description="Compare Gemmini and CPU compile-time benchmark signals")
    parser.add_argument("--onnx-mlir", default="onnx-mlir")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    cpu = timed_compile(args.onnx_mlir, args.model, accelerator=False)
    gemmini = timed_compile(args.onnx_mlir, args.model, accelerator=True)
    print(json.dumps({"cpu": cpu, "gemmini": gemmini}, sort_keys=True))
    if not cpu["ok"] or not gemmini["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
