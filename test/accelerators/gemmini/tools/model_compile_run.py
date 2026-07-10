#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    completed = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        raise SystemExit(completed.returncode)
    return completed.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--onnx-mlir", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--verify-tools", required=True)
    parser.add_argument("--min-gemmini-ops", type=int, default=1)
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists() or model.stat().st_size == 0:
        raise SystemExit(f"missing or empty model: {model}")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output_base = workdir / f"{args.label}_gemmini"

    compile_log = run(
        [
            args.onnx_mlir,
            "--maccel=Gemmini",
            "--EmitMLIR",
            str(model),
            "-o",
            str(output_base),
        ]
    )

    mlir_file = output_base.with_suffix(".onnx.mlir")
    if not mlir_file.exists():
        raise SystemExit(f"expected MLIR output was not produced: {mlir_file}")

    checker = Path(args.verify_tools) / "memory_access_checker.py"
    memory_report = json.loads(run([sys.executable, str(checker), str(mlir_file)]))

    if not memory_report["accelerator_enabled"]:
        raise SystemExit(f"{args.label} did not enable the Gemmini accelerator")
    if memory_report["gemmini_ops"] < args.min_gemmini_ops:
        raise SystemExit(
            f"{args.label} produced {memory_report['gemmini_ops']} Gemmini calls, "
            f"expected at least {args.min_gemmini_ops}"
        )
    if not memory_report["passed"]:
        raise SystemExit(f"{args.label} memory access verification failed")

    print(
        json.dumps(
            {
                "model": str(model),
                "label": args.label,
                "mlir": str(mlir_file),
                "gemmini_ops": memory_report["gemmini_ops"],
                "compile_log_lines": len(compile_log.splitlines()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
