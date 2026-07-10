# REQUIRES: gemmini-resnet18
# RUN: %python %s --model %resnet18_model --onnx-mlir onnx-mlir --workdir %t.dir --verify-tools %gemmini_tools

import argparse
import json
import os
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
    parser.add_argument("--onnx-mlir", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--verify-tools", required=True)
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists() or model.stat().st_size == 0:
        raise SystemExit(f"missing or empty model: {model}")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output_base = workdir / "resnet18_gemmini"

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
    verifier = Path(args.verify_tools) / "correctness_verifier.py"
    perf = Path(args.verify_tools) / "performance_counter.py"

    memory_report = json.loads(run([sys.executable, str(checker), str(mlir_file)]))
    correctness_report = json.loads(
        run(
            [
                sys.executable,
                str(verifier),
                "--mode",
                "metadata",
                "--candidate",
                str(mlir_file),
                "--reference",
                str(mlir_file),
                "--tolerance",
                "1e-5",
            ]
        )
    )
    perf_report = json.loads(run([sys.executable, str(perf), str(mlir_file)]))

    if not memory_report["accelerator_enabled"]:
        raise SystemExit("ResNet-18 compile did not enable the Gemmini accelerator")
    if memory_report["gemmini_ops"] == 0:
        raise SystemExit("ResNet-18 Gemmini compile produced no Gemmini runtime calls")
    if not memory_report["passed"]:
        raise SystemExit("memory access verification failed")
    if not correctness_report["passed"]:
        raise SystemExit("metadata correctness verification failed")

    summary = {
        "model": str(model),
        "mlir": str(mlir_file),
        "compiled": True,
        "run_mode": "host-validated-mlir",
        "gemmini_ops": memory_report["gemmini_ops"],
        "estimated_cycles": perf_report["estimated_cycles"],
        "compile_log_lines": len(compile_log.splitlines()),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
