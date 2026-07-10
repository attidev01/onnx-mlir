# RUN: %python %s --onnx-mlir onnx-mlir --workdir %t.dir

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Integration test: generate a small Sigmoid f32 model ([1,3,4,4] -> [1,3,4,4]),
# compile with --maccel=Gemmini --EmitMLIR, and verify that the output contains
# a krnl.call to om_gemmini_sigmoid_f32.

import argparse
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


def make_sigmoid_model(path):
    """Write a minimal Sigmoid ONNX model to *path*."""
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
    node = helper.make_node("Sigmoid", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "sigmoid_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-mlir", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit("onnx Python package is required")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_path = workdir / "sigmoid_test.onnx"
    make_sigmoid_model(model_path)

    out_stem = workdir / "sigmoid_gemmini"
    run([
        args.onnx_mlir,
        "--maccel=Gemmini",
        "--EmitMLIR",
        str(model_path),
        "-o", str(out_stem),
    ])

    mlir_file = out_stem.with_suffix(".onnx.mlir")
    if not mlir_file.exists():
        raise SystemExit(f"expected MLIR output not produced: {mlir_file}")

    mlir_text = mlir_file.read_text()
    if "om_gemmini_sigmoid_f32" not in mlir_text:
        raise SystemExit(f"FAIL: om_gemmini_sigmoid_f32 not found in {mlir_file}")
    if "krnl.call" not in mlir_text:
        raise SystemExit(f"FAIL: no krnl.call in {mlir_file}")

    print("PASS: Sigmoid f32 [1,3,4,4] compiled — om_gemmini_sigmoid_f32 present")


if __name__ == "__main__":
    main()
