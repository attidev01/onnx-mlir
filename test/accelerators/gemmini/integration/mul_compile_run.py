# RUN: %python %s --onnx-mlir onnx-mlir --workdir %t.dir

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Integration test: generate a small Mul f32 model ([1,3,4,4] x [1,3,4,4]),
# compile with --maccel=Gemmini --EmitMLIR, and verify that the output contains
# a krnl.call to om_gemmini_mul_f32.

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


def make_mul_model(path):
    """Write a minimal Mul ONNX model to *path*.

    Input:  A [1,3,4,4] f32, B [1,3,4,4] f32
    Output: C [1,3,4,4] f32 = A * B (elementwise)
    """
    import onnx
    from onnx import TensorProto, helper

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 3, 4, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 3, 4, 4])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 3, 4, 4])
    node = helper.make_node("Mul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "mul_graph", [A, B], [C])
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

    model_path = workdir / "mul_test.onnx"
    make_mul_model(model_path)

    out_stem = workdir / "mul_gemmini"
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
    if "om_gemmini_mul_f32" not in mlir_text:
        raise SystemExit(f"FAIL: om_gemmini_mul_f32 not found in {mlir_file}")
    if "krnl.call" not in mlir_text:
        raise SystemExit(f"FAIL: no krnl.call in {mlir_file}")

    print("PASS: Mul f32 [1,3,4,4]x[1,3,4,4] compiled — om_gemmini_mul_f32 present")


if __name__ == "__main__":
    main()
