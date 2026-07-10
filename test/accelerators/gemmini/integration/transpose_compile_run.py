# RUN: %python %s --onnx-mlir onnx-mlir --workdir %t.dir

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Integration test: generate a small Transpose f32 model (NCHW -> NHWC),
# compile with --maccel=Gemmini --EmitMLIR, and verify that the output contains
# a krnl.call to om_gemmini_transpose_f32_hw.

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


def make_transpose_model(path):
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4, 3])
    node = helper.make_node(
        "Transpose", inputs=["X"], outputs=["Y"], perm=[0, 2, 3, 1]
    )
    graph = helper.make_graph([node], "transpose_graph", [x], [y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)]
    )
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
        raise SystemExit("onnx Python package is required for this test")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_path = workdir / "transpose_nhwc_test.onnx"
    make_transpose_model(model_path)

    out_stem = workdir / "transpose_nhwc_gemmini"
    run([
        args.onnx_mlir,
        "--maccel=Gemmini",
        "--EmitMLIR",
        str(model_path),
        "-o",
        str(out_stem),
    ])

    mlir_file = out_stem.with_suffix(".onnx.mlir")
    if not mlir_file.exists():
        raise SystemExit(f"expected MLIR output not produced: {mlir_file}")

    mlir_text = mlir_file.read_text()
    if "om_gemmini_transpose_f32_hw" not in mlir_text:
        raise SystemExit(
            f"FAIL: om_gemmini_transpose_f32_hw not found in {mlir_file}"
        )
    if "krnl.call" not in mlir_text:
        raise SystemExit(f"FAIL: no krnl.call in {mlir_file}")

    print("PASS: Transpose f32 compiled — om_gemmini_transpose_f32_hw present")


if __name__ == "__main__":
    main()
