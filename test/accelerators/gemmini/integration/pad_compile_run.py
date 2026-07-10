# RUN: %python %s --onnx-mlir onnx-mlir --workdir %t.dir

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Integration test: generate a small constant Pad f32 model, compile with
# --maccel=Gemmini --EmitMLIR, and verify the Gemmini Pad runtime call is used.

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


def make_pad_model(path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 6, 8])
    pads = numpy_helper.from_array(
        np.array([0, 0, 1, 2, 0, 0, 1, 2], dtype=np.int64), name="pads"
    )
    constant_value = numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), name="constant_value"
    )
    node = helper.make_node(
        "Pad",
        inputs=["X", "pads", "constant_value"],
        outputs=["Y"],
        mode="constant",
    )
    graph = helper.make_graph(
        [node], "pad_graph", [x], [y], [pads, constant_value]
    )
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

    model = workdir / "pad_constant_test.onnx"
    make_pad_model(model)
    out = workdir / "pad_constant_gemmini"
    run([
        args.onnx_mlir,
        "--maccel=Gemmini",
        "--EmitMLIR",
        str(model),
        "-o",
        str(out),
    ])

    mlir_file = out.with_suffix(".onnx.mlir")
    if not mlir_file.exists():
        raise SystemExit(f"expected MLIR output not produced: {mlir_file}")
    mlir_text = mlir_file.read_text()
    expected = "om_gemmini_pad_constant_f32"
    if expected not in mlir_text:
        raise SystemExit(f"FAIL: expected {expected} in {mlir_file}")
    if "krnl.call" not in mlir_text:
        raise SystemExit(f"FAIL: no krnl.call in {mlir_file}")

    print(f"PASS: Pad f32 compiled via {expected}")


if __name__ == "__main__":
    main()
