# RUN: %python %s --onnx-mlir onnx-mlir --workdir %t.dir

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Integration test: generate a small Slice f32 model, compile with
# --maccel=Gemmini --EmitMLIR, and verify that the output contains a krnl.call
# to om_gemmini_slice_f32.
#
# Model: Input [1, 4, 8, 4] f32 → Slice H axis=[2] start=1 end=5 step=2
#        → Output [1, 4, 2, 4] f32.

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


def make_slice_model(path):
    """Write a minimal Slice ONNX model to *path*.

    Input:  X  [1, 4, 8, 4] f32
    Output: Y  [1, 4, 2, 4] f32   (H axis, start=1, end=5, step=2)
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 2, 4])

    starts_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts")
    ends_init   = numpy_helper.from_array(np.array([5], dtype=np.int64), name="ends")
    axes_init   = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")
    steps_init  = numpy_helper.from_array(np.array([2], dtype=np.int64), name="steps")

    node = helper.make_node(
        "Slice",
        inputs=["X", "starts", "ends", "axes", "steps"],
        outputs=["Y"],
    )
    graph = helper.make_graph(
        [node], "slice_graph", [X], [Y],
        initializer=[starts_init, ends_init, axes_init, steps_init],
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

    model_path = workdir / "slice_test.onnx"
    make_slice_model(model_path)

    out_stem = workdir / "slice_gemmini"
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
    if "om_gemmini_slice_f32" not in mlir_text:
        raise SystemExit(
            f"FAIL: om_gemmini_slice_f32 not found in {mlir_file}"
        )
    if "krnl.call" not in mlir_text:
        raise SystemExit(f"FAIL: no krnl.call in {mlir_file}")

    print("PASS: Slice f32 compiled — om_gemmini_slice_f32 present")


if __name__ == "__main__":
    main()
