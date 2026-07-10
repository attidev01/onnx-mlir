# RUN: %python %s --onnx-mlir onnx-mlir --workdir %t.dir

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Integration test: generate a small 2-output Split f32 model (axis=1, channel
# split [1,6,4,4] -> [1,2,4,4] + [1,4,4,4]), compile with --maccel=Gemmini
# --EmitMLIR, and verify that the output contains a krnl.call to
# om_gemmini_split_2_f32.

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


def make_split_model(path):
    """Write a minimal 2-output Split ONNX model to *path*.

    Input:   X  [1, 6, 4, 4] f32
    Outputs: Y0 [1, 2, 4, 4] f32,  Y1 [1, 4, 4, 4] f32  (axis=1)
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 6, 4, 4])
    Y0 = helper.make_tensor_value_info("Y0", TensorProto.FLOAT, [1, 2, 4, 4])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4, 4, 4])
    split_initializer = numpy_helper.from_array(
        np.array([2, 4], dtype=np.int64), name="split_sizes"
    )
    node = helper.make_node(
        "Split",
        inputs=["X", "split_sizes"],
        outputs=["Y0", "Y1"],
        axis=1,
        num_outputs=2,
    )
    graph = helper.make_graph(
        [node], "split_graph", [X], [Y0, Y1], initializer=[split_initializer]
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
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit("onnx and numpy Python packages are required")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_path = workdir / "split_ch_test.onnx"
    make_split_model(model_path)

    out_stem = workdir / "split_ch_gemmini"
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
    if "om_gemmini_split_2_f32" not in mlir_text:
        raise SystemExit(
            f"FAIL: om_gemmini_split_2_f32 not found in {mlir_file}"
        )
    if "krnl.call" not in mlir_text:
        raise SystemExit(f"FAIL: no krnl.call in {mlir_file}")

    print("PASS: Split f32 (axis=1, 2 outputs) compiled — om_gemmini_split_2_f32 present")


if __name__ == "__main__":
    main()
