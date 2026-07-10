# RUN: %python %s --onnx-mlir onnx-mlir --workdir %t.dir

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Integration test: generate a tiny ConvTranspose f32 model, compile with
# --maccel=Gemmini --EmitMLIR, and verify that the output uses the Gemmini
# accelerator.
#
# The full onnx-mlir pipeline decomposes ConvTranspose into a dilated Conv
# (ONNX Decompose pass) before the Gemmini lowering runs.  The Gemmini Conv
# lowering therefore fires on the resulting Conv, producing
# om_gemmini_conv_f32_bias.  The direct om_gemmini_convtranspose_f32_bias
# path is exercised by the mlir-level lit test (convtranspose_gemmini.mlir)
# which bypasses the Decompose pass.

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd):
    completed = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        raise SystemExit(completed.returncode)
    return completed.stdout


def make_convtranspose_model(path):
    """Write a minimal ConvTranspose ONNX model to *path*.

    Input:  [1, 3, 4, 4] f32   (N=1, C=3, H=4, W=4)
    Weight: [3, 16, 3, 3] f32  (C, M, kH, kW — ONNX ConvTranspose layout)
    Bias:   [16] f32
    Output: [1, 16, 7, 7] f32  (stride=2, pad=1: (4-1)*2+3-2=7)
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(42)
    w_data = rng.standard_normal((3, 16, 3, 3)).astype(np.float32)
    b_data = rng.standard_normal((16,)).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 7, 7])
    W = numpy_helper.from_array(w_data, name="W")
    B = numpy_helper.from_array(b_data, name="B")

    node = helper.make_node(
        "ConvTranspose",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        group=1,
    )
    graph = helper.make_graph([node], "ct_graph", [X], [Y], [W, B])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
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

    model_path = workdir / "convtranspose_test.onnx"
    make_convtranspose_model(model_path)

    output_base = workdir / "convtranspose_gemmini"
    run(
        [
            args.onnx_mlir,
            "--maccel=Gemmini",
            "--EmitMLIR",
            str(model_path),
            "-o",
            str(output_base),
        ]
    )

    mlir_file = output_base.with_suffix(".onnx.mlir")
    if not mlir_file.exists():
        raise SystemExit(f"expected MLIR output not produced: {mlir_file}")

    mlir_text = mlir_file.read_text()

    # The full pipeline decomposes ConvTranspose → Conv before Gemmini lowering,
    # so the produced call is om_gemmini_conv_f32_bias (or, if that path is ever
    # bypassed, om_gemmini_convtranspose_f32_bias).
    gemmini_calls = [
        "om_gemmini_conv_f32_bias",
        "om_gemmini_conv_f32",
        "om_gemmini_convtranspose_f32_bias",
        "om_gemmini_convtranspose_f32",
    ]
    found = [name for name in gemmini_calls if name in mlir_text]
    if not found:
        raise SystemExit(
            "FAIL: no Gemmini ConvTranspose/Conv call found in emitted MLIR\n"
            f"  expected one of: {gemmini_calls}\n"
            f"  (checked {mlir_file})"
        )

    if "krnl.call" not in mlir_text:
        raise SystemExit("FAIL: no krnl.call found in emitted MLIR")

    print(f"PASS: ConvTranspose f32 compiled — Gemmini call: {found[0]}")


if __name__ == "__main__":
    main()
