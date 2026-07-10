# RUN: %python %s --onnx-mlir onnx-mlir --workdir %t.dir

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Integration test: generate a small Resize f32 model (both nearest and linear
# modes), compile with --maccel=Gemmini --EmitMLIR, and verify that the output
# uses the Gemmini-backend resize calls.
#
# Unlike ConvTranspose, Resize does not go through a Decompose pass, so the
# full onnx-mlir pipeline directly invokes om_gemmini_resize_nearest_f32 or
# om_gemmini_resize_linear_f32 on supported inputs.

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


def make_resize_model(path, mode, coord_mode, nearest_mode=None):
    """Write a minimal Resize ONNX model to *path*.

    Input:  [1, 3, 4, 4] f32   (N=1, C=3, H=4, W=4)
    Output: [1, 3, 8, 8] f32   (2x spatial upsample via scales)
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales"
    )

    kwargs = dict(
        inputs=["X", "", "scales"],
        outputs=["Y"],
        mode=mode,
        coordinate_transformation_mode=coord_mode,
    )
    if nearest_mode is not None:
        kwargs["nearest_mode"] = nearest_mode

    node = helper.make_node("Resize", **kwargs)
    graph = helper.make_graph([node], "resize_graph", [X], [Y], [scales])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def check_gemmini_resize(mlir_file, expected_funcs, label):
    mlir_text = Path(mlir_file).read_text()
    found = [fn for fn in expected_funcs if fn in mlir_text]
    if not found:
        raise SystemExit(
            f"FAIL [{label}]: no expected Gemmini resize call in {mlir_file}\n"
            f"  expected one of: {expected_funcs}"
        )
    if "krnl.call" not in mlir_text:
        raise SystemExit(f"FAIL [{label}]: no krnl.call in {mlir_file}")
    return found[0]


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

    # --- Test A: nearest-neighbor resize ---
    model_near = workdir / "resize_nearest_test.onnx"
    make_resize_model(model_near, mode="nearest", coord_mode="asymmetric",
                      nearest_mode="floor")
    out_near = workdir / "resize_nearest_gemmini"
    run([
        args.onnx_mlir,
        "--maccel=Gemmini",
        "--EmitMLIR",
        str(model_near),
        "-o", str(out_near),
    ])
    mlir_near = out_near.with_suffix(".onnx.mlir")
    if not mlir_near.exists():
        raise SystemExit(f"expected MLIR output not produced: {mlir_near}")
    fn_near = check_gemmini_resize(
        mlir_near,
        ["om_gemmini_resize_nearest_f32"],
        "nearest",
    )

    # --- Test B: bilinear resize ---
    model_lin = workdir / "resize_linear_test.onnx"
    make_resize_model(model_lin, mode="linear", coord_mode="half_pixel")
    out_lin = workdir / "resize_linear_gemmini"
    run([
        args.onnx_mlir,
        "--maccel=Gemmini",
        "--EmitMLIR",
        str(model_lin),
        "-o", str(out_lin),
    ])
    mlir_lin = out_lin.with_suffix(".onnx.mlir")
    if not mlir_lin.exists():
        raise SystemExit(f"expected MLIR output not produced: {mlir_lin}")
    fn_lin = check_gemmini_resize(
        mlir_lin,
        ["om_gemmini_resize_linear_f32"],
        "linear",
    )

    print(f"PASS: Resize f32 compiled — nearest: {fn_near}, linear: {fn_lin}")


if __name__ == "__main__":
    main()
