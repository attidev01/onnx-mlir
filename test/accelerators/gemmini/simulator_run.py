#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd, env, cwd):
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.returncode, completed.stdout


def require_file(path, label):
    candidate = Path(path)
    if not candidate.exists():
        raise SystemExit(f"missing {label}: {candidate}")
    return candidate


def simulator_env(args, repo_root):
    env = os.environ.copy()
    env.setdefault("ONNX_MLIR_BUILD_DIR", str(repo_root / "gemmini_toolchain_build"))
    env.setdefault("LLVM_INSTALL", str(repo_root.parent / "llvm-install"))
    env.setdefault("RISCV_INSTALL", str(Path.home() / "riscv-gemmini"))
    env.setdefault("OUT_BASE", str(Path(args.workdir) / "quantized"))
    env.setdefault("OUT_DIR", str(Path(args.workdir) / "float"))
    env["LD_LIBRARY_PATH"] = (
        str(Path(env["RISCV_INSTALL"]) / "lib")
        + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    )
    return env


def check_simulator_tools(env):
    riscv = Path(env["RISCV_INSTALL"])
    require_file(riscv / "bin" / "spike", "Spike executable")
    pk = riscv / "bin" / "pk"
    alt_pk = riscv / "riscv64-linux-gnu" / "bin" / "pk"
    if not pk.exists() and alt_pk.exists():
        env["PK"] = str(alt_pk)
    else:
        require_file(pk, "RISC-V proxy kernel")
    require_file(riscv / "lib" / "libgemmini.so", "Gemmini Spike extension")


def run_matmul_i8(args, repo_root, env):
    script = require_file(
        repo_root / "gemmini" / "examples" / "tools" / "run_quantized_spike.sh",
        "quantized simulator script",
    )
    env["OUT_BASE"] = str(Path(args.workdir) / "matmul_i8")
    code, output = run(
        ["bash", str(script), "--example", "03"], env=env, cwd=repo_root
    )
    if code != 0:
        sys.stdout.write(output)
        raise SystemExit(code)
    if (
        "om_gemmini_matmulinteger_i8i8acc32" not in output
        and "Verified Gemmini MatMulInteger output" not in output
    ):
        sys.stdout.write(output)
        raise SystemExit("matmul_i8 simulation did not use Gemmini runtime")
    return {
        "kind": "matmul_i8",
        "passed": True,
        "tolerance": args.tolerance,
        "gemmini_runtime_calls": ["om_gemmini_matmulinteger_i8i8acc32"],
    }


def run_resnet18(args, repo_root, env):
    return run_float_model(args, repo_root, env, "resnet18")


def deterministic_f32(shape):
    values = np.arange(int(np.prod(shape)), dtype=np.int64)
    values = ((values * 37 + 13) % 1000).astype(np.float32)
    return ((values - 500.0) / 10000.0).reshape(shape).astype(np.float32)


def deterministic_bert_inputs(batch, seq_len):
    values = np.arange(batch * seq_len, dtype=np.int64)
    input_ids = ((values * 17 + 11) % 997).reshape(batch, seq_len)
    attention_mask = np.ones((batch, seq_len), dtype=np.int64)
    token_type_ids = np.zeros((batch, seq_len), dtype=np.int64)
    return [input_ids, attention_mask, token_type_ids]


def require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for Gemmini numerical validation; "
            "install the Python onnxruntime package and rerun this test"
        ) from exc
    return ort


def load_spike_output(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("dtype") != "f32":
        raise SystemExit(f"unexpected Spike output dtype: {payload.get('dtype')}")
    shape = tuple(int(d) for d in payload["shape"])
    data = np.asarray(payload["data"], dtype=np.float32)
    return data.reshape(shape)


def first_mismatch(actual, expected, tolerance):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    limit = tolerance + tolerance * np.abs(expected.astype(np.float64))
    bad = np.argwhere(diff > limit)
    if bad.size == 0:
        return None
    index = tuple(int(i) for i in bad[0])
    return {
        "index": list(index),
        "actual": float(actual[index]),
        "expected": float(expected[index]),
        "abs_error": float(diff[index]),
        "limit": float(limit[index]),
    }


def compare_with_onnxruntime(model, kind, actual, args):
    ort = require_onnxruntime()
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_defs = session.get_inputs()

    if kind == "bert_tiny":
        values = deterministic_bert_inputs(args.batch, args.seq_len)
        if len(input_defs) != 3:
            raise SystemExit(
                f"BERT-tiny reference expected 3 graph inputs, got {len(input_defs)}"
            )
        feeds = {input_defs[i].name: values[i] for i in range(3)}
    else:
        values = deterministic_f32((args.batch, 3, 224, 224))
        feeds = {input_defs[0].name: values}

    expected = session.run(None, feeds)[0].astype(np.float32)
    if actual.shape != expected.shape:
        raise SystemExit(
            f"Spike output shape {actual.shape} does not match ORT {expected.shape}"
        )

    abs_diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denom = np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float64).tiny)
    max_abs = float(np.max(abs_diff))
    max_rel = float(np.max(abs_diff / denom))
    mismatch = first_mismatch(actual, expected, args.tolerance)
    return {
        "passed": mismatch is None,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "mismatch": mismatch,
        "reference": "onnxruntime-cpu",
    }


def float_model_config(kind):
    if kind == "resnet18":
        return {
            "label": "ResNet-18",
            "model_kind": "cnn",
            "runtime_markers": ["om_gemmini_conv_f32", "om_gemmini_gemm_f32"],
        }
    if kind == "mobilenetv2":
        return {
            "label": "MobileNetV2",
            "model_kind": "cnn",
            "runtime_markers": ["om_gemmini_conv_f32", "om_gemmini_gemm_f32"],
        }
    if kind == "bert_tiny":
        return {
            "label": "BERT-tiny",
            "model_kind": "bert",
            "runtime_markers": ["om_gemmini_matmul_f32_nd_hw"],
        }
    raise AssertionError(f"unsupported float model kind: {kind}")


def run_float_model(args, repo_root, env, kind):
    model = require_file(args.model, f"{float_model_config(kind)['label']} ONNX model")
    config = float_model_config(kind)
    script = require_file(
        repo_root / "gemmini" / "examples" / "tools" / "run_float_model_spike.sh",
        "float simulator script",
    )
    env["OUT_DIR"] = str(Path(args.workdir) / kind)
    dump_output = Path(env["OUT_DIR"]) / "spike_output.json"
    cmd = [
        "bash",
        str(script),
        "--model",
        str(model),
        "--out-dir",
        env["OUT_DIR"],
        "--model-kind",
        config["model_kind"],
        "--in-n",
        str(args.batch),
        "--in-c",
        "3",
        "--in-h",
        "224",
        "--in-w",
        "224",
        "--seq-len",
        str(args.seq_len),
        "--dump-output",
        str(dump_output),
    ]
    code, output = run(cmd, env=env, cwd=repo_root)
    if code != 0:
        sys.stdout.write(output)
        raise SystemExit(code)
    if not any(marker in output for marker in config["runtime_markers"]):
        sys.stdout.write(output)
        raise SystemExit(f"{config['label']} simulation did not use Gemmini runtime")
    finite_match = re.search(r"nans=(\d+)\s+infs=(\d+)", output)
    if not finite_match or finite_match.group(1) != "0" or finite_match.group(2) != "0":
        sys.stdout.write(output)
        raise SystemExit(f"{config['label']} simulation did not prove finite output")
    if "[runner] PASS" not in output:
        sys.stdout.write(output)
        raise SystemExit(f"{config['label']} runner did not report PASS")
    actual = load_spike_output(dump_output)
    comparison = compare_with_onnxruntime(model, kind, actual, args)
    if not comparison["passed"]:
        sys.stdout.write(output)
        print(json.dumps(comparison, sort_keys=True))
        raise SystemExit(
            f"{config['label']} Spike output does not match ONNX Runtime CPU "
            f"within tolerance {args.tolerance}"
        )
    return {
        "kind": kind,
        "passed": True,
        "tolerance": args.tolerance,
        "model": str(model),
        "output_checked": "onnxruntime-cpu",
        "max_abs_error": comparison["max_abs_error"],
        "max_rel_error": comparison["max_rel_error"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemmini simulator validation for selected models."
    )
    parser.add_argument(
        "--kind",
        choices=["matmul_i8", "resnet18", "mobilenetv2", "bert_tiny"],
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = simulator_env(args, repo_root)
    check_simulator_tools(env)

    if args.kind == "matmul_i8":
        report = run_matmul_i8(args, repo_root, env)
    else:
        if not args.model:
            raise SystemExit(f"--model is required for --kind {args.kind}")
        report = run_float_model(args, repo_root, env, args.kind)

    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
