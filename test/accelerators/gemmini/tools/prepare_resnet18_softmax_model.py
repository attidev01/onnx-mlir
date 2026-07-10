#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Gemmini ResNet-18 test model with pretrained probabilities.

The local ``gemmini/`` workspace is intentionally ignored by git, but the
ResNet-18 Spike smoke test expects a real ImageNet model there.  This helper
downloads the ONNX model-zoo ResNet-18 logits model, verifies its SHA-256, and
adds a final Softmax node so the runner reports probabilities whose sum is 1.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

import onnx
from onnx import TensorProto, helper


RESNET18_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "resnet/model/resnet18-v1-7.onnx"
)
RESNET18_SHA256 = (
    "4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "onnx-mlir-gemmini"})
    with urllib.request.urlopen(req, timeout=180) as resp, path.open("wb") as f:
        shutil.copyfileobj(resp, f)


def ensure_logits_model(cache_path: Path, url: str, expected_sha: str) -> Path:
    if not cache_path.exists():
        download(url, cache_path)
    actual = sha256(cache_path)
    if actual != expected_sha:
        cache_path.unlink(missing_ok=True)
        raise SystemExit(
            f"SHA-256 mismatch for {cache_path}: expected {expected_sha}, got {actual}"
        )
    return cache_path


def write_softmax_model(src: Path, dest: Path) -> None:
    model = onnx.load(src)
    if not model.graph.output:
        raise SystemExit(f"model has no graph output: {src}")

    logits_name = model.graph.output[0].name
    prob_name = "resnetv15_prob"
    if not any(node.op_type == "Softmax" and prob_name in node.output
               for node in model.graph.node):
        model.graph.node.append(helper.make_node(
            "Softmax", inputs=[logits_name], outputs=[prob_name],
            axis=1, name="softmax_prob"))

    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(prob_name, TensorProto.FLOAT, [1, 1000]))
    onnx.checker.check_model(model)

    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path,
        default=Path(__file__).resolve().parents[4])
    parser.add_argument("--cache", type=Path, default=Path("/tmp/resnet18-v1-7.onnx"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--url", default=RESNET18_URL)
    parser.add_argument("--sha256", default=RESNET18_SHA256)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output = args.output or (
        repo_root / "gemmini" / "examples" / "15-resnet18-float" / "resnet18.onnx")

    logits_model = ensure_logits_model(args.cache, args.url, args.sha256)
    write_softmax_model(logits_model, output)
    print(f"[prepare_resnet18] wrote pretrained Softmax model: {output}")
    print(f"[prepare_resnet18] source SHA-256: {args.sha256}")


if __name__ == "__main__":
    main()
