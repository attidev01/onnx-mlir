#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Deterministic test-model acquisition for the Gemmini integration suite.

For each of the two non-ResNet models needed by the suite this script tries,
in order:

  1. Skip if the file already exists and its SHA-256 matches the recorded
     checksum (``--force`` overrides this).
  2. Download from the canonical ONNX Model Zoo / HuggingFace ONNX Hub URL.
  3. Generate the model locally using PyTorch + torchvision / transformers
     (random weights – sufficient for compile correctness tests).

After a successful acquisition the SHA-256 of the written file is printed so
that the hardcoded checksums below can be updated when model sources change.

Usage
-----
  # Place models in the repo root (default):
  python3 test/accelerators/gemmini/tools/fetch_test_models.py

  # Place models in a specific directory:
  python3 test/accelerators/gemmini/tools/fetch_test_models.py --outdir /tmp/models

  # Force re-download / re-generation even if files already exist:
  python3 test/accelerators/gemmini/tools/fetch_test_models.py --force

  # Fetch only one of the two models:
  python3 test/accelerators/gemmini/tools/fetch_test_models.py --models mobilenetv2
  python3 test/accelerators/gemmini/tools/fetch_test_models.py --models bert-tiny
"""

import argparse
import hashlib
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry describes where to find / how to generate a test model.
#
# ``sha256`` is the expected digest of the *downloaded* file.  Set to None to
# skip digest validation (useful when the generation path is used instead).
# Run with ``--print-sha256`` to compute and print the actual digest so you
# can update this table after a source change.

_MODELS = {
    "mobilenetv2": {
        "filename": "mobilenetv2.onnx",
        # ONNX Model Zoo – mobilenetv2-12, opset 12, 224×224 input.
        "url": (
            "https://github.com/onnx/models/raw/main/validated/"
            "vision/classification/mobilenet/model/mobilenetv2-12.onnx"
        ),
        # sha256 verified 2026-06-02 against the file downloaded from the URL above.
        "sha256": "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
        "opset": 12,
    },
    "bert-tiny": {
        "filename": "bert-tiny.onnx",
        # ONNX export of prajjwal1/bert-tiny (2L-128H-2A-512I).
        # Primary: Xenova HuggingFace mirror (has reliable ONNX subdirectory).
        # Fallback: generation via transformers.
        "url": (
            "https://huggingface.co/Xenova/bert-tiny/resolve/main/onnx/model.onnx"
        ),
        # sha256 verified 2026-06-02 against the repo-root bert-tiny.onnx used
        # in the Gemmini integration suite.
        "sha256": "92275309942fbd41c289a01488b94c51d0d60782d00641254fd076b2447958b4",
        "opset": 13,
    },
}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> bool:
    """Try to download *url* to *dest*.  Return True on success."""
    log.info("Downloading %s -> %s", url, dest)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as exc:
        log.warning("Download failed (%s): %s", type(exc).__name__, exc)
        return False


def _validate_sha256(path: Path, expected: str | None) -> bool:
    if expected is None:
        return True
    actual = sha256_of(path)
    if actual == expected:
        return True
    log.error("SHA-256 mismatch for %s:\n  expected %s\n  got      %s", path, expected, actual)
    return False


# ---------------------------------------------------------------------------
# Generation fallbacks
# ---------------------------------------------------------------------------

def _generate_mobilenetv2(dest: Path, opset: int) -> bool:
    """Generate a MobileNetV2 ONNX model with random weights via torchvision."""
    try:
        import torch
        import torchvision.models as tv
    except ImportError:
        log.warning("torch/torchvision not available – cannot generate MobileNetV2")
        return False
    log.info("Generating MobileNetV2 ONNX (random weights) -> %s", dest)
    try:
        model = tv.mobilenet_v2(weights=None)
        model.eval()
        dummy = torch.zeros(1, 3, 224, 224)
        torch.onnx.export(
            model, dummy, str(dest),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        return True
    except Exception as exc:
        log.error("MobileNetV2 generation failed: %s", exc)
        return False


def _generate_bert_tiny(dest: Path, opset: int) -> bool:
    """Generate a BERT-tiny ONNX model (2L-128H-2A) with random weights."""
    try:
        import torch
        from transformers import BertConfig, BertModel
    except ImportError:
        log.warning("torch/transformers not available – cannot generate BERT-tiny")
        return False
    log.info("Generating BERT-tiny ONNX (random weights) -> %s", dest)
    try:
        cfg = BertConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=128,
            vocab_size=30522,
        )
        model = BertModel(cfg)
        model.eval()
        seq_len = 32
        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)
        torch.onnx.export(
            model,
            (input_ids, attention_mask, token_type_ids),
            str(dest),
            opset_version=opset,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
                "pooler_output": {0: "batch"},
            },
        )
        return True
    except Exception as exc:
        log.error("BERT-tiny generation failed: %s", exc)
        return False


_GENERATORS = {
    "mobilenetv2": _generate_mobilenetv2,
    "bert-tiny": _generate_bert_tiny,
}


# ---------------------------------------------------------------------------
# Main acquisition logic
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root (four levels above this script's directory)."""
    return Path(__file__).resolve().parents[4]


def acquire(name: str, outdir: Path, force: bool = False) -> bool:
    spec = _MODELS[name]
    dest = outdir / spec["filename"]

    if not force and dest.exists() and dest.stat().st_size > 0:
        if _validate_sha256(dest, spec["sha256"]):
            log.info("%-15s already present, checksum OK – skipping (%s)", name, dest)
            print(f"[{name}] OK (cached) sha256={sha256_of(dest)}")
            return True
        log.warning("%-15s checksum mismatch – re-acquiring", name)

    # --- Strategy 0: copy from repo root if the file already lives there ---
    repo_copy = _repo_root() / spec["filename"]
    if repo_copy != dest and repo_copy.exists() and repo_copy.stat().st_size > 0:
        if _validate_sha256(repo_copy, spec["sha256"]):
            shutil.copy2(str(repo_copy), str(dest))
            log.info("%-15s copied from repo root – sha256=%s", name, sha256_of(dest))
            print(f"[{name}] OK (repo-root copy) sha256={sha256_of(dest)}")
            return True
        log.warning("%-15s repo-root copy checksum mismatch – trying download", name)

    # --- Strategy 1: download ---
    if _download(spec["url"], dest):
        if _validate_sha256(dest, spec["sha256"]):
            digest = sha256_of(dest)
            log.info("%-15s downloaded OK – sha256=%s", name, digest)
            print(f"[{name}] OK (downloaded) sha256={digest}")
            return True
        log.warning("%-15s download checksum mismatch – trying generation", name)
        dest.unlink(missing_ok=True)

    # --- Strategy 2: local generation ---
    generator = _GENERATORS.get(name)
    if generator and generator(dest, spec["opset"]):
        digest = sha256_of(dest)
        log.info("%-15s generated OK – sha256=%s", name, digest)
        print(f"[{name}] OK (generated) sha256={digest}")
        return True

    log.error("%-15s FAILED – could not download or generate", name)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write models into.  Defaults to the repository root "
             "(two levels above this script's directory).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(_MODELS),
        default=list(_MODELS),
        help="Which models to acquire (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download / re-generate even if models already exist.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    if args.outdir is None:
        # Two levels up from .../test/accelerators/gemmini/tools/
        outdir = Path(__file__).resolve().parent.parent.parent.parent.parent
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    failed = [name for name in args.models if not acquire(name, outdir, args.force)]

    if failed:
        print(f"ERROR: failed to acquire: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
