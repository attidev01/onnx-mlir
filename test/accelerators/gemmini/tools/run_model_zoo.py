#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemmini backend ONNX model zoo validation suite.

For each model in the manifest this script:
  1. Resolves the model file — local path first, then cache, then download.
  2. Compiles with ``onnx-mlir --maccel=Gemmini --EmitMLIR``.
  3. Counts Gemmini ops in the emitted MLIR (krnl.call om_gemmini_* + mvin/mvout).
  4. For models marked spike_feasible=true, delegates to the matching example
     run_test.sh to obtain a numerical correctness verdict.
  5. Writes a timestamped JSON report and prints a human-readable summary.

Usage
-----
  # Minimal: use defaults, models at repo root, write report to ./zoo_report.json
  python3 test/accelerators/gemmini/tools/run_model_zoo.py

  # Full options:
  python3 test/accelerators/gemmini/tools/run_model_zoo.py \\
      --manifest  test/accelerators/gemmini/tools/model_zoo_manifest.json \\
      --repo-root /path/to/onnx-mlir \\
      --onnx-mlir /path/to/build/bin/onnx-mlir \\
      --workdir   /tmp/gemmini_zoo \\
      --report    zoo_report.json \\
      --download  \\
      --spike     \\
      --timeout   300

CMake target ``check-gemmini-zoo`` calls this script with appropriate defaults.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns for MLIR Gemmini op counting
# ---------------------------------------------------------------------------
_MVIN_RE  = re.compile(r'gemmini(?:_low)?\.mvin')
_MVOUT_RE = re.compile(r'gemmini(?:_low)?\.mvout')
_CALL_RE  = re.compile(r'"om_gemmini_[^"]+"')
_ACCEL_RE = re.compile(r'"onnx-mlir\.accels"\s*=\s*\[([^\]]*)\]')


def _count_gemmini_ops(mlir_path: Path) -> dict:
    text = mlir_path.read_text(encoding="utf-8")
    mvin_count  = len(_MVIN_RE.findall(text))
    mvout_count = len(_MVOUT_RE.findall(text))
    call_count  = len(_CALL_RE.findall(text))
    accel_tags  = []
    for m in _ACCEL_RE.finditer(text):
        accel_tags.extend(
            t.strip().strip('"') for t in m.group(1).split(",") if t.strip()
        )
    enabled = any(t.startswith("Gemmini") for t in accel_tags)
    return {
        "accelerator_enabled": enabled,
        "gemmini_ops": mvin_count + mvout_count + call_count,
        "mvin_count": mvin_count,
        "mvout_count": mvout_count,
        "runtime_call_count": call_count,
    }


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha256: str | None, log) -> bool:
    log(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gemmini-zoo/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            downloaded = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
        log(f"  saved {downloaded // 1024} KB → {dest}")
    except Exception as exc:
        log(f"  download failed: {exc}")
        dest.unlink(missing_ok=True)
        return False
    if expected_sha256:
        actual = _sha256(dest)
        if actual != expected_sha256:
            log(f"  SHA-256 mismatch: expected {expected_sha256} got {actual}")
            dest.unlink(missing_ok=True)
            return False
    return True


def _resolve_model(entry: dict, repo_root: Path, cache_dir: Path,
                   allow_download: bool, log) -> Path | None:
    # 1. local_path relative to repo root
    if entry.get("local_path"):
        lp = repo_root / entry["local_path"]
        if lp.exists():
            log(f"  using local model: {lp}")
            return lp

    # 2. cache
    cache_file = cache_dir / f"{entry['id']}.onnx"
    if cache_file.exists():
        log(f"  using cached model: {cache_file}")
        return cache_file

    # 3. download
    if not allow_download or not entry.get("url"):
        log("  model not found locally and download disabled (or no URL)")
        return None

    ok = _download(entry["url"], cache_file, entry.get("sha256"), log)
    return cache_file if ok else None


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
def _compile(model_path: Path, onnx_mlir: Path, workdir: Path,
             label: str, timeout: int, log) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    out_base = workdir / label
    cmd = [
        str(onnx_mlir),
        "--maccel=Gemmini",
        "--EmitMLIR",
        "--mtriple=riscv64-unknown-linux-gnu",
        str(model_path),
        "-o", str(out_base),
    ]
    log(f"  compile: {' '.join(cmd)}")
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s",
                "compile_time_s": timeout, "mlir_path": None,
                "stdout": "", "stderr": ""}
    elapsed = time.monotonic() - t0

    mlir_path = out_base.with_suffix(".onnx.mlir")
    ok = result.returncode == 0 and mlir_path.exists()
    # Check for unsupported op errors even on success
    unsupported = re.findall(r"unsupported op[^'\n]*'([^']+)'", result.stderr, re.I)
    unsupported += re.findall(r"unsupported op[^'\n]*'([^']+)'", result.stdout, re.I)

    return {
        "ok": ok,
        "error": "" if ok else (result.stderr[-2000:] or result.stdout[-2000:]),
        "compile_time_s": round(elapsed, 2),
        "mlir_path": str(mlir_path) if mlir_path.exists() else None,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "unsupported_ops": list(dict.fromkeys(unsupported)),
    }


# ---------------------------------------------------------------------------
# Spike run
# ---------------------------------------------------------------------------
def _run_spike(entry: dict, repo_root: Path, riscv_install: Path, log) -> dict:
    example_dir_name = entry.get("spike_example_dir")
    if not example_dir_name:
        return {"ok": False, "reason": "no spike_example_dir in manifest"}
    example_dir = (repo_root / example_dir_name).resolve()
    run_sh = example_dir / "run_test.sh"
    if not run_sh.exists():
        return {"ok": False, "reason": f"run_test.sh not found: {run_sh}"}
    log(f"  spike: running {run_sh}")
    env = {**os.environ,
           "ONNX_MLIR_BUILD_DIR": str(repo_root.resolve() / "gemmini_toolchain_build"),
           "RISCV_INSTALL": str(riscv_install)}
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["bash", str(run_sh)], capture_output=True, text=True,
            cwd=str(example_dir), env=env, timeout=600
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "spike timeout (600s)"}
    elapsed = time.monotonic() - t0
    output = result.stdout + result.stderr
    if result.returncode != 0:
        # Tolerate SKIP as a non-failure
        if "SKIP:" in output:
            reason = next((l for l in output.splitlines() if "SKIP:" in l), "SKIP")
            return {"ok": None, "reason": reason, "elapsed_s": round(elapsed, 1)}
        return {"ok": False, "reason": output[-1000:], "elapsed_s": round(elapsed, 1)}

    max_abs_diff = None
    m = re.search(r"max_abs_diff=([0-9.e+\-]+)", output)
    if m:
        max_abs_diff = float(m.group(1))
    return {
        "ok": True,
        "max_abs_diff": max_abs_diff,
        "elapsed_s": round(elapsed, 1),
        "output_tail": output[-500:],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _find_onnx_mlir(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "gemmini_toolchain_build" / "Release" / "bin" / "onnx-mlir",
        repo_root / "gemmini_toolchain_build" / "bin" / "onnx-mlir",
        Path("/opt/onnx-mlir-x86/bin/onnx-mlir"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Gemmini model zoo validator")
    parser.add_argument("--manifest", default=None,
        help="Path to model_zoo_manifest.json (default: auto-detect next to this script)")
    parser.add_argument("--repo-root", default=None,
        help="Repo root (default: 3 levels up from this script)")
    parser.add_argument("--onnx-mlir", default=None,
        help="onnx-mlir binary (default: auto-detect)")
    parser.add_argument("--workdir", default=None,
        help="Working directory for compiled artefacts (default: /tmp/gemmini_zoo)")
    parser.add_argument("--report", default="zoo_report.json",
        help="Output JSON report path (default: zoo_report.json)")
    parser.add_argument("--download", action="store_true",
        help="Download missing models from URLs in the manifest")
    parser.add_argument("--spike", action="store_true",
        help="Run Spike simulation for spike_feasible models")
    parser.add_argument("--timeout", type=int, default=300,
        help="Compile timeout per model in seconds (default: 300)")
    parser.add_argument("--models", nargs="+", default=None,
        help="Restrict to specific model IDs")
    parser.add_argument("--riscv-install", default=None,
        help="RISC-V Spike install prefix (default: ~/riscv-gemmini)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root) if args.repo_root else script_dir.parents[3]
    manifest_path = Path(args.manifest) if args.manifest else script_dir / "model_zoo_manifest.json"
    onnx_mlir = Path(args.onnx_mlir) if args.onnx_mlir else _find_onnx_mlir(repo_root)
    workdir = Path(args.workdir) if args.workdir else Path("/tmp/gemmini_zoo")
    riscv_install = Path(args.riscv_install) if args.riscv_install \
                    else Path(os.environ.get("RISCV_INSTALL", Path.home() / "riscv-gemmini"))

    if not onnx_mlir or not onnx_mlir.exists():
        print(f"[ERROR] onnx-mlir not found — set --onnx-mlir or ONNX_MLIR_BUILD_DIR", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_dir = Path(manifest.get("cache_dir", "~/.cache/gemmini-model-zoo")).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    models = manifest["models"]
    if args.models:
        models = [m for m in models if m["id"] in args.models]

    print(f"onnx-mlir : {onnx_mlir}")
    print(f"repo root : {repo_root}")
    print(f"workdir   : {workdir}")
    print(f"models    : {len(models)}")
    print()

    results = []
    unsupported_ops_log: dict[str, list[str]] = {}
    t_suite_start = time.monotonic()

    for entry in models:
        mid = entry["id"]
        label = entry["display_name"]

        lines = []
        def log(msg, _lines=lines):
            _lines.append(msg)

        print(f"[ {label} ]")
        log(f"id={mid}")

        # ── resolve model file ──────────────────────────────────────────────
        model_path = _resolve_model(entry, repo_root, cache_dir, args.download, log)
        if model_path is None:
            status = "SKIP"
            reason = "model file not found (use --download to fetch)"
            print(f"  {status}: {reason}")
            results.append({
                "id": mid, "display_name": label, "category": entry.get("category", ""),
                "status": status, "reason": reason, "log": lines,
            })
            print()
            continue

        file_size_mb = round(model_path.stat().st_size / 1_048_576, 1)
        log(f"model={model_path} ({file_size_mb} MB)")

        # ── compile ─────────────────────────────────────────────────────────
        compile_result = _compile(model_path, onnx_mlir, workdir / mid, mid,
                                  args.timeout, log)
        if not compile_result["ok"]:
            status = "FAIL"
            reason = f"compile error: {compile_result['error'][:200]}"
            print(f"  {status}: compile failed in {compile_result['compile_time_s']}s")
            results.append({
                "id": mid, "display_name": label, "category": entry.get("category", ""),
                "status": status, "compile_time_s": compile_result["compile_time_s"],
                "error": compile_result["error"][:500],
                "log": lines,
            })
            print()
            continue

        print(f"  compile OK ({compile_result['compile_time_s']}s)")
        if compile_result["unsupported_ops"]:
            print(f"  unsupported ops: {compile_result['unsupported_ops']}")
            unsupported_ops_log[mid] = compile_result["unsupported_ops"]

        # ── count Gemmini ops ───────────────────────────────────────────────
        ops_info: dict = {}
        if compile_result["mlir_path"]:
            mlir_path = Path(compile_result["mlir_path"])
            ops_info = _count_gemmini_ops(mlir_path)
            gemmini_ops = ops_info["gemmini_ops"]
            min_expected = entry.get("expected_gemmini_ops_min", 1)
            ops_ok = ops_info["accelerator_enabled"] and gemmini_ops >= min_expected
            print(f"  Gemmini ops: {gemmini_ops} "
                  f"(mvin={ops_info['mvin_count']} mvout={ops_info['mvout_count']} "
                  f"calls={ops_info['runtime_call_count']})")
            if not ops_ok:
                print(f"  WARNING: expected >= {min_expected} Gemmini ops, "
                      f"accelerator_enabled={ops_info['accelerator_enabled']}")

        # ── Spike run ────────────────────────────────────────────────────────
        spike_result: dict | None = None
        if args.spike and entry.get("spike_feasible"):
            spike_result = _run_spike(entry, repo_root, riscv_install, log)
            if spike_result["ok"] is True:
                diff_str = (f"max_abs_diff={spike_result.get('max_abs_diff')}"
                            if spike_result.get("max_abs_diff") is not None else "")
                print(f"  Spike: PASS {diff_str} ({spike_result.get('elapsed_s')}s)")
            elif spike_result["ok"] is None:
                print(f"  Spike: SKIP — {spike_result.get('reason', '')}")
            else:
                print(f"  Spike: FAIL — {spike_result.get('reason', '')[:100]}")

        # ── determine overall status ─────────────────────────────────────────
        if spike_result and spike_result["ok"] is False:
            status = "FAIL"
        elif not ops_info.get("accelerator_enabled", True):
            status = "WARN"
        else:
            status = "PASS"

        print(f"  → {status}")
        results.append({
            "id": mid,
            "display_name": label,
            "category": entry.get("category", ""),
            "status": status,
            "model_path": str(model_path),
            "file_size_mb": file_size_mb,
            "compile_time_s": compile_result["compile_time_s"],
            "unsupported_ops": compile_result.get("unsupported_ops", []),
            "gemmini_ops": ops_info.get("gemmini_ops"),
            "accelerator_enabled": ops_info.get("accelerator_enabled"),
            "mvin_count": ops_info.get("mvin_count"),
            "mvout_count": ops_info.get("mvout_count"),
            "runtime_call_count": ops_info.get("runtime_call_count"),
            "spike": spike_result,
            "log": lines,
        })
        print()

    suite_elapsed = round(time.monotonic() - t_suite_start, 1)

    # ── summary ──────────────────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_total = len(results)

    print("=" * 60)
    print(f"  TOTAL: {n_total}  PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}  SKIP: {n_skip}")
    print(f"  suite time: {suite_elapsed}s")
    if unsupported_ops_log:
        all_unsup = sorted({op for ops in unsupported_ops_log.values() for op in ops})
        print(f"  unsupported ops seen: {all_unsup}")
    print("=" * 60)

    # ── write JSON report ────────────────────────────────────────────────────
    report = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "onnx_mlir": str(onnx_mlir),
        "models_tested": n_total,
        "models_passed": n_pass,
        "models_warned": n_warn,
        "models_failed": n_fail,
        "models_skipped": n_skip,
        "suite_time_s": suite_elapsed,
        "unsupported_ops_log": unsupported_ops_log,
        "results": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to: {report_path.resolve()}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
