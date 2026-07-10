# REQUIRES: gemmini-simulator, gemmini-resnet18, gemmini-mobilenetv2, gemmini-bert-tiny
# RUN: %python %s \
# RUN:   --repo-root %gemmini_repo_root \
# RUN:   --workdir %t.dir \
# RUN:   --resnet18-model %resnet18_model \
# RUN:   --mobilenetv2-model %mobilenetv2_model \
# RUN:   --bert-tiny-model %bert_tiny_model \
# RUN:   --tolerance 1e-5
"""
Full numerical correctness runner for the Gemmini Spike integration suite.

Loops over ResNet-18, MobileNetV2, and BERT-tiny.  For each model:
  1. Ensures the ONNX model file is present (via fetch_test_models.py if needed).
  2. Compiles the model with ``onnx-mlir --maccel=Gemmini --mtriple=riscv64``.
  3. Runs the resulting RV64 static ELF under Spike + Gemmini extension.
  4. Captures the output tensor JSON written by float_model_runner.c.
  5. Compares it against ONNX Runtime CPU at the specified tolerance.

The test fails (exit code 1) if **any** model exceeds the tolerance threshold
or if the Spike simulation does not complete successfully.  A combined JSON
report is printed to stdout on success so that CI systems can parse results.

This test supersedes the three individual simulator stubs
(simulator_resnet18_run.py, simulator_mobilenetv2_run.py,
simulator_bert_tiny_run.py) for the purpose of auditing aggregate correctness.
Those stubs remain in the suite for per-model selective re-running.

Usage (manual):
  python3 test/accelerators/gemmini/integration/simulator_numerical_correctness.py \\
    --repo-root . \\
    --workdir /tmp/gemmini_num_correct \\
    --resnet18-model resnet18-v1-7.onnx \\
    --mobilenetv2-model mobilenetv2.onnx \\
    --bert-tiny-model bert-tiny.onnx \\
    --tolerance 1e-5

  # Auto-fetch missing models and run:
  python3 ... --fetch-missing-models
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root_default():
    return str(Path(__file__).resolve().parents[4])


def fetch_missing_models(models: dict[str, Path], repo_root: Path,
                         fetch_script: Path) -> None:
    """Run fetch_test_models.py for any model that is absent or empty."""
    needed = [
        name.replace("_", "-").replace("resnet18", "resnet18")
        for name, path in models.items()
        if name != "resnet18" and (not path.exists() or path.stat().st_size == 0)
    ]
    if not needed:
        return
    cmd = [sys.executable, str(fetch_script), "--models"] + needed
    print(f"[correctness] fetching missing models: {needed}")
    result = subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        raise SystemExit(f"fetch_test_models.py failed for: {needed}")


def run_simulator(kind: str, model_path: Path, repo_root: Path,
                  workdir: Path, tolerance: float, batch: int,
                  seq_len: int, simulator_script: Path) -> dict:
    """Invoke simulator_run.py for one model and return its JSON report."""
    cmd = [
        sys.executable,
        str(simulator_script),
        "--kind", kind,
        "--model", str(model_path),
        "--repo-root", str(repo_root),
        "--workdir", str(workdir / kind),
        "--tolerance", str(tolerance),
        "--batch", str(batch),
        "--seq-len", str(seq_len),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        raise SystemExit(
            f"simulator_run.py failed for {kind} (exit {result.returncode})"
        )
    # Last non-empty line is the JSON report from simulator_run.py.
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    try:
        report = json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        print(result.stdout, end="")
        raise SystemExit(f"simulator_run.py for {kind} produced no JSON report")
    report["_stdout_lines"] = len(result.stdout.splitlines())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-root", default=_repo_root_default())
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--resnet18-model", required=True,
                        dest="resnet18_model")
    parser.add_argument("--mobilenetv2-model", required=True,
                        dest="mobilenetv2_model")
    parser.add_argument("--bert-tiny-model", required=True,
                        dest="bert_tiny_model")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8, dest="seq_len")
    parser.add_argument(
        "--fetch-missing-models", action="store_true",
        help="Run fetch_test_models.py for MobileNetV2 and BERT-tiny if they "
             "are absent (ResNet-18 must always be provided explicitly).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    tools_dir = repo_root / "test" / "accelerators" / "gemmini" / "tools"
    simulator_script = (
        repo_root / "test" / "accelerators" / "gemmini" / "simulator_run.py"
    )

    models = {
        "resnet18":    Path(args.resnet18_model),
        "mobilenetv2": Path(args.mobilenetv2_model),
        "bert_tiny":   Path(args.bert_tiny_model),
    }

    # Optional auto-fetch for non-ResNet models.
    if args.fetch_missing_models:
        fetch_script = tools_dir / "fetch_test_models.py"
        if not fetch_script.exists():
            raise SystemExit(f"fetch script not found: {fetch_script}")
        fetch_missing_models(models, repo_root, fetch_script)

    # Validate model files before starting any simulation.
    for kind, path in models.items():
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(
                f"Missing or empty model for {kind}: {path}\n"
                f"Run with --fetch-missing-models or provide the file manually."
            )

    # Run simulations for all three models.
    results = []
    failed = []
    for kind in ("resnet18", "mobilenetv2", "bert_tiny"):
        model_path = models[kind]
        print(f"[correctness] running {kind} … ", flush=True)
        try:
            report = run_simulator(
                kind, model_path, repo_root, workdir,
                args.tolerance, args.batch, args.seq_len, simulator_script,
            )
        except SystemExit as exc:
            print(f"[correctness] {kind}: FAILED — {exc}")
            failed.append(kind)
            results.append({"kind": kind, "passed": False, "error": str(exc)})
            continue

        if not report.get("passed", False):
            print(f"[correctness] {kind}: FAILED — tolerance exceeded")
            failed.append(kind)
        else:
            max_err = report.get("max_abs_error", "n/a")
            print(f"[correctness] {kind}: PASS  max_abs_error={max_err}")
        results.append(report)

    summary = {
        "tolerance": args.tolerance,
        "models_tested": len(results),
        "models_passed": len(results) - len(failed),
        "models_failed": len(failed),
        "failed": failed,
        "results": results,
    }
    print(json.dumps(summary, sort_keys=True))

    if failed:
        raise SystemExit(
            f"Numerical correctness FAILED for: {', '.join(failed)}"
        )


if __name__ == "__main__":
    main()
