#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Spike-based Gemmini MatMul performance benchmark.

For each requested square matrix size M×M this script:

  1. Generates a minimal ONNX model containing a single ``MatMul`` op with
     float32 inputs A[M,M] and B[M,M].
  2. Drives the full cross-compilation pipeline using
     ``run_float_model_spike.sh`` with the custom ``matmul_benchmark_runner.c``
     as the runner (the standard ``--runner`` flag).  The build-only step
     (``--no-sim``) is used first so compilation failures are reported cheaply.
  3. Runs the resulting RV64 static ELF under ``spike --extension=gemmini``.
     Wall time of the subprocess is measured with ``time.perf_counter``.
     The ``rdcycle`` count emitted by the runner is parsed from stdout.
  4. Runs the same matrix multiply using NumPy ``float32`` on the host CPU
     and records the minimum wall time over ``--repeats`` iterations.
  5. Calculates:
     - ``spike_wall_s``   – wall time of the Spike subprocess (full simulation)
     - ``numpy_min_s``    – best NumPy f32 matmul time
     - ``spike_cycles``   – ``rdcycle`` delta from ``matmul_benchmark_runner.c``
     - ``hw_est_ms``      – analytical tile-model estimate at the given DIM/freq
     - ``hw_speedup``     – ``numpy_min_s / (hw_est_ms/1000)``
     - ``spike_vs_numpy`` – ``numpy_min_s / spike_wall_s`` (simulation ratio)

Output
------
A human-readable table is printed to stdout; ``--json`` appends a machine-
readable JSON object suitable for CI ingestion.

Usage
-----
  # Minimal (uses auto-detected tools from the repo root):
  python3 test/accelerators/gemmini/tools/matmul_spike_benchmark.py

  # With explicit paths:
  python3 test/accelerators/gemmini/tools/matmul_spike_benchmark.py \\
    --repo-root . \\
    --workdir /tmp/matmul_bench \\
    --sizes 128 256 512 \\
    --repeats 3 \\
    --json

  # Build only (skip Spike simulation, useful for smoke-testing the pipeline):
  python3 ... --no-sim

Requirements
------------
  onnx, numpy – Python packages (pip install onnx numpy)
  onnx-mlir   – built with Gemmini enabled (ONNX_MLIR_BUILD_DIR)
  llvm        – with riscv64 backend (LLVM_INSTALL)
  riscv-gcc   – riscv64-linux-gnu-gcc or riscv64-unknown-linux-gnu-gcc
  spike, pk   – Spike + Gemmini extension (RISCV_INSTALL)
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import onnx
    from onnx import helper, TensorProto
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


# ── Analytical hardware model (shared with matmul_throughput_benchmark.py) ──

def gemmini_hw_est_ms(M: int, N: int, K: int,
                      dim: int = 16,
                      freq_ghz: float = 1.0,
                      mem_bw: int = 16) -> float:
    tiles_M = math.ceil(M / dim)
    tiles_N = math.ceil(N / dim)
    tiles_K = math.ceil(K / dim)
    compute_cycles = tiles_M * tiles_N * tiles_K * dim
    a_b = tiles_M * tiles_K * dim * dim
    b_b = tiles_K * tiles_N * dim * dim
    c_b = tiles_M * tiles_N * dim * dim * 4
    mem_cycles = math.ceil((a_b + b_b + c_b) / mem_bw)
    total = compute_cycles + mem_cycles
    return total / (freq_ghz * 1e6)


# ── ONNX model generation ────────────────────────────────────────────────────

def make_matmul_onnx(M: int, out_path: Path) -> None:
    """Write a single-op float32 MatMul ONNX model for square matrices of size M."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, M])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [M, M])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [M, M])
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], f"matmul_{M}x{M}", [A, B], [C])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


# ── Tool discovery ────────────────────────────────────────────────────────────

def find_repo_root(script_path: Path) -> Path:
    return script_path.resolve().parents[4]


def discover_env(repo_root: Path) -> dict:
    env = os.environ.copy()
    env.setdefault("ONNX_MLIR_BUILD_DIR",
                   str(repo_root / "gemmini_toolchain_build"))
    env.setdefault("LLVM_INSTALL", str(repo_root.parent / "llvm-install"))
    env.setdefault("RISCV_INSTALL", str(Path.home() / "riscv-gemmini"))
    riscv = env["RISCV_INSTALL"]
    env["LD_LIBRARY_PATH"] = (
        f"{riscv}/lib"
        + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
    )
    return env


def check_tools(repo_root: Path, env: dict) -> dict[str, str]:
    build_dir = env["ONNX_MLIR_BUILD_DIR"]
    llvm_bin = str(Path(env["LLVM_INSTALL"]) / "bin")
    riscv = env["RISCV_INSTALL"]

    def require(path, label):
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Missing {label}: {p}")
        return str(p)

    tools = {
        "onnx_mlir": require(f"{build_dir}/Release/bin/onnx-mlir", "onnx-mlir"),
        "mlir_translate": require(f"{llvm_bin}/mlir-translate", "mlir-translate"),
        "llc": require(f"{llvm_bin}/llc", "llc"),
        "spike": require(f"{riscv}/bin/spike", "spike"),
        "run_script": require(
            repo_root / "gemmini" / "examples" / "tools" / "run_float_model_spike.sh",
            "run_float_model_spike.sh",
        ),
        "runner_c": require(
            repo_root / "gemmini" / "examples" / "tools" / "matmul_benchmark_runner.c",
            "matmul_benchmark_runner.c",
        ),
    }
    # pk: standard or alternate location
    pk = Path(riscv) / "bin" / "pk"
    alt_pk = Path(riscv) / "riscv64-linux-gnu" / "bin" / "pk"
    if pk.exists():
        tools["pk"] = str(pk)
    elif alt_pk.exists():
        tools["pk"] = str(alt_pk)
        env["PK"] = str(alt_pk)
    else:
        raise SystemExit(f"Missing RISC-V proxy kernel: {pk} or {alt_pk}")
    return tools


# ── Compilation pipeline ─────────────────────────────────────────────────────

def build_rv64_binary(M: int, model_path: Path, workdir: Path,
                      tools: dict, env: dict, run_sim: bool = False) -> Path:
    """Run run_float_model_spike.sh to build (and optionally simulate) the ELF."""
    out_dir = workdir / f"matmul_{M}x{M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash", tools["run_script"],
        "--model", str(model_path),
        "--runner", tools["runner_c"],
        "--out-dir", str(out_dir),
        "--model-kind", "matmul",   # passed through to runner; ignored by shell
    ]
    if not run_sim:
        cmd.append("--no-sim")

    result = subprocess.run(
        cmd, env=env, cwd=str(model_path.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        raise SystemExit(f"Build failed for M={M} (exit {result.returncode})")

    basename = model_path.stem  # e.g. "matmul_128x128"
    elf = out_dir / f"{basename}_rv64"
    if not elf.exists():
        raise SystemExit(f"Expected ELF not found: {elf}")
    return elf


# ── Spike execution ───────────────────────────────────────────────────────────

_CYCLES_RE = re.compile(r"\[benchmark\] cycle_count=(\d+)")
_PASS_RE   = re.compile(r"\[benchmark\] PASS")


def run_spike(M: int, elf: Path, tools: dict, env: dict) -> dict:
    """Run the ELF under Spike and return timing + rdcycle data."""
    spike_cmd = [
        tools["spike"],
        "--extension=gemmini",
        "--isa=rv64imafdc",
        "-m2048",
        tools["pk"],
        str(elf),
        "--mat-size", str(M),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(
        spike_cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    wall_s = time.perf_counter() - t0

    if result.returncode != 0:
        print(result.stdout)
        raise SystemExit(f"Spike simulation failed for M={M} (exit {result.returncode})")

    if not _PASS_RE.search(result.stdout):
        print(result.stdout)
        raise SystemExit(f"Spike runner did not print [benchmark] PASS for M={M}")

    m = _CYCLES_RE.search(result.stdout)
    spike_cycles = int(m.group(1)) if m else None
    return {"wall_s": wall_s, "cycles": spike_cycles, "stdout": result.stdout}


# ── NumPy reference timing ────────────────────────────────────────────────────

def numpy_matmul_time(M: int, repeats: int) -> float:
    rng = np.random.default_rng(seed=0)
    A = rng.standard_normal((M, M), dtype=np.float32)
    B = rng.standard_normal((M, M), dtype=np.float32)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = np.matmul(A, B)
        times.append(time.perf_counter() - t0)
    return min(times)


# ── Main benchmark driver ─────────────────────────────────────────────────────

def run_benchmark(args, repo_root: Path, env: dict, tools: dict) -> list[dict]:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    col_w = [6, 14, 14, 14, 14, 12, 12]
    header = (
        f"{'Size':>{col_w[0]}}  {'numpy_min_ms':>{col_w[1]}}  "
        f"{'spike_wall_s':>{col_w[2]}}  {'spike_cycles':>{col_w[3]}}  "
        f"{'hw_est_ms':>{col_w[4]}}  {'hw_speedup':>{col_w[5]}}  "
        f"{'spike/numpy':>{col_w[6]}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    results = []
    for M in args.sizes:
        # 1. Generate ONNX model
        model_name = f"matmul_{M}x{M}.onnx"
        model_path = workdir / model_name
        print(f"[matmul_spike] generating {model_name} …", flush=True)
        make_matmul_onnx(M, model_path)

        # 2. Compile (build + link, no simulation yet)
        print(f"[matmul_spike] compiling M={M} RV64 binary …", flush=True)
        elf = build_rv64_binary(M, model_path, workdir, tools, env,
                                run_sim=False)

        # 3. NumPy reference time
        numpy_min_s = numpy_matmul_time(M, args.repeats)

        # 4. Spike simulation
        if args.no_sim:
            spike_data = {"wall_s": None, "cycles": None}
            print(f"[matmul_spike] M={M}: Spike simulation skipped (--no-sim)")
        else:
            print(f"[matmul_spike] running Spike M={M} …", flush=True)
            spike_data = run_spike(M, elf, tools, env)

        # 5. Analytical estimate
        hw_ms = gemmini_hw_est_ms(M, M, M, args.dim, args.freq_ghz,
                                  args.mem_bw)
        hw_speedup = (numpy_min_s * 1000 / hw_ms) if hw_ms > 0 else None

        # 6. Spike vs NumPy (wall time ratio)
        spike_vs_numpy = None
        if spike_data["wall_s"] is not None and numpy_min_s > 0:
            spike_vs_numpy = numpy_min_s / spike_data["wall_s"]

        spike_w = spike_data["wall_s"]
        spike_c = spike_data["cycles"]

        # Format
        def fmt_f(v, fmt=".3f"):
            return f"{v:{fmt}}" if v is not None else "n/a"

        print(
            f"{M:>{col_w[0]}}  "
            f"{numpy_min_s*1000:>{col_w[1]}.3f}  "
            f"{fmt_f(spike_w):>{col_w[2]}}  "
            f"{str(spike_c) if spike_c else 'n/a':>{col_w[3]}}  "
            f"{hw_ms:>{col_w[4]}.3f}  "
            f"{fmt_f(hw_speedup, '.2f')+('x' if hw_speedup else ''):>{col_w[5]}}  "
            f"{fmt_f(spike_vs_numpy, '.4f'):>{col_w[6]}}"
        )

        results.append({
            "size": M,
            "numpy_min_ms": round(numpy_min_s * 1000, 4),
            "spike_wall_s": round(spike_w, 4) if spike_w else None,
            "spike_cycles": spike_c,
            "gemmini_hw_est_ms": round(hw_ms, 4),
            "hw_speedup_vs_numpy": round(hw_speedup, 3) if hw_speedup else None,
            "spike_wall_vs_numpy": round(spike_vs_numpy, 5) if spike_vs_numpy else None,
        })

    print(sep)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        default=str(find_repo_root(Path(__file__))),
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/gemmini_matmul_bench",
        help="Directory for generated models and build artefacts.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+", type=int,
        default=[128, 256, 512],
        metavar="M",
        help="Square matrix sizes (all must be > 100). Default: 128 256 512",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="NumPy timing repetitions (default: 3).",
    )
    parser.add_argument(
        "--dim", type=int, default=16,
        help="Gemmini DIM for analytic estimate (default: 16).",
    )
    parser.add_argument(
        "--freq-ghz", type=float, default=1.0, dest="freq_ghz",
        help="Assumed Gemmini clock frequency in GHz (default: 1.0).",
    )
    parser.add_argument(
        "--mem-bw", type=int, default=16, dest="mem_bw",
        help="Gemmini scratchpad bandwidth bytes/cycle (default: 16).",
    )
    parser.add_argument(
        "--no-sim", action="store_true",
        help="Build RV64 binaries but skip Spike simulation.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print JSON result block after the table.",
    )
    args = parser.parse_args(argv)

    bad = [s for s in args.sizes if s <= 100]
    if bad:
        parser.error(f"all sizes must be > 100; got: {bad}")

    if not _HAS_NUMPY:
        raise SystemExit("numpy is required (pip install numpy)")
    if not _HAS_ONNX:
        raise SystemExit("onnx is required (pip install onnx)")

    repo_root = Path(args.repo_root).resolve()
    env = discover_env(repo_root)

    print(
        f"\nGemmini Spike MatMul benchmark"
        f"  DIM={args.dim}  freq={args.freq_ghz} GHz"
        f"  sim={'OFF' if args.no_sim else 'ON'}\n"
    )
    print(
        "Columns:\n"
        "  numpy_min_ms  – best NumPy float32 matmul time (ms)\n"
        "  spike_wall_s  – Spike simulation wall time (seconds; includes Python overhead)\n"
        "  spike_cycles  – rdcycle delta from matmul_benchmark_runner.c\n"
        "  hw_est_ms     – analytic tile-cycle estimate at given DIM/freq\n"
        "  hw_speedup    – numpy_min_ms / hw_est_ms (expected HW speedup)\n"
        "  spike/numpy   – numpy_min_s / spike_wall_s (simulation ratio)\n"
    )

    print("[matmul_spike] checking tools …", flush=True)
    tools = check_tools(repo_root, env)
    print(f"[matmul_spike] onnx-mlir : {tools['onnx_mlir']}")
    print(f"[matmul_spike] spike     : {tools['spike']}")
    print(f"[matmul_spike] pk        : {tools['pk']}")

    results = run_benchmark(args, repo_root, env, tools)

    if not args.no_sim:
        print(
            "\nNote: spike_wall_s includes Spike startup and Python subprocess\n"
            "      overhead; spike_cycles is the pure hardware cycle count.\n"
            "      hw_speedup is the expected physical Gemmini speedup over host CPU.\n"
        )

    if args.json:
        print(json.dumps({
            "config": {
                "dim": args.dim,
                "freq_ghz": args.freq_ghz,
                "mem_bw_bytes_per_cycle": args.mem_bw,
                "numpy_repeats": args.repeats,
                "spike_simulated": not args.no_sim,
            },
            "results": results,
        }, indent=2))


if __name__ == "__main__":
    main()
