#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemmini double-buffer DMA/compute overlap benchmark on Spike.

The Gemmini tiling pass (--gemmini-tiling) emits alternating buffer-slot
schedules designed to overlap DMA loads with systolic-array compute across
K-tiles.  Since rdcycle (CSR 0xC00) is inaccessible in user mode under the
proxy kernel, exact hardware cycle counts cannot be read.  Instead this
script quantifies the overlap benefit analytically and validates the compiled
schedule via Spike wall-time measurements.

Two analytical estimates are computed for each square M×M matmul:

  seq_est_ms     — sequential compute + DMA, no overlap (upper bound)
  overlap_est_ms — max(compute, DMA) per tile, perfect overlap (lower bound)

  hw_overlap_benefit_pct = (seq_est - overlap_est) / seq_est × 100

This is the expected hardware speedup from the double-buffer schedule on a
DIM=16 Gemmini systolic array.

When Spike simulation is enabled, a linear regression

  spike_wall_s ≈ startup_s  +  per_tile_s × n_tiles

is fitted to the measurements.  R² ≥ 0.999 confirms the compiled schedule
scales correctly with tile count.  Spike models sequential instruction
execution (no DMA/compute overlap timing in the ISS), so the apparent overlap
ratio from wall time alone is ~0%; the analytical hw_overlap_benefit_pct
shows what physical Gemmini hardware would achieve.

Usage
-----
  # Full benchmark (Spike simulation, sizes = 32 64 128 256):
  python3 test/accelerators/gemmini/tools/double_buffer_spike_benchmark.py \\
    --workdir /tmp/gemmini_dbl_buf --json

  # Analytical-only (no build, no Spike):
  python3 ... --no-sim --sizes 32 64 128 256 512

  # Custom hardware profile:
  python3 ... --dim 32 --freq-ghz 1.5 --mem-bw 32
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
    import onnx
    from onnx import helper, TensorProto
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


# ── Analytical tile-cycle model ────────────────────────────────────────────

def _tile_cycles(M: int, dim: int, mem_bw: int):
    """Return (compute_cycles, mem_cycles) for an M×M square matmul.

    Uses the same formula as matmul_spike_benchmark.py / matmul_throughput_benchmark.py
    so results are directly comparable across tools.
    """
    tiles = math.ceil(M / dim)
    compute_cycles = tiles ** 3 * dim
    # A and B scratchpad loads (int8, 1 byte/element after quantization)
    a_b = tiles * tiles * dim * dim
    b_b = tiles * tiles * dim * dim
    # C accumulator store (float32, 4 bytes/element)
    c_b = tiles * tiles * dim * dim * 4
    mem_cycles = math.ceil((a_b + b_b + c_b) / mem_bw)
    return compute_cycles, mem_cycles


def seq_est_ms(M: int, dim: int = 16, freq_ghz: float = 1.0,
               mem_bw: int = 16) -> float:
    """Sequential estimate: all DMA then all compute, no overlap."""
    comp, mem = _tile_cycles(M, dim, mem_bw)
    return (comp + mem) / (freq_ghz * 1e6)


def overlap_est_ms(M: int, dim: int = 16, freq_ghz: float = 1.0,
                   mem_bw: int = 16) -> float:
    """Perfect-overlap estimate: max(compute, DMA) – DMA fully hidden."""
    comp, mem = _tile_cycles(M, dim, mem_bw)
    return max(comp, mem) / (freq_ghz * 1e6)


def hw_overlap_benefit_pct(M: int, dim: int = 16, freq_ghz: float = 1.0,
                            mem_bw: int = 16) -> float:
    """Expected HW speedup from perfect double-buffer overlap, as a percentage."""
    s = seq_est_ms(M, dim, freq_ghz, mem_bw)
    o = overlap_est_ms(M, dim, freq_ghz, mem_bw)
    return (s - o) / s * 100.0 if s > 0 else 0.0


# ── ONNX model generation ────────────────────────────────────────────────────

def make_matmul_onnx(M: int, out_path: Path) -> None:
    """Write a single-op float32 M×M MatMul ONNX model."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, M])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [M, M])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [M, M])
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], f"matmul_{M}x{M}", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


# ── Tool discovery ────────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _discover_env(repo_root: Path) -> dict:
    env = os.environ.copy()
    env.setdefault("ONNX_MLIR_BUILD_DIR", str(repo_root / "gemmini_toolchain_build"))
    env.setdefault("RISCV_INSTALL", str(Path.home() / "riscv-gemmini"))
    riscv = env["RISCV_INSTALL"]
    env["LD_LIBRARY_PATH"] = (
        f"{riscv}/lib"
        + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
    )
    return env


def _check_tools(repo_root: Path, env: dict) -> dict:
    build_dir = env["ONNX_MLIR_BUILD_DIR"]
    riscv = env["RISCV_INSTALL"]

    def require(path, label):
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"[dblbuf] missing {label}: {p}")
        return str(p)

    tools = {
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
    for cand in (f"{riscv}/bin/pk", f"{riscv}/riscv64-linux-gnu/bin/pk"):
        if Path(cand).exists():
            tools["pk"] = cand
            if "riscv64-linux-gnu" in cand:
                env["PK"] = cand
            break
    else:
        raise SystemExit(f"[dblbuf] missing proxy kernel under {riscv}")
    return tools


# ── Build and Spike run ───────────────────────────────────────────────────────

_PASS_RE = re.compile(r"\[benchmark\] PASS")


def _build_elf(M: int, model_path: Path, workdir: Path,
               tools: dict, env: dict) -> Path:
    out_dir = workdir / f"dblbuf_{M}x{M}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash", tools["run_script"],
        "--model", str(model_path),
        "--runner", tools["runner_c"],
        "--out-dir", str(out_dir),
        "--model-kind", "matmul",
        "--no-sim",
    ]
    r = subprocess.run(cmd, env=env, cwd=str(model_path.parent),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout)
        raise SystemExit(f"[dblbuf] build failed for M={M} (exit {r.returncode})")
    elf = out_dir / f"{model_path.stem}_rv64"
    if not elf.exists():
        raise SystemExit(f"[dblbuf] expected ELF not found: {elf}")
    return elf


def _run_spike(M: int, elf: Path, tools: dict, env: dict) -> float:
    """Run ELF under Spike; return wall-clock time in seconds."""
    cmd = [tools["spike"], "--extension=gemmini", "--isa=rv64imafdc", "-m2048",
           tools["pk"], str(elf), "--mat-size", str(M)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stdout)
        raise SystemExit(f"[dblbuf] Spike failed for M={M} (exit {r.returncode})")
    if not _PASS_RE.search(r.stdout):
        print(r.stdout)
        raise SystemExit(f"[dblbuf] runner did not print [benchmark] PASS for M={M}")
    return elapsed


# ── Linear regression ─────────────────────────────────────────────────────────

def _linear_fit(xs, ys):
    """Fit y = slope*x + intercept; return (slope, intercept, r²)."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, float("nan")
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2 = sum(x * x for x in xs)
    denom = n * sx2 - sx * sx
    if denom == 0:
        return 0.0, sy / n, float("nan")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-root", default=str(_find_repo_root()))
    parser.add_argument("--workdir", default="/tmp/gemmini_dbl_buf")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[32, 64, 128, 256], metavar="M",
        help="Square matrix sizes (multiples of --dim). Default: 32 64 128 256",
    )
    parser.add_argument("--dim", type=int, default=16,
                        help="Gemmini systolic-array DIM (default: 16).")
    parser.add_argument("--freq-ghz", type=float, default=1.0, dest="freq_ghz",
                        help="Assumed clock frequency in GHz (default: 1.0).")
    parser.add_argument("--mem-bw", type=int, default=16, dest="mem_bw",
                        help="Scratchpad bus bandwidth bytes/cycle (default: 16).")
    parser.add_argument("--no-sim", action="store_true",
                        help="Show analytical models only; skip build and Spike.")
    parser.add_argument("--json", action="store_true",
                        help="Append machine-readable JSON after the table.")
    args = parser.parse_args(argv)

    if not args.no_sim and not _HAS_ONNX:
        raise SystemExit("onnx is required: pip install onnx")

    repo_root = Path(args.repo_root).resolve()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    env = _discover_env(repo_root)
    tools = None
    if not args.no_sim:
        tools = _check_tools(repo_root, env)

    print(f"\nGemmini double-buffer overlap benchmark"
          f"  DIM={args.dim}  freq={args.freq_ghz} GHz"
          f"  mem_bw={args.mem_bw} B/cyc"
          f"  sim={'OFF' if args.no_sim else 'ON'}\n")
    print(
        "Columns:\n"
        "  seq_est_ms     — analytical sequential estimate (compute + DMA, no overlap)\n"
        "  overlap_est_ms — analytical perfect-overlap estimate (max(compute, DMA))\n"
        "  hw_benefit_pct — expected HW speedup from double-buffering\n"
        "  spike_wall_s   — measured Spike simulation wall time\n"
    )

    HDR = (f"{'Size':>6}  {'n_tiles':>8}  {'seq_est_ms':>12}  "
           f"{'overlap_ms':>12}  {'hw_benefit':>11}  {'spike_wall_s':>12}")
    SEP = "-" * len(HDR)
    print(SEP)
    print(HDR)
    print(SEP)

    rows = []
    tile_counts = []
    wall_times = []

    for M in args.sizes:
        n_tiles = math.ceil(M / args.dim) ** 3
        s_ms = seq_est_ms(M, args.dim, args.freq_ghz, args.mem_bw)
        o_ms = overlap_est_ms(M, args.dim, args.freq_ghz, args.mem_bw)
        benefit = hw_overlap_benefit_pct(M, args.dim, args.freq_ghz, args.mem_bw)

        wall_s = None
        if not args.no_sim:
            model_path = workdir / f"dblbuf_{M}x{M}.onnx"
            print(f"[dblbuf] building M={M} …", flush=True)
            make_matmul_onnx(M, model_path)
            elf = _build_elf(M, model_path, workdir, tools, env)
            print(f"[dblbuf] running Spike M={M} …", flush=True)
            wall_s = _run_spike(M, elf, tools, env)
            tile_counts.append(n_tiles)
            wall_times.append(wall_s)

        wall_str = f"{wall_s:.3f}" if wall_s is not None else "n/a"
        print(f"{M:>6}  {n_tiles:>8}  {s_ms:>12.4f}  "
              f"{o_ms:>12.4f}  {benefit:>10.1f}%  {wall_str:>12}")

        rows.append({
            "size": M,
            "n_tiles": n_tiles,
            "seq_est_ms": round(s_ms, 6),
            "overlap_est_ms": round(o_ms, 6),
            "hw_overlap_benefit_pct": round(benefit, 2),
            "spike_wall_s": round(wall_s, 4) if wall_s is not None else None,
        })

    print(SEP)

    regression = None
    if len(wall_times) >= 2:
        slope, intercept, r2 = _linear_fit(tile_counts, wall_times)
        startup_ms = intercept * 1e3
        per_tile_us = slope * 1e6
        sim_rate = int(1.0 / slope) if slope > 0 else 0

        avg_benefit = sum(r["hw_overlap_benefit_pct"] for r in rows) / len(rows)

        print(f"\nRegression: wall_s = {intercept:.4f} + {slope:.4e} × n_tiles"
              f"  (R² = {r2:.4f})")
        print(f"  Startup overhead : {startup_ms:.1f} ms")
        print(f"  Per-tile sim rate: {per_tile_us:.2f} µs/tile"
              f"  (~{sim_rate:,} tiles/s on Spike)")
        print(f"\nAnalytical HW overlap benefit (DIM={args.dim}, "
              f"{args.freq_ghz} GHz, {args.mem_bw} B/cyc):")
        print(f"  Average hw_benefit_pct = {avg_benefit:.1f}%  "
              f"(range: {min(r['hw_overlap_benefit_pct'] for r in rows):.1f}–"
              f"{max(r['hw_overlap_benefit_pct'] for r in rows):.1f}%)")
        print(f"  Spike models sequential execution → apparent overlap ≈ 0%"
              f"  (ISS has no DMA/compute timing overlap)")
        print(f"  R² = {r2:.4f} confirms schedule scales linearly with tile count"
              + (" (PASS)" if r2 >= 0.999 else " (low – check startup overhead)"))

        regression = {
            "slope_s_per_tile": round(slope, 9),
            "intercept_s": round(intercept, 6),
            "r_squared": round(r2, 6),
            "startup_ms": round(startup_ms, 3),
            "per_tile_us": round(per_tile_us, 3),
            "spike_sim_tiles_per_s": sim_rate,
        }

    if args.json:
        print(json.dumps({
            "config": {
                "dim": args.dim,
                "freq_ghz": args.freq_ghz,
                "mem_bw_bytes_per_cycle": args.mem_bw,
                "spike_simulated": not args.no_sim,
            },
            "results": rows,
            "regression": regression,
        }, indent=2))


if __name__ == "__main__":
    main()
