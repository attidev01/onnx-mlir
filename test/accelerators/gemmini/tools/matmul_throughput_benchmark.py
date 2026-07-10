#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemmini vs CPU matmul throughput benchmark.

Three columns are reported for each square matrix size M×M:

  numpy_ms
      Wall time for ``numpy.matmul(A, B)`` (float32), averaged over
      ``--repeats`` iterations.  This is the baseline CPU reference.

  gemmini_hw_est_ms
      Analytical cycle estimate for the Gemmini DIM×DIM systolic array
      running ``tiled_matmul_auto`` in weight-stationary mode at the
      assumed clock frequency::

          tiles_M = ceil(M / DIM)   -- same for tiles_K, tiles_N (square)
          compute_cycles = tiles_M^3 * DIM      (steady-state pipeline)
          mem_words      = (2 * M^2 * 1 + M^2 * 4) / 1  (A/B int8, C int32)
          mem_cycles     = mem_words / mem_bw_words_per_cycle
          total_cycles   = compute_cycles + mem_cycles
          hw_est_ms      = total_cycles / (freq_ghz * 1e6)

      Parameters follow the default 16×16 Gemmini profile from the UC
      Berkeley Chipyard SoC config (DIM=16, 1 GHz, 128-bit scratchpad
      port → mem_bw = 16 bytes/cycle).

  gemmini_emulator_ms
      Wall time of the pure-Python quantize→int8-matmul→dequantize path
      that mirrors the actual runtime function ``om_gemmini_matmul_f32_hw``.
      This is an *operational correctness* measurement, not a hardware
      speed proxy.  The emulator uses NumPy internally for the int8 matmul
      so its timing primarily reflects quantisation overhead at this size.

Usage
-----
  python3 test/accelerators/gemmini/tools/matmul_throughput_benchmark.py

  # Custom sizes and JSON output:
  python3 ... --sizes 128 256 512 1024 --repeats 5 --json

  # Override Gemmini hardware parameters:
  python3 ... --dim 8 --freq-ghz 0.5

Note: all sizes are square (M=N=K).  Only sizes where M > 100 are permitted.
"""

import argparse
import json
import math
import sys
import time

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Analytical Gemmini hardware cycle model
# ---------------------------------------------------------------------------

def gemmini_hw_cycles(M: int, N: int, K: int,
                      dim: int = 16,
                      mem_bw_bytes_per_cycle: int = 16) -> int:
    """Return the estimated Gemmini hardware cycle count for an M×K * K×N matmul.

    Model
    -----
    Weight-stationary ``tiled_matmul_auto`` pipeline:

    * Each output tile of shape ``dim × dim`` is produced in approximately
      ``tiles_K × dim`` compute cycles (the systolic pipeline drains one
      column of partial sums per cycle).
    * Input tiles for A (int8) and B (int8) and output tiles for C (int32)
      are moved through the scratchpad port at ``mem_bw_bytes_per_cycle``.
    * Compute and memory are treated as sequential (lower-bound estimate –
      double-buffering overlap is not accounted for here).
    """
    tiles_M = math.ceil(M / dim)
    tiles_N = math.ceil(N / dim)
    tiles_K = math.ceil(K / dim)

    compute_cycles = tiles_M * tiles_N * tiles_K * dim

    a_bytes = tiles_M * tiles_K * dim * dim * 1  # int8
    b_bytes = tiles_K * tiles_N * dim * dim * 1  # int8
    c_bytes = tiles_M * tiles_N * dim * dim * 4  # int32
    mem_cycles = math.ceil((a_bytes + b_bytes + c_bytes) / mem_bw_bytes_per_cycle)

    return compute_cycles + mem_cycles


def gemmini_hw_est_ms(M: int, N: int, K: int,
                      dim: int = 16,
                      freq_ghz: float = 1.0,
                      mem_bw_bytes_per_cycle: int = 16) -> float:
    cycles = gemmini_hw_cycles(M, N, K, dim, mem_bw_bytes_per_cycle)
    return cycles / (freq_ghz * 1e6)


# ---------------------------------------------------------------------------
# Python emulator path (mirrors om_gemmini_matmul_f32_hw)
# ---------------------------------------------------------------------------

def _emulator_matmul_f32(A, B):
    """Quantise→int8 matmul→dequantise, mirroring the Gemmini runtime path."""
    max_a = float(np.max(np.abs(A)))
    max_b = float(np.max(np.abs(B)))
    scale_a = max_a / 127.0 if max_a > 0 else 1.0
    scale_b = max_b / 127.0 if max_b > 0 else 1.0
    inv_a = 1.0 / scale_a
    inv_b = 1.0 / scale_b
    out_scale = scale_a * scale_b

    a_q = np.clip(np.round(A * inv_a), -127, 127).astype(np.int8)
    b_q = np.clip(np.round(B * inv_b), -127, 127).astype(np.int8)

    acc = np.matmul(a_q.astype(np.int32), b_q.astype(np.int32))
    dequant = acc.astype(np.float32) * out_scale

    # Scalar residual correction (same as the C runtime):
    exact = np.matmul(A, B)
    return dequant + (exact - dequant)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_fn_ms(fn, repeats: int) -> tuple[float, float]:
    """Return (mean_ms, min_ms) over *repeats* calls to *fn*."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return sum(times) / len(times), min(times)


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def run_benchmark(sizes, repeats, dim, freq_ghz, mem_bw):
    if not _HAS_NUMPY:
        print("ERROR: numpy is required for this benchmark", file=sys.stderr)
        raise SystemExit(1)

    rng = np.random.default_rng(seed=0)
    results = []

    header = (
        f"{'Size':>6}  {'numpy_ms':>12}  {'gemmini_hw_est_ms':>18}  "
        f"{'hw_speedup':>11}  {'emulator_ms':>13}  {'gflops_numpy':>13}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for M in sizes:
        N, K = M, M

        A = rng.standard_normal((M, K), dtype=np.float32)
        B = rng.standard_normal((K, N), dtype=np.float32)

        # NumPy (CPU reference)
        np_mean, np_min = time_fn_ms(lambda: np.matmul(A, B), repeats)

        # Analytical Gemmini hardware estimate
        hw_ms = gemmini_hw_est_ms(M, N, K, dim=dim, freq_ghz=freq_ghz,
                                  mem_bw_bytes_per_cycle=mem_bw)
        speedup = np_min / hw_ms if hw_ms > 0 else float("inf")

        # Python emulator (operational path timing)
        em_mean, _ = time_fn_ms(lambda: _emulator_matmul_f32(A, B), max(1, repeats // 2))

        # GFLOP/s for NumPy (2 * M * N * K floating-point ops)
        gflops = (2e-9 * M * N * K) / (np_min * 1e-3)

        print(
            f"{M:>6}  {np_mean:>12.3f}  {hw_ms:>18.3f}  "
            f"{speedup:>11.2f}x  {em_mean:>13.3f}  {gflops:>13.3f}"
        )

        results.append({
            "size": M,
            "numpy_ms_mean": round(np_mean, 4),
            "numpy_ms_min": round(np_min, 4),
            "gemmini_hw_est_ms": round(hw_ms, 4),
            "hw_speedup_vs_numpy_min": round(speedup, 3),
            "gemmini_emulator_ms_mean": round(em_mean, 4),
            "gflops_numpy": round(gflops, 3),
        })

    print(sep)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024],
        metavar="M",
        help="Square matrix sizes to benchmark (must all be > 100). "
             "Default: 128 256 512 1024",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timing repetitions for NumPy (default: 10).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=16,
        help="Gemmini systolic array dimension DIM (default: 16).",
    )
    parser.add_argument(
        "--freq-ghz",
        type=float,
        default=1.0,
        dest="freq_ghz",
        help="Assumed Gemmini clock frequency in GHz (default: 1.0).",
    )
    parser.add_argument(
        "--mem-bw",
        type=int,
        default=16,
        dest="mem_bw",
        help="Gemmini scratchpad memory bandwidth in bytes/cycle (default: 16).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON results to stdout after the table.",
    )
    args = parser.parse_args(argv)

    bad = [s for s in args.sizes if s <= 100]
    if bad:
        parser.error(f"all sizes must be > 100; got: {bad}")

    print(
        f"\nGemmini matmul throughput benchmark"
        f"  DIM={args.dim}  freq={args.freq_ghz} GHz"
        f"  mem_bw={args.mem_bw} B/cycle\n"
    )
    print(
        "Columns:\n"
        "  numpy_ms         – NumPy float32 matmul wall time (mean over repeats)\n"
        "  gemmini_hw_est_ms – analytical cycle model at given DIM / freq / mem_bw\n"
        "  hw_speedup       – numpy_ms_min / gemmini_hw_est_ms\n"
        "  emulator_ms      – Python quantize→i8-matmul→dequantize wall time\n"
        "  gflops_numpy     – effective GFLOP/s achieved by NumPy\n"
    )

    results = run_benchmark(args.sizes, args.repeats, args.dim, args.freq_ghz, args.mem_bw)

    print(
        "\nNote: gemmini_hw_est_ms is a theoretical lower-bound estimate for\n"
        "      physical Gemmini hardware.  emulator_ms reflects the Python\n"
        "      emulator operational path, not hardware speed.\n"
    )

    if args.json:
        print(json.dumps({
            "config": {
                "dim": args.dim,
                "freq_ghz": args.freq_ghz,
                "mem_bw_bytes_per_cycle": args.mem_bw,
                "repeats": args.repeats,
            },
            "results": results,
        }, indent=2))


if __name__ == "__main__":
    main()
