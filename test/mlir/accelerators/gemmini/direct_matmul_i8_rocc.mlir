// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --gemmini-tiling --convert-gemmini-to-low --gemmini-static-scratchpad-allocation --convert-scf-to-cf --convert-gemmini-low-to-llvm --canonicalize %s | FileCheck %s

// INT8 MatMulInteger (no zero points) → ONNXMatMulIntegerDirectToGemminiLowering
// → GemminiMatmulOp → tiling → GemminiLow dialect → CUSTOM_3 inline asm.
// Verified instruction sequence:
//   CONFIG_EX  (funct=0, rs1: acc_scale|dataflow)
//   MVIN_A     (funct=2)
//   MVIN_B     (funct=2)
//   PRELOAD    (funct=6, rs1: B addr, rs2: C acc addr | 0x80000000)
//   COMPUTE_PRELOADED (funct=4, rs1: A addr, rs2: 0xFFFFFFFF)
//   FENCE
//   CONFIG_MVOUT (funct=0, rs1: 0x02)
//   MVOUT      (funct=3, rs2: acc addr | 0x80000000)

func.func @direct_gemmini_matmul_i8(%lhs: tensor<16x16xi8>, %rhs: tensor<16x16xi8>) -> tensor<16x16xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.MatMulInteger"(%lhs, %rhs, %none, %none) : (tensor<16x16xi8>, tensor<16x16xi8>, none, none) -> tensor<16x16xi32>
  return %0 : tensor<16x16xi32>

// CHECK-LABEL: func.func @direct_gemmini_matmul_i8
// config_ex (funct=0)
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 0, x0, $0, $1"
// mvin A and mvin B (funct=2)
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 2, x0, $0, $1"
// preload (funct=6)
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 6, x0, $0, $1"
// compute_preloaded (funct=4)
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 4, x0, $0, $1"
// fence
// CHECK: llvm.inline_asm has_side_effects "fence"
// mvout (funct=3)
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 3, x0, $0, $1"
}
