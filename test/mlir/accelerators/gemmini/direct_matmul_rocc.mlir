// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --gemmini-tiling --convert-gemmini-to-low --gemmini-static-scratchpad-allocation --convert-scf-to-cf --convert-gemmini-low-to-llvm --canonicalize %s | FileCheck %s

func.func @direct_gemmini_matmul(%lhs: tensor<16x16xi32>, %rhs: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %0 = "onnx.MatMul"(%lhs, %rhs) : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
  return %0 : tensor<16x16xi32>

// CHECK-LABEL: func.func @direct_gemmini_matmul
// Gemmini listens on XCUSTOM_ACC=3 (custom3, opcode 0x7b), not custom0.
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 0, x0, $0, $1"
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 2, x0, $0, $1"
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 4, x0, $0, $1"
// CHECK: llvm.inline_asm has_side_effects "fence"
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 3, x0, $0, $1"
}
