// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --gemmini-tiling --convert-gemmini-to-low --gemmini-static-scratchpad-allocation --convert-scf-to-cf --convert-gemmini-low-to-llvm --canonicalize %s | FileCheck %s
// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --gemmini-tiling --convert-gemmini-to-low --gemmini-static-scratchpad-allocation %s | FileCheck %s --check-prefix=ACC

func.func @edge_gemmini_matmul(%lhs: tensor<20x10xi32>, %rhs: tensor<10x18xi32>) -> tensor<20x18xi32> {
  %0 = "onnx.MatMul"(%lhs, %rhs) : (tensor<20x10xi32>, tensor<10x18xi32>) -> tensor<20x18xi32>
  return %0 : tensor<20x18xi32>

// CHECK-LABEL: func.func @edge_gemmini_matmul
// CHECK: memref.alloca() : memref<16x16xi32>
// CHECK: memref.copy
// Gemmini listens on XCUSTOM_ACC=3 (custom3, opcode 0x7b), not custom0.
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 2, x0, $0, $1"
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 4, x0, $0, $1"
// CHECK: llvm.inline_asm has_side_effects ".insn r CUSTOM_3, 0x3, 3, x0, $0, $1"
// CHECK: memref.copy

// ACC-LABEL: func.func @edge_gemmini_matmul
// ACC: gemmini_low.matmul
// ACC-SAME: acc_offset_rows = 0 : i64
// ACC: gemmini_low.matmul
// ACC-SAME: acc_offset_rows = 0 : i64
// ACC: gemmini_low.matmul
// ACC-SAME: acc_offset_rows = 0 : i64
// ACC: gemmini_low.matmul
// ACC-SAME: acc_offset_rows = 0 : i64
// ACC: gemmini_low.mvout
// ACC-SAME: from 0 rows
}
