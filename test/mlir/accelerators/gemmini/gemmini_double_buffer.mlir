// RUN: onnx-mlir-opt --maccel=Gemmini --gemmini-tiling %s | FileCheck %s
// RUN: onnx-mlir-opt --maccel=Gemmini --gemmini-tiling --convert-gemmini-to-low --gemmini-static-scratchpad-allocation %s | FileCheck %s --check-prefix=LOW

func.func @double_buffered_matmul(%lhs: memref<16x32xi32>, %rhs: memref<32x16xi32>, %out: memref<16x16xi32>) {
  gemmini.matmul %lhs, %rhs into %out mode "identity" dataflow "ws" {gemmini.high_level} : memref<16x32xi32>, memref<32x16xi32>, memref<16x16xi32>
  return
}

// CHECK-LABEL: func.func @double_buffered_matmul
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: scf.if
// CHECK: gemmini.mvin {{.*}} to 0 rows tile(16 x 16)
// CHECK: gemmini.mvin {{.*}} to 16 rows tile(16 x 16)
// CHECK: gemmini.matmul {{.*}} {gemmini.buffer_slot = 0
// CHECK: gemmini.mvin {{.*}} to 32 rows tile(16 x 16)
// CHECK: gemmini.mvin {{.*}} to 48 rows tile(16 x 16)
// CHECK: gemmini.matmul {{.*}} {gemmini.buffer_slot = 1

// LOW-LABEL: func.func @double_buffered_matmul
// LOW: gemmini_low.mvin {{.*}} to 0 rows tile(16 x 16)
// LOW: gemmini_low.mvin {{.*}} to 16 rows tile(16 x 16)
// LOW: gemmini_low.matmul {{.*}} {acc_offset_rows
// LOW-SAME: gemmini.buffer_slot = 0
// LOW-SAME: lhs_spad_offset_rows = 0
// LOW-SAME: rhs_spad_offset_rows = 16
// LOW: gemmini_low.mvin {{.*}} to 32 rows tile(16 x 16)
// LOW: gemmini_low.mvin {{.*}} to 48 rows tile(16 x 16)
// LOW: gemmini_low.matmul {{.*}} {acc_offset_rows
// LOW-SAME: gemmini.buffer_slot = 1
// LOW-SAME: lhs_spad_offset_rows = 32
// LOW-SAME: rhs_spad_offset_rows = 48
