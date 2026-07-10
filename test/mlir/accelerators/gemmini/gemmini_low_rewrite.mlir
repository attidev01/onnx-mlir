// RUN: onnx-mlir-opt --maccel=Gemmini --gemmini-low-rewrite %s | FileCheck %s

func.func @rewrite_redundant_low_ops(%lhs: memref<16x16xi32>, %rhs: memref<16x16xi32>, %out: memref<16x16xi32>) {
  gemmini_low.config "ws"
  gemmini_low.config "ws"
  gemmini_low.config "os"
  gemmini_low.mvin %lhs to 0 rows tile (16 x 16) : memref<16x16xi32>
  gemmini_low.mvin %lhs to 0 rows tile (16 x 16) : memref<16x16xi32>
  gemmini_low.mvin %rhs to 16 rows tile (16 x 16) : memref<16x16xi32>
  gemmini_low.matmul %lhs, %rhs into %out mode "identity" dataflow "ws" : memref<16x16xi32>, memref<16x16xi32>, memref<16x16xi32>
  gemmini_low.fence
  gemmini_low.fence
  gemmini_low.mvout %out from 0 rows tile (16 x 16) : memref<16x16xi32>
  return
}

// CHECK-LABEL: func.func @rewrite_redundant_low_ops
// CHECK: gemmini_low.config "ws"
// CHECK-NOT: gemmini_low.config "ws"
// CHECK: gemmini_low.config "os"
// CHECK: gemmini_low.mvin {{.*}} to 0 rows tile(16 x 16)
// CHECK-NOT: gemmini_low.mvin {{.*}} to 0 rows tile(16 x 16)
// CHECK: gemmini_low.mvin {{.*}} to 16 rows tile(16 x 16)
// CHECK: gemmini_low.matmul
// CHECK: gemmini_low.fence
// CHECK-NOT: gemmini_low.fence
// CHECK: gemmini_low.mvout
