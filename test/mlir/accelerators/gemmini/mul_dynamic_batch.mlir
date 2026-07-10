// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — dynamic batch [?x3x4x4] × [?x3x4x4] elementwise mul → [?x3x4x4].

func.func @mul_dynamic_batch(%a: tensor<?x3x4x4xf32>, %b: tensor<?x3x4x4xf32>) -> tensor<?x3x4x4xf32> {
  %c = "onnx.Mul"(%a, %b) : (tensor<?x3x4x4xf32>, tensor<?x3x4x4xf32>) -> tensor<?x3x4x4xf32>
  return %c : tensor<?x3x4x4xf32>
}

// CHECK-LABEL: func.func @mul_dynamic_batch
// Dynamic alloc: batch dim comes from memref.dim of the first input.
// CHECK: [[DIM:%.+]] = memref.dim %arg0, {{%.+}} : memref<?x3x4x4xf32>
// CHECK: [[OUT:%.+]] = memref.alloc([[DIM]]) {{.*}}: memref<?x3x4x4xf32>
// CHECK: "krnl.call"([[OUT]], %arg0, %arg1) <{funcName = "om_gemmini_mul_f32"
// CHECK-SAME: numOfOutput = 1
// CHECK: return [[OUT]] : memref<?x3x4x4xf32>
