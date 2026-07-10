// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — static [1,3,4,4] × [1,3,4,4] elementwise mul → [1,3,4,4].

func.func @mul_static(%a: tensor<1x3x4x4xf32>, %b: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
  %c = "onnx.Mul"(%a, %b) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
  return %c : tensor<1x3x4x4xf32>
}

// CHECK-LABEL: func.func @mul_static
// CHECK: [[OUT:%.+]] = memref.alloc() {{.*}}: memref<1x3x4x4xf32>
// CHECK: "krnl.call"([[OUT]], %arg0, %arg1) <{funcName = "om_gemmini_mul_f32"
// CHECK-SAME: numOfOutput = 1
// CHECK: return [[OUT]] : memref<1x3x4x4xf32>
