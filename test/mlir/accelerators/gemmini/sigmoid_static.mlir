// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — static [1,3,4,4] sigmoid → [1,3,4,4].

func.func @sigmoid_static(%x: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
  %y = "onnx.Sigmoid"(%x) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
  return %y : tensor<1x3x4x4xf32>
}

// CHECK-LABEL: func.func @sigmoid_static
// CHECK: [[OUT:%.+]] = memref.alloc() {{.*}}: memref<1x3x4x4xf32>
// CHECK: "krnl.call"([[OUT]], %arg0) <{funcName = "om_gemmini_sigmoid_f32"
// CHECK-SAME: numOfOutput = 1
// CHECK: return [[OUT]] : memref<1x3x4x4xf32>
