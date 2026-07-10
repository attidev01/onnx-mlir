// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — Static [1,3,2,4] + [1,3,3,4] → [1,3,5,4], axis=2 (height concat).

func.func @concat_axis2(
    %x0: tensor<1x3x2x4xf32>,
    %x1: tensor<1x3x3x4xf32>) -> tensor<1x3x5x4xf32> {
  %0 = "onnx.Concat"(%x0, %x1) {axis = 2 : si64} :
      (tensor<1x3x2x4xf32>, tensor<1x3x3x4xf32>) -> tensor<1x3x5x4xf32>
  return %0 : tensor<1x3x5x4xf32>
}

// CHECK-LABEL: func.func @concat_axis2
// Fully static alloc.
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x3x5x4xf32>
// krnl.call: out, x0, x1, axis (i64=2).
// CHECK: "krnl.call"([[ALLOC]], %arg0, %arg1, {{%.+}}) <{funcName = "om_gemmini_concat_f32"
// CHECK: return [[ALLOC]] : memref<1x3x5x4xf32>
