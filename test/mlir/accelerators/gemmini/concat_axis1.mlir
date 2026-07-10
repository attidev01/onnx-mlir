// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — Static [1,2,4,4] + [1,3,4,4] → [1,5,4,4], axis=1 (channel concat).

func.func @concat_axis1(
    %x0: tensor<1x2x4x4xf32>,
    %x1: tensor<1x3x4x4xf32>) -> tensor<1x5x4x4xf32> {
  %0 = "onnx.Concat"(%x0, %x1) {axis = 1 : si64} :
      (tensor<1x2x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x5x4x4xf32>
  return %0 : tensor<1x5x4x4xf32>
}

// CHECK-LABEL: func.func @concat_axis1
// Fully static alloc.
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x5x4x4xf32>
// krnl.call: out, x0, x1, axis (i64=1).
// CHECK: "krnl.call"([[ALLOC]], %arg0, %arg1, {{%.+}}) <{funcName = "om_gemmini_concat_f32"
// CHECK: return [[ALLOC]] : memref<1x5x4x4xf32>
