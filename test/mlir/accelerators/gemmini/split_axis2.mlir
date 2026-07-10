// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — static [1,3,8,4] split on height axis (axis=2) into [1,3,3,4] and [1,3,5,4].

func.func @split_axis2(%x: tensor<1x3x8x4xf32>) -> (tensor<1x3x3x4xf32>, tensor<1x3x5x4xf32>) {
  %split_sz = "onnx.Constant"() {value = dense<[3, 5]> : tensor<2xi64>} : () -> tensor<2xi64>
  %out0, %out1 = "onnx.Split"(%x, %split_sz) {axis = 2 : si64, num_outputs = 2 : si64} :
      (tensor<1x3x8x4xf32>, tensor<2xi64>) -> (tensor<1x3x3x4xf32>, tensor<1x3x5x4xf32>)
  return %out0, %out1 : tensor<1x3x3x4xf32>, tensor<1x3x5x4xf32>
}

// CHECK-LABEL: func.func @split_axis2
// Both outputs are fully static allocs.
// CHECK-DAG: [[OUT0:%.+]] = memref.alloc() {{.*}}: memref<1x3x3x4xf32>
// CHECK-DAG: [[OUT1:%.+]] = memref.alloc() {{.*}}: memref<1x3x5x4xf32>
// krnl.call passes out0, out1, input, axis (i64=2).
// CHECK: "krnl.call"([[OUT0]], [[OUT1]], %arg0, {{%.+}}) <{funcName = "om_gemmini_split_2_f32"
// CHECK-SAME: numOfOutput = 2
// CHECK: return [[OUT0]], [[OUT1]] : memref<1x3x3x4xf32>, memref<1x3x5x4xf32>
