// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — static [1,6,4,4] split on channel axis (axis=1) into [1,2,4,4] and [1,4,4,4].

func.func @split_axis1(%x: tensor<1x6x4x4xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x4x4x4xf32>) {
  %split_sz = "onnx.Constant"() {value = dense<[2, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %out0, %out1 = "onnx.Split"(%x, %split_sz) {axis = 1 : si64, num_outputs = 2 : si64} :
      (tensor<1x6x4x4xf32>, tensor<2xi64>) -> (tensor<1x2x4x4xf32>, tensor<1x4x4x4xf32>)
  return %out0, %out1 : tensor<1x2x4x4xf32>, tensor<1x4x4x4xf32>
}

// CHECK-LABEL: func.func @split_axis1
// Both outputs are fully static allocs.
// CHECK-DAG: [[OUT0:%.+]] = memref.alloc() {{.*}}: memref<1x2x4x4xf32>
// CHECK-DAG: [[OUT1:%.+]] = memref.alloc() {{.*}}: memref<1x4x4x4xf32>
// krnl.call passes out0, out1, input, axis (i64=1).
// CHECK: "krnl.call"([[OUT0]], [[OUT1]], %arg0, {{%.+}}) <{funcName = "om_gemmini_split_2_f32"
// CHECK-SAME: numOfOutput = 2
// CHECK: return [[OUT0]], [[OUT1]] : memref<1x2x4x4xf32>, memref<1x4x4x4xf32>
