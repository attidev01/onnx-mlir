// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — static [1,6,4,4] split on axis=1 into three equal [1,2,4,4] outputs.

func.func @split_3_outputs(%x: tensor<1x6x4x4xf32>)
    -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
  %split_sz = "onnx.Constant"() {value = dense<[2, 2, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
  %out0, %out1, %out2 = "onnx.Split"(%x, %split_sz) {axis = 1 : si64, num_outputs = 3 : si64} :
      (tensor<1x6x4x4xf32>, tensor<3xi64>)
      -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
  return %out0, %out1, %out2
      : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
}

// CHECK-LABEL: func.func @split_3_outputs
// Three fully static allocs.
// CHECK-COUNT-3: memref.alloc() {{.*}}: memref<1x2x4x4xf32>
// krnl.call dispatches to the 3-output runtime function.
// CHECK: "krnl.call"({{.*}}) <{funcName = "om_gemmini_split_3_f32"
// CHECK-SAME: numOfOutput = 3
