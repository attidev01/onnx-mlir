// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — dynamic batch, split [?,3,8,4] on axis=2 into [?,3,4,4] and [?,3,4,4].
// The output batch dimension is sourced from the input via memref.dim.

func.func @split_dynamic_batch(%x: tensor<?x3x8x4xf32>) -> (tensor<?x3x4x4xf32>, tensor<?x3x4x4xf32>) {
  %split_sz = "onnx.Constant"() {value = dense<[4, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %out0, %out1 = "onnx.Split"(%x, %split_sz) {axis = 2 : si64, num_outputs = 2 : si64} :
      (tensor<?x3x8x4xf32>, tensor<2xi64>) -> (tensor<?x3x4x4xf32>, tensor<?x3x4x4xf32>)
  return %out0, %out1 : tensor<?x3x4x4xf32>, tensor<?x3x4x4xf32>
}

// CHECK-LABEL: func.func @split_dynamic_batch
// Dynamic batch must be sourced from the input at runtime.
// CHECK: [[C0:%.+]] = arith.constant 0 : index
// CHECK: [[N0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x3x8x4xf32>
// CHECK: [[OUT0:%.+]] = memref.alloc([[N0]]) : memref<?x3x4x4xf32>
// CHECK: [[N1:%.+]] = memref.dim %arg0, [[C0]] : memref<?x3x8x4xf32>
// CHECK: [[OUT1:%.+]] = memref.alloc([[N1]]) : memref<?x3x4x4xf32>
// krnl.call: out0, out1, input, axis (i64=2).
// CHECK: "krnl.call"([[OUT0]], [[OUT1]], %arg0, {{%.+}}) <{funcName = "om_gemmini_split_2_f32"
// CHECK-SAME: numOfOutput = 2
// CHECK: return [[OUT0]], [[OUT1]] : memref<?x3x4x4xf32>, memref<?x3x4x4xf32>
