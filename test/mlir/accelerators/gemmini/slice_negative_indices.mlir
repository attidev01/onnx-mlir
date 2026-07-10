// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — Static [1,2,4,4] sliced on W axis=[3] start=-3 end=-1 → [1,2,4,2].
// Negative indices: -3+4=1, -1+4=3 → two elements starting at W=1.

func.func @slice_negative_indices(%data: tensor<1x2x4x4xf32>) -> tensor<1x2x4x2xf32> {
  %starts = "onnx.Constant"() {value = dense<[-3]> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends   = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes   = "onnx.Constant"() {value = dense<[3]>  : tensor<1xi64>} : () -> tensor<1xi64>
  %noval  = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %noval) :
      (tensor<1x2x4x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, none) -> tensor<1x2x4x2xf32>
  return %0 : tensor<1x2x4x2xf32>
}

// CHECK-LABEL: func.func @slice_negative_indices
// Fully static alloc for [1,2,4,2].
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x2x4x2xf32>
// krnl.call passes raw negative constants; runtime normalizes them.
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{.*}}) <{funcName = "om_gemmini_slice_f32"
// CHECK: return [[ALLOC]] : memref<1x2x4x2xf32>
