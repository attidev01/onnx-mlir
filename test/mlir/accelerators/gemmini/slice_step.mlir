// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — Static [1,2,8,4] sliced on H axis=[2] start=0 end=8 step=2 → [1,2,4,4].
// Step=2: every other row → H_out = (8-0+2-1)/2 = 4.

func.func @slice_step(%data: tensor<1x2x8x4xf32>) -> tensor<1x2x4x4xf32> {
  %starts = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends   = "onnx.Constant"() {value = dense<[8]> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes   = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %steps  = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %steps) :
      (tensor<1x2x8x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>
}

// CHECK-LABEL: func.func @slice_step
// Fully static alloc for [1,2,4,4].
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x2x4x4xf32>
// krnl.call: H step constant equals 2.
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{.*}}) <{funcName = "om_gemmini_slice_f32"
// CHECK: return [[ALLOC]] : memref<1x2x4x4xf32>
