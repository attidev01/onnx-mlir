// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — Static [1,4,4,4] sliced on H axis=[2] start=1 end=3 → [1,4,2,4].

func.func @slice_start_end(%data: tensor<1x4x4x4xf32>) -> tensor<1x4x2x4xf32> {
  %starts = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %ends   = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  %axes   = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %noval  = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %noval) :
      (tensor<1x4x4x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, none) -> tensor<1x4x2x4xf32>
  return %0 : tensor<1x4x2x4xf32>
}

// CHECK-LABEL: func.func @slice_start_end
// Fully static alloc for [1,4,2,4].
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x4x2x4xf32>
// krnl.call: out, data, then 12 i64 constants (sn,en,tn, sc,ec,tc, sh,eh,th, sw,ew,tw).
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{.*}}) <{funcName = "om_gemmini_slice_f32"
// CHECK: return [[ALLOC]] : memref<1x4x2x4xf32>
