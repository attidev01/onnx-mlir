// RUN: onnx-mlir-opt --maccel=Gemmini --shape-inference --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — Dynamic batch [?,4,6,4] sliced on axes=[2,3] with starts=[1,0] ends=[3,4]
//   H slice: 1→3 (H_in=6) → H_out=2
//   W slice: 0→4 (W_in=4) → W_out=4 (identity)
//   Batch is NOT in axes → output batch stays dynamic.
// Output: [?,4,2,4].
//
// --shape-inference is required before --convert-onnx-to-krnl to materialize
// the default NoneValue for the absent 'steps' input; otherwise DimAnalysis
// crashes when it tries to read the element type of a none-typed Value.

func.func @slice_dynamic_batch(%data: tensor<?x4x6x4xf32>) -> tensor<?x4x2x4xf32> {
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
  %ends   = "onnx.Constant"() {value = dense<[3, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %axes   = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %noval  = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %noval) :
      (tensor<?x4x6x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<?x4x2x4xf32>
  return %0 : tensor<?x4x2x4xf32>
}

// CHECK-LABEL: func.func @slice_dynamic_batch
// Dynamic batch: alloc reads dim 0 from input.
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[N:%.+]] = memref.dim %arg0, [[C0]] : memref<?x4x6x4xf32>
// CHECK: [[ALLOC:%.+]] = memref.alloc([[N]]) {{.*}}: memref<?x4x2x4xf32>
// krnl.call: out, data, then 12 i64 constants.
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{.*}}) <{funcName = "om_gemmini_slice_f32"
// CHECK: return [[ALLOC]] : memref<?x4x2x4xf32>
