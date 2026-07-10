// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_CONCAT %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

// Test — Static [1,3,4,4] + [2,3,4,4] → [3,3,4,4], axis=0 (batch concat).
// Both inputs have static batch; axis=0.

func.func @concat_axis0(
    %x0: tensor<1x3x4x4xf32>,
    %x1: tensor<2x3x4x4xf32>) -> tensor<3x3x4x4xf32> {
  %0 = "onnx.Concat"(%x0, %x1) {axis = 0 : si64} :
      (tensor<1x3x4x4xf32>, tensor<2x3x4x4xf32>) -> tensor<3x3x4x4xf32>
  return %0 : tensor<3x3x4x4xf32>
}

// CHECK-LABEL: func.func @concat_axis0
// Fully static alloc — no dynamic-dim arguments.
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<3x3x4x4xf32>
// krnl.call: out, x0, x1, axis (i64).
// CHECK: "krnl.call"([[ALLOC]], %arg0, %arg1, {{%.+}}) <{funcName = "om_gemmini_concat_f32"
// CHECK: return [[ALLOC]] : memref<3x3x4x4xf32>

// --------------------------------------------------------------------------
// RUNTIME_CONCAT-LABEL: void om_gemmini_concat_f32(
// RUNTIME_CONCAT: axis
// RUNTIME_CONCAT: off0
