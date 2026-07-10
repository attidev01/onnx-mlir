// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — dynamic batch [?,3,4,4] sigmoid → [?,3,4,4].
// The output batch dimension must be sourced from the input via memref.dim.

func.func @sigmoid_dynamic_batch(%x: tensor<?x3x4x4xf32>) -> tensor<?x3x4x4xf32> {
  %y = "onnx.Sigmoid"(%x) : (tensor<?x3x4x4xf32>) -> tensor<?x3x4x4xf32>
  return %y : tensor<?x3x4x4xf32>
}

// CHECK-LABEL: func.func @sigmoid_dynamic_batch
// Dynamic alloc: batch dim comes from memref.dim of the input.
// CHECK: [[DIM:%.+]] = memref.dim %arg0, {{%.+}} : memref<?x3x4x4xf32>
// CHECK: [[OUT:%.+]] = memref.alloc([[DIM]]) {{.*}}: memref<?x3x4x4xf32>
// CHECK: "krnl.call"([[OUT]], %arg0) <{funcName = "om_gemmini_sigmoid_f32"
// CHECK-SAME: numOfOutput = 1
// CHECK: return [[OUT]] : memref<?x3x4x4xf32>
