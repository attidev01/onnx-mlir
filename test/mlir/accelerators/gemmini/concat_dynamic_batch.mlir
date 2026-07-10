// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Test — Dynamic batch [?,3,4,4] + [?,3,4,4] → [?,3,8,4], axis=2 (height concat).
// Batch dimension is dynamic; axis=2 so the output batch matches input batch.

func.func @concat_dynamic_batch(
    %x0: tensor<?x3x4x4xf32>,
    %x1: tensor<?x3x4x4xf32>) -> tensor<?x3x8x4xf32> {
  %0 = "onnx.Concat"(%x0, %x1) {axis = 2 : si64} :
      (tensor<?x3x4x4xf32>, tensor<?x3x4x4xf32>) -> tensor<?x3x8x4xf32>
  return %0 : tensor<?x3x8x4xf32>
}

// CHECK-LABEL: func.func @concat_dynamic_batch
// Dynamic batch: alloc reads dim 0 from input 0.
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[N:%.+]] = memref.dim %arg0, [[C0]] : memref<?x3x4x4xf32>
// CHECK: [[ALLOC:%.+]] = memref.alloc([[N]]) {{.*}}: memref<?x3x8x4xf32>
// krnl.call: out, x0, x1, axis (i64=2).
// CHECK: "krnl.call"([[ALLOC]], %arg0, %arg1, {{%.+}}) <{funcName = "om_gemmini_concat_f32"
// CHECK: return [[ALLOC]] : memref<?x3x8x4xf32>
