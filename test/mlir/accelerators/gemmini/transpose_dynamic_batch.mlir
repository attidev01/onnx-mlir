// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

func.func @transpose_dynamic_batch(
    %x: tensor<?x3x4x5xf32>) -> tensor<?x4x5x3xf32> {
  %0 = "onnx.Transpose"(%x) {perm = [0, 2, 3, 1]} :
      (tensor<?x3x4x5xf32>) -> tensor<?x4x5x3xf32>
  return %0 : tensor<?x4x5x3xf32>
}

// CHECK-LABEL: func.func @transpose_dynamic_batch
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[N:%.+]] = memref.dim %arg0, [[C0]] : memref<?x3x4x5xf32>
// CHECK: [[ALLOC:%.+]] = memref.alloc([[N]]) {{.*}}: memref<?x4x5x3xf32>
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_transpose_f32_hw"
// CHECK: return [[ALLOC]] : memref<?x4x5x3xf32>
