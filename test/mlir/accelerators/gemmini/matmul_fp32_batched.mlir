// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

func.func @matmul_fp32_batched_hardware(%lhs: tensor<?x?x256xf32>, %rhs: tensor<256x64xf32>) -> tensor<?x?x64xf32> {
  %0 = "onnx.MatMul"(%lhs, %rhs) : (tensor<?x?x256xf32>, tensor<256x64xf32>) -> tensor<?x?x64xf32>
  return %0 : tensor<?x?x64xf32>
}

// CHECK-LABEL: func.func @matmul_fp32_batched_hardware
// CHECK: [[D0:%.+]] = memref.dim %arg0, %c0 : memref<?x?x256xf32>
// CHECK: [[D1:%.+]] = memref.dim %arg0, %c1 : memref<?x?x256xf32>
// CHECK: [[ALLOC:%.+]] = memref.alloc([[D0]], [[D1]]) {{.*}}: memref<?x?x64xf32>
// CHECK: "krnl.call"([[ALLOC]], {{.*}}, {{.*}}) <{funcName = "om_gemmini_matmul_f32_nd_hw"
// CHECK: return [[ALLOC]] : memref<?x?x64xf32>
