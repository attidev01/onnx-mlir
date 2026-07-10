// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

func.func @matmul_fp32_hardware(%lhs: tensor<16x32xf32>, %rhs: tensor<32x16xf32>) -> tensor<16x16xf32> {
  %0 = "onnx.MatMul"(%lhs, %rhs) : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @matmul_fp32_hardware
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<16x16xf32>
// CHECK: "krnl.call"([[ALLOC]], {{.*}}, {{.*}}) <{funcName = "om_gemmini_matmul_f32_hw"
// CHECK-NOT: funcName = "om_gemmini_matmul_f32"
// CHECK: return [[ALLOC]] : memref<16x16xf32>
