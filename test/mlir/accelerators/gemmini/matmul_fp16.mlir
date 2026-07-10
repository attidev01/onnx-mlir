// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

func.func @matmul_fp16_hardware(%lhs: tensor<16x32xf16>, %rhs: tensor<32x16xf16>) -> tensor<16x16xf16> {
  %0 = "onnx.MatMul"(%lhs, %rhs) : (tensor<16x32xf16>, tensor<32x16xf16>) -> tensor<16x16xf16>
  return %0 : tensor<16x16xf16>
}

// CHECK-LABEL: func.func @matmul_fp16_hardware
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<16x16xf16>
// CHECK: "krnl.call"([[ALLOC]], {{.*}}, {{.*}}) <{funcName = "om_gemmini_matmul_f16_hw"
// CHECK-NOT: funcName = "om_gemmini_matmul_f32_hw"
// CHECK: return [[ALLOC]] : memref<16x16xf16>
