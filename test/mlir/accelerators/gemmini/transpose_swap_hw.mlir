// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

func.func @transpose_swap_hw(
    %x: tensor<1x3x4x5xf32>) -> tensor<1x3x5x4xf32> {
  %0 = "onnx.Transpose"(%x) {perm = [0, 1, 3, 2]} :
      (tensor<1x3x4x5xf32>) -> tensor<1x3x5x4xf32>
  return %0 : tensor<1x3x5x4xf32>
}

// CHECK-LABEL: func.func @transpose_swap_hw
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x3x5x4xf32>
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_transpose_f32_hw"
// CHECK: return [[ALLOC]] : memref<1x3x5x4xf32>
