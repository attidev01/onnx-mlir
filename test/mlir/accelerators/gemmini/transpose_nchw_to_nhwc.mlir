// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

func.func @transpose_nchw_to_nhwc(
    %x: tensor<1x3x4x5xf32>) -> tensor<1x4x5x3xf32> {
  %0 = "onnx.Transpose"(%x) {perm = [0, 2, 3, 1]} :
      (tensor<1x3x4x5xf32>) -> tensor<1x4x5x3xf32>
  return %0 : tensor<1x4x5x3xf32>
}

// CHECK-LABEL: func.func @transpose_nchw_to_nhwc
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x4x5x3xf32>
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_transpose_f32_hw"
// CHECK: return [[ALLOC]] : memref<1x4x5x3xf32>
