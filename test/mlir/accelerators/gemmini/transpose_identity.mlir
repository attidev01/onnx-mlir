// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_TRANSPOSE %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

func.func @transpose_identity(
    %x: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
  %0 = "onnx.Transpose"(%x) {perm = [0, 1, 2, 3]} :
      (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// CHECK-LABEL: func.func @transpose_identity
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x3x4x4xf32>
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_transpose_f32_hw"
// CHECK: return [[ALLOC]] : memref<1x3x4x4xf32>

// RUNTIME_TRANSPOSE-LABEL: void om_gemmini_transpose_f32_hw(
// RUNTIME_TRANSPOSE: om_gemmini_transpose_f32_scalar_impl(out, x, perm);
