// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_PAD_CONST %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

func.func @pad_constant_f32_static(
    %x: tensor<1x3x4x4xf32>) -> tensor<1x3x6x8xf32> {
  %pads = "onnx.Constant"() {value = dense<[0, 0, 1, 2, 0, 0, 1, 2]> : tensor<8xi64>} : () -> tensor<8xi64>
  %constant = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %axes = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%x, %pads, %constant, %axes) {mode = "constant"} :
      (tensor<1x3x4x4xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x3x6x8xf32>
  return %0 : tensor<1x3x6x8xf32>
}

// CHECK-LABEL: func.func @pad_constant_f32_static
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x3x6x8xf32>
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_pad_constant_f32"
// CHECK-NOT: acc +=
// CHECK: return [[ALLOC]] : memref<1x3x6x8xf32>

// RUNTIME_PAD_CONST-LABEL: void om_gemmini_pad_constant_f32(
// RUNTIME_PAD_CONST: pad_left
// RUNTIME_PAD_CONST: 0.0f
