// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_PAD_EDGE %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

func.func @pad_edge_f32_static(
    %x: tensor<1x3x4x4xf32>) -> tensor<1x3x6x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %pads = "onnx.Constant"() {value = dense<[0, 0, 1, 2, 0, 0, 1, 2]> : tensor<8xi64>} : () -> tensor<8xi64>
  %0 = "onnx.Pad"(%x, %pads, %none, %none) {mode = "edge"} :
      (tensor<1x3x4x4xf32>, tensor<8xi64>, none, none) -> tensor<1x3x6x8xf32>
  return %0 : tensor<1x3x6x8xf32>
}

// CHECK-LABEL: func.func @pad_edge_f32_static
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x3x6x8xf32>
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_pad_edge_f32"
// CHECK-NOT: acc +=
// CHECK: return [[ALLOC]] : memref<1x3x6x8xf32>

// RUNTIME_PAD_EDGE-LABEL: void om_gemmini_pad_edge_f32(
// RUNTIME_PAD_EDGE: pad_clamp_index
