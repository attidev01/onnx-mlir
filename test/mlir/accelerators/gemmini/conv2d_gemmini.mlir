// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_CONV %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

func.func @conv2d_f32_dynamic_batch(%x: tensor<?x3x5x5xf32>, %w: tensor<4x3x3x3xf32>,
    %b: tensor<4xf32>) -> tensor<?x4x3x3xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<?x3x5x5xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<?x4x3x3xf32>
  return %0 : tensor<?x4x3x3xf32>
}

// CHECK-LABEL: func.func @conv2d_f32_dynamic_batch
// CHECK: [[N:%.+]] = memref.dim %arg0, %c0 : memref<?x3x5x5xf32>
// CHECK: [[ALLOC:%.+]] = memref.alloc([[N]]) {{.*}}: memref<?x4x3x3xf32>
// CHECK: "krnl.call"([[ALLOC]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) <{funcName = "om_gemmini_conv_f32_bias"
// CHECK: return [[ALLOC]] : memref<?x4x3x3xf32>

// RUNTIME_CONV-LABEL: static void om_gemmini_quantized_matmul_f32_ws
// RUNTIME_CONV: om_gemmini_tiled_matmul_i8i8acc32_ws(
// RUNTIME_CONV-LABEL: static void om_gemmini_conv_f32_impl
// RUNTIME_CONV-NOT: acc +=
// RUNTIME_CONV: om_gemmini_quantized_matmul_f32_ws(
// RUNTIME_CONV: void om_gemmini_conv_f32(
