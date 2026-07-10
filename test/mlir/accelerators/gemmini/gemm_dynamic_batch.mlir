// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Verify dynamic batch (dim 0) Gemm lowering to om_gemmini_gemm_f32_bias.
// Uses MobileNetV2-scale hidden dimensions: M=?, K=1024, N=512.

func.func @gemm_f32_dynamic_batch_large(%a: tensor<?x1024xf32>,
    %b: tensor<1024x512xf32>, %c: tensor<512xf32>) -> tensor<?x512xf32> {
  %0 = "onnx.Gemm"(%a, %b, %c) {
    alpha = 1.000000e+00 : f32,
    beta = 1.000000e+00 : f32,
    transA = 0 : si64,
    transB = 0 : si64
  } : (tensor<?x1024xf32>, tensor<1024x512xf32>, tensor<512xf32>)
      -> tensor<?x512xf32>
  return %0 : tensor<?x512xf32>
}

// CHECK-LABEL: func.func @gemm_f32_dynamic_batch_large
// CHECK:       [[M:%.+]] = memref.dim %arg0, %c0 : memref<?x1024xf32>
// CHECK:       [[ALLOC:%.+]] = memref.alloc([[M]]) {{.*}}: memref<?x512xf32>
// CHECK:       "krnl.call"([[ALLOC]], {{.*}}) <{funcName = "om_gemmini_gemm_f32_bias"
// CHECK:       return [[ALLOC]] : memref<?x512xf32>
