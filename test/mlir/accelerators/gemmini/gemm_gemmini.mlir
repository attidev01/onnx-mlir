// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_GEMM %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

func.func @gemm_f32_dynamic_batch(%a: tensor<?x512xf32>, %b: tensor<512x1000xf32>,
    %c: tensor<1000xf32>) -> tensor<?x1000xf32> {
  %0 = "onnx.Gemm"(%a, %b, %c) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x512xf32>, tensor<512x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>
  return %0 : tensor<?x1000xf32>
}

// CHECK-LABEL: func.func @gemm_f32_dynamic_batch
// CHECK: [[M:%.+]] = memref.dim %arg0, %c0 : memref<?x512xf32>
// CHECK: [[ALLOC:%.+]] = memref.alloc([[M]]) {{.*}}: memref<?x1000xf32>
// CHECK: "krnl.call"([[ALLOC]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) <{funcName = "om_gemmini_gemm_f32_bias"
// CHECK: return [[ALLOC]] : memref<?x1000xf32>

// RUNTIME_GEMM-LABEL: static void om_gemmini_quantized_matmul_f32_ws
// RUNTIME_GEMM: om_gemmini_tiled_matmul_i8i8acc32_ws(
// RUNTIME_GEMM-LABEL: static void om_gemmini_gemm_f32_impl
// RUNTIME_GEMM-NOT: acc +=
// RUNTIME_GEMM: om_gemmini_quantized_matmul_f32_ws(
// RUNTIME_GEMM: void om_gemmini_gemm_f32(
