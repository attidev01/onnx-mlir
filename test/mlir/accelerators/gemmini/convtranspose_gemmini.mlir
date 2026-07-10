// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_CT %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

// Test 1 — Static ConvTranspose [1, 3, 4, 4] → [1, 16, 7, 7]
// kernel 3×3, stride 2, pad 1, output_pad 0 (default).
// ONNX formula: H_out = (H - 1)*stride + kH - 2*pad + output_pad
//                     = (4-1)*2 + 3 - 2 + 0 = 7.

func.func @convtranspose_f32_static_bias(
    %x: tensor<1x3x4x4xf32>,
    %w: tensor<3x16x3x3xf32>,
    %b: tensor<16xf32>) -> tensor<1x16x7x7xf32> {
  %0 = "onnx.ConvTranspose"(%x, %w, %b) {
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [2, 2]
  } : (tensor<1x3x4x4xf32>, tensor<3x16x3x3xf32>, tensor<16xf32>)
      -> tensor<1x16x7x7xf32>
  return %0 : tensor<1x16x7x7xf32>
}

// CHECK-LABEL: func.func @convtranspose_f32_static_bias
// Output is fully static — alloc needs no dynamic-dim arguments.
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x16x7x7xf32>
// Runtime call with stride=2, pad=1, output_pad=0 passed as i64 constants.
// CHECK: "krnl.call"([[ALLOC]], %arg0, %arg1, %arg2, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_convtranspose_f32_bias"
// CHECK: return [[ALLOC]] : memref<1x16x7x7xf32>

// --------------------------------------------------------------------------
// Test 2 — No-bias variant (om_gemmini_convtranspose_f32)

func.func @convtranspose_f32_static_no_bias(
    %x: tensor<1x3x4x4xf32>,
    %w: tensor<3x16x3x3xf32>) -> tensor<1x16x7x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%x, %w, %none) {
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [2, 2]
  } : (tensor<1x3x4x4xf32>, tensor<3x16x3x3xf32>, none)
      -> tensor<1x16x7x7xf32>
  return %0 : tensor<1x16x7x7xf32>
}

// CHECK-LABEL: func.func @convtranspose_f32_static_no_bias
// CHECK: [[ALLOC2:%.+]] = memref.alloc() {{.*}}: memref<1x16x7x7xf32>
// No bias — function name is om_gemmini_convtranspose_f32 (no _bias suffix).
// CHECK: "krnl.call"([[ALLOC2]], %arg0, %arg1, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_convtranspose_f32"
// CHECK: return [[ALLOC2]] : memref<1x16x7x7xf32>

// --------------------------------------------------------------------------
// Test 3 — Dynamic batch and spatial: [?, 3, ?, ?] → [?, 16, ?, ?]
// Same kernel 3×3, stride 2, pad 1, output_pad 0.
// H_out = H_in * stride + (kH - stride - 2*pad + output_pad)
//       = H_in * 2 + (3 - 2 - 2 + 0)
//       = H_in * 2 + (-1)
// So adj = -1, stride_const = 2 for both spatial dims.

func.func @convtranspose_f32_dynamic_spatial(
    %x: tensor<?x3x?x?xf32>,
    %w: tensor<3x16x3x3xf32>,
    %b: tensor<16xf32>) -> tensor<?x16x?x?xf32> {
  %0 = "onnx.ConvTranspose"(%x, %w, %b) {
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [2, 2]
  } : (tensor<?x3x?x?xf32>, tensor<3x16x3x3xf32>, tensor<16xf32>)
      -> tensor<?x16x?x?xf32>
  return %0 : tensor<?x16x?x?xf32>
}

// CHECK-LABEL: func.func @convtranspose_f32_dynamic_spatial

// Runtime dim reads: batch (dim 0), input H (dim 2), input W (dim 3).
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG: [[N:%.+]]  = memref.dim %arg0, [[C0]] : memref<?x3x?x?xf32>
// CHECK-DAG: [[IH:%.+]] = memref.dim %arg0, [[C2]] : memref<?x3x?x?xf32>
// CHECK-DAG: [[IW:%.+]] = memref.dim %arg0, [[C3]] : memref<?x3x?x?xf32>

// i64 stride (2) and adjustment constant (-1 = kH - stride - 2*pad + output_pad).
// CHECK-DAG: [[STR:%.+]] = arith.constant 2 : i64
// CHECK-DAG: [[ADJ:%.+]] = arith.constant -1 : i64

// Output H: cast → multiply by stride → add adj → cast back.
// CHECK-DAG: [[IH64:%.+]]  = arith.index_cast [[IH]] : index to i64
// CHECK-DAG: [[IW64:%.+]]  = arith.index_cast [[IW]] : index to i64
// CHECK-DAG: [[PH:%.+]]   = arith.muli [[IH64]], [[STR]] : i64
// CHECK-DAG: [[OH64:%.+]] = arith.addi [[PH]], [[ADJ]] : i64
// CHECK-DAG: [[OH:%.+]]   = arith.index_cast [[OH64]] : i64 to index
// CHECK-DAG: [[PW:%.+]]   = arith.muli [[IW64]], [[STR]] : i64
// CHECK-DAG: [[OW64:%.+]] = arith.addi [[PW]], [[ADJ]] : i64
// CHECK-DAG: [[OW:%.+]]   = arith.index_cast [[OW64]] : i64 to index

// Output allocated with all three dynamic dims.
// CHECK: [[ALLOC3:%.+]] = memref.alloc([[N]], [[OH]], [[OW]]) {{.*}}: memref<?x16x?x?xf32>

// Single runtime call; stride, pad, output_pad passed as i64 constants.
// CHECK: "krnl.call"([[ALLOC3]], %arg0, %arg1, %arg2, {{%.+}}, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_convtranspose_f32_bias"
// CHECK: return [[ALLOC3]] : memref<?x16x?x?xf32>

// --------------------------------------------------------------------------
// RUNTIME_CT checks: verify that the runtime implementation uses the
// Gemmini-backed quantised matmul path (no plain acc += in the inner loop).

// RUNTIME_CT-LABEL: static void om_gemmini_convtranspose_f32_impl
// RUNTIME_CT-NOT: acc +=
// RUNTIME_CT: om_gemmini_quantized_matmul_f32_ws(
// RUNTIME_CT: void om_gemmini_convtranspose_f32(
// RUNTIME_CT: void om_gemmini_convtranspose_f32_bias(
