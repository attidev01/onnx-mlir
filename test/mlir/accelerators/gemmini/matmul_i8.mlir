// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

func.func @matmulinteger_i8_i32(%lhs: tensor<4x8xi8>, %rhs: tensor<8x16xi8>) -> tensor<4x16xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.MatMulInteger"(%lhs, %rhs, %none, %none) : (tensor<4x8xi8>, tensor<8x16xi8>, none, none) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>

// CHECK-LABEL: func.func @matmulinteger_i8_i32
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<4x16xi32>
// CHECK: "krnl.call"([[ALLOC]], [[ARG0:%.+]], [[ARG1:%.+]]) <{funcName = "om_gemmini_matmulinteger_i8i8acc32"
// CHECK: return [[ALLOC]] : memref<4x16xi32>
}

// -----

// Scalar int8 zero-points are corrected in the runtime after the Gemmini
// int32 accumulation path, so they are still eligible for the accelerator.
func.func @matmulinteger_with_scalar_zeropoints(%lhs: tensor<4x8xi8>, %rhs: tensor<8x16xi8>,
    %a_zp: tensor<i8>, %b_zp: tensor<i8>) -> tensor<4x16xi32> {
  %0 = "onnx.MatMulInteger"(%lhs, %rhs, %a_zp, %b_zp) : (tensor<4x8xi8>, tensor<8x16xi8>, tensor<i8>, tensor<i8>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>

// CHECK-LABEL: func.func @matmulinteger_with_scalar_zeropoints
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<4x16xi32>
// CHECK: "krnl.call"([[ALLOC]], [[ARG0:%.+]], [[ARG1:%.+]], [[AZP:%.+]], [[BZP:%.+]]) <{funcName = "om_gemmini_matmulinteger_i8i8acc32_zp"
// CHECK: return [[ALLOC]] : memref<4x16xi32>
}

// -----

// Rank-1 zero-points matching the row/column broadcast rules are also
// supported by the Gemmini runtime correction path.
func.func @matmulinteger_with_vector_zeropoints(%lhs: tensor<4x8xi8>, %rhs: tensor<8x16xi8>,
    %a_zp: tensor<4xi8>, %b_zp: tensor<16xi8>) -> tensor<4x16xi32> {
  %0 = "onnx.MatMulInteger"(%lhs, %rhs, %a_zp, %b_zp) : (tensor<4x8xi8>, tensor<8x16xi8>, tensor<4xi8>, tensor<16xi8>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>

// CHECK-LABEL: func.func @matmulinteger_with_vector_zeropoints
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<4x16xi32>
// CHECK: "krnl.call"([[ALLOC]], [[ARG0:%.+]], [[ARG1:%.+]], [[AZP:%.+]], [[BZP:%.+]]) <{funcName = "om_gemmini_matmulinteger_i8i8acc32_zp"
// CHECK: return [[ALLOC]] : memref<4x16xi32>
}

// -----

// Unsupported zero-point ranks still stay on the generic path.
func.func @matmulinteger_with_bad_rank_zeropoints(%lhs: tensor<4x8xi8>, %rhs: tensor<8x16xi8>,
    %a_zp: tensor<?xi8>, %b_zp: tensor<16xi8>) -> tensor<4x16xi32> {
  %0 = "onnx.MatMulInteger"(%lhs, %rhs, %a_zp, %b_zp) : (tensor<4x8xi8>, tensor<8x16xi8>, tensor<?xi8>, tensor<16xi8>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>

// CHECK-LABEL: func.func @matmulinteger_with_bad_rank_zeropoints
// CHECK-NOT: krnl.call "om_gemmini_matmulinteger_i8i8acc32_zp"
}

// -----

func.func @qlinearconv_i8_basic(%x: tensor<1x3x5x5xi8>, %x_scale: tensor<f32>,
    %x_zp: tensor<i8>, %w: tensor<4x3x3x3xi8>, %w_scale: tensor<f32>,
    %w_zp: tensor<i8>, %y_scale: tensor<f32>, %y_zp: tensor<i8>) -> tensor<1x4x3x3xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.QLinearConv"(%x, %x_scale, %x_zp, %w, %w_scale, %w_zp, %y_scale, %y_zp, %none) {group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x5x5xi8>, tensor<f32>, tensor<i8>, tensor<4x3x3x3xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>, none) -> tensor<1x4x3x3xi8>
  return %0 : tensor<1x4x3x3xi8>

// CHECK-LABEL: func.func @qlinearconv_i8_basic
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x4x3x3xi8>
// CHECK: "krnl.call"([[ALLOC]], {{.*}}) <{funcName = "om_gemmini_qlinearconv_i8"
// CHECK: return [[ALLOC]] : memref<1x4x3x3xi8>
}

// -----

func.func @qlinearconv_i8_with_bias(%x: tensor<1x3x5x5xi8>, %x_scale: tensor<f32>,
    %x_zp: tensor<i8>, %w: tensor<4x3x3x3xi8>, %w_scale: tensor<f32>,
    %w_zp: tensor<i8>, %y_scale: tensor<f32>, %y_zp: tensor<i8>,
    %bias: tensor<4xi32>) -> tensor<1x4x3x3xi8> {
  %0 = "onnx.QLinearConv"(%x, %x_scale, %x_zp, %w, %w_scale, %w_zp, %y_scale, %y_zp, %bias) {group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x5x5xi8>, tensor<f32>, tensor<i8>, tensor<4x3x3x3xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>, tensor<4xi32>) -> tensor<1x4x3x3xi8>
  return %0 : tensor<1x4x3x3xi8>

// CHECK-LABEL: func.func @qlinearconv_i8_with_bias
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x4x3x3xi8>
// CHECK: "krnl.call"([[ALLOC]], {{.*}}) <{funcName = "om_gemmini_qlinearconv_i8_bias"
// CHECK: return [[ALLOC]] : memref<1x4x3x3xi8>
}

// -----

// Non-zero input zero-points stay eligible because the runtime folds the
// correction into the bias term before invoking Gemmini.
func.func @qlinearconv_i8_with_bias_and_nonzero_xzp(%x: tensor<1x3x5x5xi8>,
    %w: tensor<4x3x3x3xi8>, %bias: tensor<4xi32>) -> tensor<1x4x3x3xi8> {
  %x_scale = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %x_zp = "onnx.Constant"() {value = dense<2> : tensor<i8>} : () -> tensor<i8>
  %w_scale = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %w_zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
  %y_scale = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %y_zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
  %0 = "onnx.QLinearConv"(%x, %x_scale, %x_zp, %w, %w_scale, %w_zp, %y_scale, %y_zp, %bias) {group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x5x5xi8>, tensor<f32>, tensor<i8>, tensor<4x3x3x3xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>, tensor<4xi32>) -> tensor<1x4x3x3xi8>
  return %0 : tensor<1x4x3x3xi8>

// CHECK-LABEL: func.func @qlinearconv_i8_with_bias_and_nonzero_xzp
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x4x3x3xi8>
// CHECK: "krnl.call"([[ALLOC]], {{.*}}) <{funcName = "om_gemmini_qlinearconv_i8_bias"
// CHECK: return [[ALLOC]] : memref<1x4x3x3xi8>
}

// -----

func.func @qlinearconv_i8_bad_batch(%x: tensor<2x3x5x5xi8>, %x_scale: tensor<f32>,
    %x_zp: tensor<i8>, %w: tensor<4x3x3x3xi8>, %w_scale: tensor<f32>,
    %w_zp: tensor<i8>, %y_scale: tensor<f32>, %y_zp: tensor<i8>) -> tensor<2x4x3x3xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.QLinearConv"(%x, %x_scale, %x_zp, %w, %w_scale, %w_zp, %y_scale, %y_zp, %none) {group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<2x3x5x5xi8>, tensor<f32>, tensor<i8>, tensor<4x3x3x3xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>, none) -> tensor<2x4x3x3xi8>
  return %0 : tensor<2x4x3x3xi8>

// CHECK-LABEL: func.func @qlinearconv_i8_bad_batch
// CHECK-NOT: krnl.call "om_gemmini_qlinearconv_i8"
}

// -----

func.func @qlinearconv_i8_bad_bias_shape(%x: tensor<1x3x5x5xi8>, %x_scale: tensor<f32>,
    %x_zp: tensor<i8>, %w: tensor<4x3x3x3xi8>, %w_scale: tensor<f32>,
    %w_zp: tensor<i8>, %y_scale: tensor<f32>, %y_zp: tensor<i8>,
    %bias: tensor<3xi32>) -> tensor<1x4x3x3xi8> {
  %0 = "onnx.QLinearConv"(%x, %x_scale, %x_zp, %w, %w_scale, %w_zp, %y_scale, %y_zp, %bias) {group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x5x5xi8>, tensor<f32>, tensor<i8>, tensor<4x3x3x3xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>, tensor<3xi32>) -> tensor<1x4x3x3xi8>
  return %0 : tensor<1x4x3x3xi8>

// CHECK-LABEL: func.func @qlinearconv_i8_bad_bias_shape
// CHECK-NOT: krnl.call "om_gemmini_qlinearconv_i8_bias"
}
