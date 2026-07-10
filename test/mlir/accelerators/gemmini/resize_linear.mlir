// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_LIN %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

// Test 1 — Static [1,3,4,4] → [1,3,8,8], linear, half_pixel coords.
// coord_mode = 1 (half_pixel).

func.func @resize_linear_static_half_pixel(
    %x: tensor<1x3x4x4xf32>) -> tensor<1x3x8x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%x, %none, %scales, %none) {
    mode = "linear",
    coordinate_transformation_mode = "half_pixel"
  } : (tensor<1x3x4x4xf32>, none, tensor<4xf32>, none)
      -> tensor<1x3x8x8xf32>
  return %0 : tensor<1x3x8x8xf32>
}

// CHECK-LABEL: func.func @resize_linear_static_half_pixel
// Fully static alloc.
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x3x8x8xf32>
// krnl.call for linear: out, x, coord_mode (i64) — no nearest_mode argument.
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{%.+}}) <{funcName = "om_gemmini_resize_linear_f32"
// CHECK: return [[ALLOC]] : memref<1x3x8x8xf32>

// --------------------------------------------------------------------------
// Test 2 — Dynamic batch [?,3,4,4] → [?,3,8,8], linear, align_corners.
// coord_mode = 2 (align_corners).

func.func @resize_linear_dynamic_batch(
    %x: tensor<?x3x4x4xf32>) -> tensor<?x3x8x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%x, %none, %scales, %none) {
    mode = "linear",
    coordinate_transformation_mode = "align_corners"
  } : (tensor<?x3x4x4xf32>, none, tensor<4xf32>, none)
      -> tensor<?x3x8x8xf32>
  return %0 : tensor<?x3x8x8xf32>
}

// CHECK-LABEL: func.func @resize_linear_dynamic_batch
// Dynamic batch: alloc reads dim 0 from the input.
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[N:%.+]] = memref.dim %arg0, [[C0]] : memref<?x3x4x4xf32>
// CHECK: [[ALLOC2:%.+]] = memref.alloc([[N]]) {{.*}}: memref<?x3x8x8xf32>
// CHECK: "krnl.call"([[ALLOC2]], %arg0, {{%.+}}) <{funcName = "om_gemmini_resize_linear_f32"
// CHECK: return [[ALLOC2]] : memref<?x3x8x8xf32>

// --------------------------------------------------------------------------
// Test 3 — Static [1,3,4,4] → [1,3,4,4] identity, linear, asymmetric.
// Scales = [1.0,1.0,1.0,1.0]: no spatial change.  coord_mode = 0.

func.func @resize_linear_identity(
    %x: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%x, %none, %scales, %none) {
    mode = "linear",
    coordinate_transformation_mode = "asymmetric"
  } : (tensor<1x3x4x4xf32>, none, tensor<4xf32>, none)
      -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// CHECK-LABEL: func.func @resize_linear_identity
// CHECK: [[ALLOC3:%.+]] = memref.alloc() {{.*}}: memref<1x3x4x4xf32>
// CHECK: "krnl.call"([[ALLOC3]], %arg0, {{%.+}}) <{funcName = "om_gemmini_resize_linear_f32"
// CHECK: return [[ALLOC3]] : memref<1x3x4x4xf32>

// --------------------------------------------------------------------------
// RUNTIME_LIN-LABEL: void om_gemmini_resize_linear_f32(
// RUNTIME_LIN: coord_mode
// RUNTIME_LIN: resize_input_coord
