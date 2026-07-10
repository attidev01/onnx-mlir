// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s
// RUN: FileCheck --check-prefix=RUNTIME_NEAR %s < %S/../../../../src/Accelerators/Gemmini/Runtime/OMRuntimeGemmini.cpp

// Test 1 — Static [1,3,4,4] → [1,3,8,8], nearest, asymmetric coords, floor rounding.
// coord_mode = 0 (asymmetric), nearest_mode = 0 (floor).

func.func @resize_nearest_static_asym(
    %x: tensor<1x3x4x4xf32>) -> tensor<1x3x8x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%x, %none, %scales, %none) {
    mode = "nearest",
    coordinate_transformation_mode = "asymmetric",
    nearest_mode = "floor"
  } : (tensor<1x3x4x4xf32>, none, tensor<4xf32>, none)
      -> tensor<1x3x8x8xf32>
  return %0 : tensor<1x3x8x8xf32>
}

// CHECK-LABEL: func.func @resize_nearest_static_asym
// Fully static alloc (no dynamic-dim arguments).
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x3x8x8xf32>
// krnl.call: out, x, coord_mode (i64), nearest_mode (i64).
// CHECK: "krnl.call"([[ALLOC]], %arg0, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_resize_nearest_f32"
// CHECK: return [[ALLOC]] : memref<1x3x8x8xf32>

// --------------------------------------------------------------------------
// Test 2 — Dynamic batch [?,3,4,4] → [?,3,8,8], nearest, half_pixel coords,
// round_prefer_floor rounding.  coord_mode = 1, nearest_mode = 2.

func.func @resize_nearest_dynamic_batch(
    %x: tensor<?x3x4x4xf32>) -> tensor<?x3x8x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%x, %none, %scales, %none) {
    mode = "nearest",
    coordinate_transformation_mode = "half_pixel",
    nearest_mode = "round_prefer_floor"
  } : (tensor<?x3x4x4xf32>, none, tensor<4xf32>, none)
      -> tensor<?x3x8x8xf32>
  return %0 : tensor<?x3x8x8xf32>
}

// CHECK-LABEL: func.func @resize_nearest_dynamic_batch
// Dynamic batch: alloc reads dim 0 from the input.
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[N:%.+]] = memref.dim %arg0, [[C0]] : memref<?x3x4x4xf32>
// CHECK: [[ALLOC2:%.+]] = memref.alloc([[N]]) {{.*}}: memref<?x3x8x8xf32>
// CHECK: "krnl.call"([[ALLOC2]], %arg0, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_resize_nearest_f32"
// CHECK: return [[ALLOC2]] : memref<?x3x8x8xf32>

// --------------------------------------------------------------------------
// Test 3 — Static downsample [1,3,8,8] → [1,3,4,4], nearest, align_corners.
// coord_mode = 2 (align_corners), nearest_mode = 2 (round_prefer_floor, default).

func.func @resize_nearest_downsample_align(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 0.5, 0.5]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%x, %none, %scales, %none) {
    mode = "nearest",
    coordinate_transformation_mode = "align_corners"
  } : (tensor<1x3x8x8xf32>, none, tensor<4xf32>, none)
      -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// CHECK-LABEL: func.func @resize_nearest_downsample_align
// CHECK: [[ALLOC3:%.+]] = memref.alloc() {{.*}}: memref<1x3x4x4xf32>
// CHECK: "krnl.call"([[ALLOC3]], %arg0, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_resize_nearest_f32"
// CHECK: return [[ALLOC3]] : memref<1x3x4x4xf32>

// --------------------------------------------------------------------------
// RUNTIME_NEAR-LABEL: void om_gemmini_resize_nearest_f32(
// RUNTIME_NEAR: coord_mode
// RUNTIME_NEAR: nearest_mode
// RUNTIME_NEAR: resize_input_coord
