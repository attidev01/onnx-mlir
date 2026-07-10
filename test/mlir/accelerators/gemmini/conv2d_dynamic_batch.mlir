// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Verify dynamic batch (dim 0) Conv lowering to om_gemmini_conv_f32_bias.
// Uses ResNet-18 first-conv dimensions: 7x7 kernel, stride 2, pad 3.
// Output spatial: floor((224 + 2*3 - 7)/2) + 1 = 112.

func.func @conv2d_f32_dynamic_batch_resnet(%x: tensor<?x3x224x224xf32>,
    %w: tensor<64x3x7x7xf32>, %b: tensor<64xf32>) -> tensor<?x64x112x112xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
    group = 1 : si64,
    kernel_shape = [7, 7],
    pads = [3, 3, 3, 3],
    strides = [2, 2]
  } : (tensor<?x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>)
      -> tensor<?x64x112x112xf32>
  return %0 : tensor<?x64x112x112xf32>
}

// CHECK-LABEL: func.func @conv2d_f32_dynamic_batch_resnet
// CHECK:       [[N:%.+]] = memref.dim %arg0, %c0 : memref<?x3x224x224xf32>
// CHECK:       [[ALLOC:%.+]] = memref.alloc([[N]]) {{.*}}: memref<?x64x112x112xf32>
// CHECK:       "krnl.call"([[ALLOC]], {{.*}}) <{funcName = "om_gemmini_conv_f32_bias"
// CHECK:       return [[ALLOC]] : memref<?x64x112x112xf32>
