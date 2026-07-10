// RUN: onnx-mlir-opt --maccel=Gemmini --convert-onnx-to-krnl --canonicalize %s | FileCheck %s

// Verify that dynamic H and W (spatial dims 2, 3) in a Conv2D lower to
// om_gemmini_conv_f32_bias with memref.dim queries and ONNX output-size
// arithmetic: out_dim = (in_dim + 2*pad - kernel) / stride + 1.
//
// Kernel 3x3, stride 2, pad 1 → adj = 2*1 - 3 = -1
//   out_H = (in_H - 1) / 2 + 1
//   out_W = (in_W - 1) / 2 + 1

func.func @conv2d_f32_dynamic_spatial(%x: tensor<?x3x?x?xf32>,
    %w: tensor<16x3x3x3xf32>, %b: tensor<16xf32>) -> tensor<?x16x?x?xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
    group = 1 : si64,
    kernel_shape = [3, 3],
    pads = [1, 1, 1, 1],
    strides = [2, 2]
  } : (tensor<?x3x?x?xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>)
      -> tensor<?x16x?x?xf32>
  return %0 : tensor<?x16x?x?xf32>
}

// CHECK-LABEL: func.func @conv2d_f32_dynamic_spatial

// Runtime dim reads: batch (dim 0), input H (dim 2), input W (dim 3).
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG: [[N:%.+]]    = memref.dim %arg0, [[C0]] : memref<?x3x?x?xf32>
// CHECK-DAG: [[IH:%.+]]   = memref.dim %arg0, [[C2]] : memref<?x3x?x?xf32>
// CHECK-DAG: [[IW:%.+]]   = memref.dim %arg0, [[C3]] : memref<?x3x?x?xf32>

// Signed i64 arithmetic for output spatial sizes.
// CHECK-DAG: [[ADJ:%.+]] = arith.constant -1 : i64
// CHECK-DAG: [[STR:%.+]] = arith.constant 2 : i64
// CHECK-DAG: [[ONE:%.+]] = arith.constant 1 : i64
// CHECK-DAG: [[IH64:%.+]] = arith.index_cast [[IH]] : index to i64
// CHECK-DAG: [[IW64:%.+]] = arith.index_cast [[IW]] : index to i64
// CHECK-DAG: [[NH:%.+]] = arith.addi [[IH64]], [[ADJ]] : i64
// CHECK-DAG: [[DH:%.+]] = arith.divsi [[NH]], [[STR]] : i64
// CHECK-DAG: [[OH64:%.+]] = arith.addi [[DH]], [[ONE]] : i64
// CHECK-DAG: [[OH:%.+]] = arith.index_cast [[OH64]] : i64 to index
// CHECK-DAG: [[NW:%.+]] = arith.addi [[IW64]], [[ADJ]] : i64
// CHECK-DAG: [[DW:%.+]] = arith.divsi [[NW]], [[STR]] : i64
// CHECK-DAG: [[OW64:%.+]] = arith.addi [[DW]], [[ONE]] : i64
// CHECK-DAG: [[OW:%.+]] = arith.index_cast [[OW64]] : i64 to index

// Output is allocated with all three dynamic dims.
// CHECK: [[ALLOC:%.+]] = memref.alloc([[N]], [[OH]], [[OW]]) {{.*}}: memref<?x16x?x?xf32>

// Single runtime call; stride and pad are passed as i64 constants.
// CHECK: "krnl.call"([[ALLOC]], %arg0, %arg1, %arg2, {{%.+}}, {{%.+}}) <{funcName = "om_gemmini_conv_f32_bias"
// CHECK: return [[ALLOC]] : memref<?x16x?x?xf32>
