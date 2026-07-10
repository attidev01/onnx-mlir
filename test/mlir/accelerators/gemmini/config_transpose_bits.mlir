// RUN: onnx-mlir-opt --maccel=Gemmini --convert-gemmini-low-to-llvm %s | FileCheck %s

// Verify that a_transpose (bit 8) and b_transpose (bit 9) are packed into
// the config_ex rs1 constant alongside sys_acc_shift=1.0f (bits 63:32),
// a_stride=1 (bits 31:16), and dataflow=WS (bit 2).
// rs2 = c_stride=1 in bits[63:48] = 0x0001000000000000 = 281474976710656.

// CHECK-LABEL: func.func @config_no_transpose
func.func @config_no_transpose() {
  // a_stride=1 → rs1 = 0x3F80000000010004 = 4575657221408489476
  gemmini_low.config "ws"
  return
}
// CHECK: llvm.mlir.constant(4575657221408489476 : i64)

// CHECK-LABEL: func.func @config_a_transpose
func.func @config_a_transpose() {
  // a_transpose=true sets bit 8 → rs1 = 0x3F80000000010104 = 4575657221408489732
  gemmini_low.config "ws" {a_transpose = true}
  return
}
// CHECK: llvm.mlir.constant(4575657221408489732 : i64)

// CHECK-LABEL: func.func @config_b_transpose
func.func @config_b_transpose() {
  // b_transpose=true sets bit 9 → rs1 = 0x3F80000000010204 = 4575657221408489988
  gemmini_low.config "ws" {b_transpose = true}
  return
}
// CHECK: llvm.mlir.constant(4575657221408489988 : i64)

// CHECK-LABEL: func.func @config_both_transpose
func.func @config_both_transpose() {
  // bits 8+9 → rs1 = 0x3F80000000010304 = 4575657221408490244
  gemmini_low.config "ws" {a_transpose = true, b_transpose = true}
  return
}
// CHECK: llvm.mlir.constant(4575657221408490244 : i64)
