// SPDX-License-Identifier: Apache-2.0

#ifndef GEMMINI_LOW_OPS
#define GEMMINI_LOW_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GemminiLow_Dialect : Dialect {
  let name = "gemmini_low";
  let summary = "Gemmini low-level instruction dialect.";
  let cppNamespace = "::onnx_mlir::gemmini";
}

class GemminiLow_Op<string mnemonic, list<Trait> traits = []> :
    Op<GemminiLow_Dialect, mnemonic, traits>;

def GemminiLowConfigOp : GemminiLow_Op<"config",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Configure Gemmini execution dataflow and transpose bits";
  let description = [{
    Direct encoding of the CONFIG_EX RoCC instruction.
    `a_transpose` (rs1[8]) and `b_transpose` (rs1[9]) activate the
    AlwaysOutTransposer for the A and B matrix paths respectively.
    `acc_scale` (rs1[63:32]) is always emitted as ACC_SCALE_IDENTITY (1.0f).
  }];
  let arguments = (ins StrAttr:$dataflow,
                       OptionalAttr<BoolAttr>:$a_transpose,
                       OptionalAttr<BoolAttr>:$b_transpose);
  let assemblyFormat = "$dataflow attr-dict";
}

def GemminiLowMvinOp : GemminiLow_Op<"mvin",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Move a tile from main memory into Gemmini scratchpad";
  let arguments = (ins AnyMemRef:$source,
                       I64Attr:$spad_offset_rows,
                       I64Attr:$tile_rows,
                       I64Attr:$tile_cols);
  let assemblyFormat = "$source `to` $spad_offset_rows `rows` `tile` `(` $tile_rows `x` $tile_cols `)` attr-dict `:` type($source)";
}

def GemminiLowMvoutOp : GemminiLow_Op<"mvout",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Move a tile from Gemmini scratchpad or accumulator to main memory";
  let arguments = (ins AnyMemRef:$dest,
                       I64Attr:$spad_offset_rows,
                       I64Attr:$tile_rows,
                       I64Attr:$tile_cols);
  let assemblyFormat = "$dest `from` $spad_offset_rows `rows` `tile` `(` $tile_rows `x` $tile_cols `)` attr-dict `:` type($dest)";
}

def GemminiLowFenceOp : GemminiLow_Op<"fence",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Serialize Gemmini side effects before dependent memory access";
  let assemblyFormat = "attr-dict";
}

def GemminiLowMatmulOp : GemminiLow_Op<"matmul",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Issue a Gemmini matrix multiply against pre-assigned scratchpad rows";
  let arguments = (ins AnyMemRef:$lhs,
                       AnyMemRef:$rhs,
                       AnyMemRef:$out,
                       StrAttr:$mode,
                       StrAttr:$dataflow);
  let assemblyFormat = "$lhs `,` $rhs `into` $out `mode` $mode `dataflow` $dataflow attr-dict `:` type($lhs) `,` type($rhs) `,` type($out)";
}

#endif
