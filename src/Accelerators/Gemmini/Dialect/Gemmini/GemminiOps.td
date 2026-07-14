// SPDX-License-Identifier: Apache-2.0

// TableGen definitions for the high-level Gemmini dialect. These ops represent
// scheduled accelerator work before scratchpad addresses and RoCC instruction
// details are made explicit by later lowering passes.

#ifndef GEMMINI_OPS
#define GEMMINI_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Gemmini_Dialect : Dialect {
  let name = "gemmini";
  let summary = "Gemmini accelerator dialect.";
  let cppNamespace = "::onnx_mlir::gemmini";
}

class Gemmini_Op<string mnemonic, list<Trait> traits = []> :
    Op<Gemmini_Dialect, mnemonic, traits>;

def GemminiConfigOp : Gemmini_Op<"config",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Configure Gemmini execution dataflow and transpose bits";
  let description = [{
    Emits a CONFIG_EX instruction.  `a_transpose` sets rs1[8] to enable the
    hardware AlwaysOutTransposer on the A-matrix input; `b_transpose` sets
    rs1[9] for the B-matrix input.  Both default to false (cleared).
  }];
  let arguments = (ins StrAttr:$dataflow,
                       OptionalAttr<BoolAttr>:$a_transpose,
                       OptionalAttr<BoolAttr>:$b_transpose);
  let assemblyFormat = "$dataflow attr-dict";
}

def GemminiMvinOp : Gemmini_Op<"mvin",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Move a tile from main memory into Gemmini scratchpad";
  let arguments = (ins AnyMemRef:$source,
                       I64Attr:$spad_offset_rows,
                       I64Attr:$tile_rows,
                       I64Attr:$tile_cols);
  let assemblyFormat = "$source `to` $spad_offset_rows `rows` `tile` `(` $tile_rows `x` $tile_cols `)` attr-dict `:` type($source)";
}

def GemminiMvoutOp : Gemmini_Op<"mvout",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Move a tile from Gemmini scratchpad or accumulator to main memory";
  let arguments = (ins AnyMemRef:$dest,
                       I64Attr:$spad_offset_rows,
                       I64Attr:$tile_rows,
                       I64Attr:$tile_cols);
  let assemblyFormat = "$dest `from` $spad_offset_rows `rows` `tile` `(` $tile_rows `x` $tile_cols `)` attr-dict `:` type($dest)";
}

def GemminiFenceOp : Gemmini_Op<"fence",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Serialize Gemmini side effects before dependent memory access";
  let assemblyFormat = "attr-dict";
}

def GemminiMatmulOp : Gemmini_Op<"matmul",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Issue a Gemmini matrix multiply";
  let arguments = (ins AnyMemRef:$lhs,
                       AnyMemRef:$rhs,
                       AnyMemRef:$out,
                       StrAttr:$mode,
                       StrAttr:$dataflow);
  let assemblyFormat = "$lhs `,` $rhs `into` $out `mode` $mode `dataflow` $dataflow attr-dict `:` type($lhs) `,` type($rhs) `,` type($out)";
}

#endif
