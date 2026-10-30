//===- TmTensorOps.td -----------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_OPS_TD
#define HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_OPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.td"
include "hexagon/Dialect/TmTensor/IR/TmTensorTypes.td"

// Base class for TmTensor dialect ops.
class TmTensor_Op<string mnemonic, list<Trait> traits> :
    Op<TmTensor_Dialect, mnemonic, traits> {

}

//===----------------------------------------------------------------------===//
// TmTensor Operations
//===----------------------------------------------------------------------===//

def TmTensor_AttentionOp : TmTensor_Op<"attention", []> {

  let summary = "Scaled Dot Product Attention operation";
  let description = [{
    Computes SDPA given Query, Key, Value, and Mask tensors.
    Equivalent to : softmax(QK^T / sqrt(d_k) + mask) * V
    Produces one output tensor with the attention result.
  }];

  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    AnyRankedTensor:$mask,
    AnyRankedTensor:$out
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `ins` `(`
       $query `,` $key `,`  $value `,` $mask
       `:` type($query) `,` type($key) `,` type($value) `,` type($mask)
        `)`
    `outs` `(` $out `:` type($out) `)`
    attr-dict `->` type($result)
  }];

  let hasVerifier = 1;
}

// Tensor scan emitted by torch-mlir for aten.cumsum. Keep the upstream custom
// syntax so ordinary exported programs can be read without source rewriting.
def TmTensor_YieldOp : TmTensor_Op<"yield", [Pure, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "$values attr-dict (`:` type($values)^)?";
}

def TmTensor_ScanOp : TmTensor_Op<"scan", [SingleBlock]> {
  let summary = "Inclusive or exclusive ordered tensor scan";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$output,
                       AnyRankedTensor:$accumulator,
                       I64Attr:$dimension, BoolAttr:$inclusive);
  let results = (outs AnyRankedTensor:$result, AnyRankedTensor:$accumulator_result);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    `dimension` `(` $dimension `)` `inclusive` `(` $inclusive `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `,` $accumulator `:` type($output) `,` type($accumulator) `)`
    $region attr-dict `->` type($result) `,` type($accumulator_result)
  }];
  let hasVerifier = 1;
}

// Ordinary torch-mlir indexed updates, including functionalized index_copy_.
def TmTensor_ScatterOp : TmTensor_Op<"scatter", [SingleBlock]> {
  let summary = "Ordered indexed updates of a tensor";
  let arguments = (ins AnyRankedTensor:$updates, AnyRankedTensor:$indices,
                       AnyRankedTensor:$original,
                       DenseI64ArrayAttr:$dimension_map, BoolAttr:$unique_indices);
  let results = (outs AnyRankedTensor:$result);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    attr-dict `unique_indices` `(` $unique_indices `)`
    `ins` `(` $updates `,` $indices `:` type($updates) `,` type($indices) `)`
    `outs` `(` $original `:` type($original) `)`
    $region `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_OPS_TD
