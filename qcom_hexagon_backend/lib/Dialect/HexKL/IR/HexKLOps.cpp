//===- HexKLOps.cpp - HexKL Ops  ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::hexkl;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void HexKLDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/HexKL/IR/HexKLOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/HexKL/IR/HexKLOps.cpp.inc"

// Requires 1D memref<i8> in VTCM (memory space 1).
static LogicalResult verifyHmxBlockMemref(Operation *op, Value v) {
  auto memrefType = dyn_cast<MemRefType>(v.getType());
  if (!memrefType)
    return op->emitOpError("expected hmx_block to be a memref");

  if (!memrefType.hasRank() || memrefType.getRank() != 1)
    return op->emitOpError("hmx_block must be a rank-1 memref");

  auto elem = dyn_cast<IntegerType>(memrefType.getElementType());
  if (!elem || elem.getWidth() != 8)
    return op->emitOpError("hmx_block element type must be i8");

  if (memrefType.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
    return op->emitOpError(
        "hmx_block must be in VTCM address space (memory space 1)");

  return success();
}

/// Verify that if a value is a constant integer, it must be non-negative.
static LogicalResult verifyNonNegativeIfConst(Operation *op, Value v,
                                              StringRef name) {
  if (auto c = mlir::getConstantIntValue(v))
    if (*c < 0)
      return op->emitOpError() << name << " must be non-negative";
  return success();
}

/// Helper function to verify non-negative operands.
/// Returns failure if verification fails.
template <typename... Args>
static LogicalResult verifyNonNegative(Operation *op, Value v, StringRef name,
                                       Args... args) {
  if (failed(verifyNonNegativeIfConst(op, v, name)))
    return failure();
  if constexpr (sizeof...(args) > 0) {
    return verifyNonNegative(op, args...);
  }
  return success();
}

LogicalResult MicroHMXSetupAccReadF16Op::verify() {
  return verifyHmxBlockMemref(getOperation(), getHmxBlock());
}

LogicalResult MicroHMXCopySubmatrixToF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getOutOffset(), "out_offset", getTileRow(),
                           "tile_row", getTileCol(), "tile_col", getInputRows(),
                           "input_rows", getInputCols(), "input_cols");
}

LogicalResult MicroHMXRmToAhF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getActivationOutOffset(),
                           "activation_out_offset", getFlatInOffset(),
                           "flat_in_offset");
}

LogicalResult MicroHMXRmToWhF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getWeightOffset(), "weight_offset", getTileRow(),
                           "tile_row", getTileCol(), "tile_col", getWtCols(),
                           "wt_cols");
}

LogicalResult MicroHMXMmF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getActivationOffset(), "activation_offset",
                           getWeightOffset(), "weight_offset");
}

LogicalResult MicroHMXAccReadF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getOutOffset(), "out_offset");
}

LogicalResult MicroHMXAhToRmF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getFlatOutOffset(), "flat_out_offset",
                           getActivationInOffset(), "activation_in_offset");
}

LogicalResult MicroHMXCopyF16ToF32SubmatrixOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getInOffset(), "in_offset", getTileRow(),
                           "tile_row", getTileCol(), "tile_col",
                           getOutputRows(), "output_rows", getOutputCols(),
                           "output_cols");
}

/// Verify a statically-known memref dimension is a multiple of `mult`.
/// No-op when the dimension is dynamic (the multiple is then a runtime
/// precondition the caller must uphold).
static LogicalResult verifyStaticDimMultiple(Operation *op, Value memref,
                                             unsigned dim, int64_t mult,
                                             StringRef name) {
  auto t = dyn_cast<MemRefType>(memref.getType());
  if (!t || !t.hasRank() || dim >= t.getRank())
    return success();
  int64_t d = t.getDimSize(dim);
  if (ShapedType::isDynamic(d))
    return success();
  if (d % mult != 0)
    return op->emitOpError()
           << name << " (" << d << ") must be a multiple of " << mult;
  return success();
}

LogicalResult HvxLayoutWeightOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  // src is [K][N]; the pair-tile vscatter requires K%32==0 and N%64==0.
  if (failed(verifyStaticDimMultiple(op, getSrc(), 0, 32, "K")) ||
      failed(verifyStaticDimMultiple(op, getSrc(), 1, 64, "N")))
    return failure();

  return verifyNonNegative(op, getWtOff(), "wt_off", getK(), "k", getN(), "n");
}

LogicalResult HvxLayoutWeightQ8Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  // q8_src is [N][row_stride]; the pair-tile vscatter requires N%64==0. K is
  // carried as a scalar operand (the q8x4x2 row interleaves quants+scales so
  // the contraction dim is not a static memref dim) and validated as K%32 at
  // the call site / lowering (k must be a multiple of 32).
  if (failed(verifyStaticDimMultiple(op, getQ8Src(), 0, 64, "N")))
    return failure();

  return verifyNonNegative(op, getWtOff(), "wt_off", getK(), "k", getN(), "n");
}

LogicalResult HvxLayoutActivationOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  // src is [M][K]; the pair-tile vscatter requires M%32==0 and K%64==0.
  if (failed(verifyStaticDimMultiple(op, getSrc(), 0, 32, "M")) ||
      failed(verifyStaticDimMultiple(op, getSrc(), 1, 64, "K")))
    return failure();

  return verifyNonNegative(op, getActOff(), "act_off", getM(), "m", getK(), "k");
}

LogicalResult HvxReadoutOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  // dst is [M][N]; the gather/widen requires M%32==0 and N%32==0.
  if (failed(verifyStaticDimMultiple(op, getDst(), 0, 32, "M")) ||
      failed(verifyStaticDimMultiple(op, getDst(), 1, 32, "N")))
    return failure();

  return verifyNonNegative(op, getOutOff(), "out_off", getScratchOff(),
                           "scratch_off", getM(), "m", getN(), "n");
}

LogicalResult HmxCoreDotOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getOutOff(), "out_off", getActOff(), "act_off",
                           getWtOff(), "wt_off", getBiasOff(), "bias_off",
                           getNRowTiles(), "n_row_tiles", getNColTiles(),
                           "n_col_tiles", getNDotTiles(), "n_dot_tiles");
}
