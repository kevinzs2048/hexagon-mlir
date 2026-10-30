//===- TmTensorOps.cpp - TmTensor dialect operations implementation ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::tm_tensor;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void TmTensorDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/TmTensor/IR/TmTensorOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult AttentionOp::verify() {
  auto queryType = llvm::dyn_cast<RankedTensorType>(getQuery().getType());
  auto keyType = llvm::dyn_cast<RankedTensorType>(getKey().getType());
  auto valueType = llvm::dyn_cast<RankedTensorType>(getValue().getType());
  auto maskType = llvm::dyn_cast<RankedTensorType>(getMask().getType());

  if (!queryType || !keyType || !valueType || !maskType) {
    return emitOpError("query, key, value and mask must be ranked tensors");
  }

  // All tensors must be 3D and have same element type
  if (queryType.getRank() != 3 || keyType.getRank() != 3 ||
      valueType.getRank() != 3 || maskType.getRank() != 3)
    return emitOpError("all tensors must be 3D");

  auto elType = queryType.getElementType();
  if (keyType.getElementType() != elType ||
      valueType.getElementType() != elType ||
      maskType.getElementType() != elType)
    return emitOpError("all tensors must have same element type");

  // Shape constraints:
  //  Q[batch,seq_q,head_dim]
  //  K[batch,seq_kv,head_dim]
  //  V[batch,seq_kv,head_dim]
  //  M[batch,seq_q,seq_kv]
  auto queryShape = queryType.getShape();
  auto keyShape = keyType.getShape();
  auto valueShape = valueType.getShape();
  auto maskShape = maskType.getShape();

  // independent dims
  int64_t batch = queryShape[0];
  int64_t seq_q = queryShape[1];
  int64_t head_dim = queryShape[2];
  int64_t seq_kv = keyShape[1];

  if (ShapedType::isDynamic(batch) || ShapedType::isDynamic(seq_q) ||
      ShapedType::isDynamic(head_dim) || ShapedType::isDynamic(seq_kv))
    return emitOpError("dynamic dimensions not supported");

  // dependendent dims
  if (keyShape[0] != batch || keyShape[2] != head_dim ||
      valueShape[0] != batch || valueShape[1] != seq_kv ||
      valueShape[2] != head_dim || maskShape[0] != batch ||
      maskShape[1] != seq_q || maskShape[2] != seq_kv)
    return emitOpError("tensor shapes don't satisfy attention constraints");

  return success();
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

LogicalResult ScanOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto outputType = cast<RankedTensorType>(getOutput().getType());
  auto accumulatorType = cast<RankedTensorType>(getAccumulator().getType());
  if (getDimension() < 0 || getDimension() >= inputType.getRank())
    return emitOpError("scan dimension must be within the input rank");
  if (inputType != outputType || getResult().getType() != outputType ||
      getAccumulatorResult().getType() != accumulatorType)
    return emitOpError("scan input/output/result types must match");
  SmallVector<int64_t> accumulatorShape(inputType.getShape());
  accumulatorShape.erase(accumulatorShape.begin() + getDimension());
  if (accumulatorType.getShape() != ArrayRef<int64_t>(accumulatorShape) ||
      accumulatorType.getElementType() != inputType.getElementType())
    return emitOpError("accumulator must remove the scan dimension from input");
  Block &body = getRegion().front();
  if (body.getNumArguments() != 2 ||
      body.getArgument(0).getType() != inputType.getElementType() ||
      body.getArgument(1).getType() != inputType.getElementType())
    return emitOpError("scan region requires two input-element arguments");
  if (body.empty())
    return emitOpError("scan region must yield one input-element value");
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0).getType() != inputType.getElementType())
    return emitOpError("scan region must yield one input-element value");
  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  auto updates = cast<RankedTensorType>(getUpdates().getType());
  auto indices = cast<RankedTensorType>(getIndices().getType());
  auto original = cast<RankedTensorType>(getOriginal().getType());
  if (indices.getRank() != 2 || !isa<IntegerType>(indices.getElementType()) ||
      indices.isDynamicDim(1))
    return emitOpError("indices must be rank-two integers with static index depth");
  if (updates.getRank() < 1 || original.getRank() < 1 ||
      updates.getRank() - 1 > original.getRank())
    return emitOpError("invalid updates or original rank");
  if (!indices.isDynamicDim(0) && !updates.isDynamicDim(0) &&
      indices.getDimSize(0) != updates.getDimSize(0))
    return emitOpError("indices and updates must have the same number of rows");
  if (getResult().getType() != original ||
      updates.getElementType() != original.getElementType() ||
      !original.getElementType().isIntOrFloat())
    return emitOpError("original/result and scalar update types must match");
  auto dimensions = getDimensionMap();
  if (static_cast<int64_t>(dimensions.size()) != indices.getDimSize(1))
    return emitOpError("dimension map must match index depth");
  SmallVector<bool> mapped(original.getRank(), false);
  for (int64_t dim : dimensions) {
    if (dim < 0 || dim >= original.getRank() || mapped[dim])
      return emitOpError("dimension map must contain distinct valid dimensions");
    mapped[dim] = true;
  }
  int64_t trailingStart = original.getRank() - updates.getRank() + 1;
  for (int64_t dim = 0; dim < trailingStart; ++dim)
    if (!mapped[dim])
      return emitOpError("indices and trailing update dimensions must cover the original rank");
  for (int64_t dim = trailingStart; dim < original.getRank(); ++dim) {
    int64_t updateDim = dim - trailingStart + 1;
    if (!original.isDynamicDim(dim) && !updates.isDynamicDim(updateDim) &&
        updates.getDimSize(updateDim) > original.getDimSize(dim))
      return emitOpError("update slice exceeds the original tensor shape");
  }
  Block &body = getRegion().front();
  if (body.getNumArguments() != 2 ||
      body.getArgument(0).getType() != original.getElementType() ||
      body.getArgument(1).getType() != original.getElementType())
    return emitOpError("scatter region requires update and original scalar arguments");
  auto yield = body.empty() ? YieldOp() : dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0).getType() != original.getElementType())
    return emitOpError("scatter region must yield one scalar of the original element type");
  return success();
}

// ODS-Generated Definitions
#define GET_OP_CLASSES
#include "hexagon/Dialect/TmTensor/IR/TmTensorOps.cpp.inc"
