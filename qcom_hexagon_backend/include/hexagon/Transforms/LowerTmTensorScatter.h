// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_TRANSFORMS_LOWER_TMTENSOR_SCATTER_H
#define HEXAGON_TRANSFORMS_LOWER_TMTENSOR_SCATTER_H
#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include <functional>

namespace mlir::hexagon {
// Preserve scalar region order and duplicate-index dependencies. This is a
// correctness lowering, not an assertion that all indexed writes are parallel.
// As in upstream scatter, the caller must provide in-bounds indices.
struct LowerScatterOp : OpRewritePattern<tm_tensor::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tm_tensor::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto updatesType = cast<RankedTensorType>(op.getUpdates().getType());
    auto originalType = cast<RankedTensorType>(op.getOriginal().getType());
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> bounds, indices;
    for (int64_t dim = 0; dim < updatesType.getRank(); ++dim)
      bounds.push_back(tensor::DimOp::create(rewriter, loc, op.getUpdates(), dim));
    std::function<Value(OpBuilder &, unsigned, Value)> nest;
    nest = [&](OpBuilder &builder, unsigned dim, Value carried) -> Value {
      if (dim < bounds.size()) {
        auto loop = scf::ForOp::create(
            builder, loc, zero, bounds[dim], one, ValueRange{carried},
            [&](OpBuilder &body, Location bodyLoc, Value iv, ValueRange values) {
              indices.push_back(iv);
              Value result = nest(body, dim + 1, values[0]);
              indices.pop_back();
              scf::YieldOp::create(body, bodyLoc, result);
            });
        return loop.getResult(0);
      }
      SmallVector<Value> destination(originalType.getRank());
      int64_t offset = originalType.getRank() - updatesType.getRank() + 1;
      for (unsigned i = 1; i < indices.size(); ++i)
        destination[offset + i - 1] = indices[i];
      for (auto entry : llvm::enumerate(op.getDimensionMap())) {
        Value column = arith::ConstantIndexOp::create(builder, loc, entry.index());
        Value coordinate = tensor::ExtractOp::create(
            builder, loc, op.getIndices(), ValueRange{indices[0], column});
        coordinate = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), coordinate);
        int64_t dimension = entry.value();
        if (destination[dimension])
          coordinate = arith::AddIOp::create(builder, loc, coordinate, destination[dimension]);
        destination[dimension] = coordinate;
      }
      Value update = tensor::ExtractOp::create(builder, loc, op.getUpdates(), indices);
      Value previous = tensor::ExtractOp::create(builder, loc, carried, destination);
      Block &region = op.getRegion().front();
      IRMapping mapping;
      mapping.map(region.getArgument(0), update);
      mapping.map(region.getArgument(1), previous);
      for (Operation &operation : region.without_terminator())
        builder.clone(operation, mapping);
      Value value = mapping.lookupOrDefault(region.getTerminator()->getOperand(0));
      return tensor::InsertOp::create(builder, loc, value, carried, destination);
    };
    rewriter.replaceOp(op, nest(rewriter, 0, op.getOriginal()));
    return success();
  }
};
} // namespace mlir::hexagon
#endif
