// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_FOLD_CONSTANT_TRANSPOSES_H
#define HEXAGON_FOLD_CONSTANT_TRANSPOSES_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace mlir::hexagon {
// Pure byte permutation: no float evaluation or quantization. Torch FX stores
// weights as resources; folding only DenseElementsAttr misses those constants.
inline bool foldConstantTranspose(linalg::TransposeOp op, IRRewriter &rewriter) {
  if (!op.hasPureTensorSemantics() || op->getNumResults() != 1 ||
      op.getPermutation() != ArrayRef<int64_t>({1, 0})) return false;
  auto in = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!in || !out || in.getRank() != 2 || !in.hasStaticShape() ||
      !out.hasStaticShape() || in.getDimSize(0) < 1 || in.getDimSize(1) < 1 ||
      out.getShape() != ArrayRef<int64_t>({in.getDimSize(1), in.getDimSize(0)}))
    return false;
  Type element = in.getElementType();
  if (element != out.getElementType() || !element.isIntOrFloat()) return false;
  unsigned bits = element.getIntOrFloatBitWidth();
  if (bits != 8 && bits != 16 && bits != 32 && bits != 64) return false;
  // Named Linalg regions can be customized. Check their exact copy semantics.
  Block &body = op->getRegion(0).front();
  if (!llvm::hasSingleElement(body)) return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != body.getArgument(0)) return false;
  auto constant = op.getInput().getDefiningOp<arith::ConstantOp>();
  if (!constant) return false;
  ArrayRef<char> raw;
  if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue())) {
    if (dense.isSplat()) {
      rewriter.setInsertionPoint(op);
      auto value = rewriter.create<arith::ConstantOp>(op.getLoc(), dense.reshape(out));
      rewriter.replaceOp(op, value.getResult());
      return true;
    }
    raw = dense.getRawData();
  } else if (auto resource = dyn_cast<DenseResourceElementsAttr>(constant.getValue())) {
    auto *blob = resource.getRawHandle().getBlob();
    if (!blob) return false;
    raw = blob->getData();
  } else return false;
  uint64_t rows = in.getDimSize(0), cols = in.getDimSize(1), width = bits / 8;
  if (rows > std::numeric_limits<size_t>::max() / cols ||
      rows * cols > std::numeric_limits<size_t>::max() / width) return false;
  size_t bytes = rows * cols * width;
  if (raw.size() != bytes) return false; // No guessing about external storage.
  std::vector<char> transposed(bytes);
  for (uint64_t rb = 0; rb < rows; rb += 32)
    for (uint64_t cb = 0; cb < cols; cb += 32)
      for (uint64_t r = rb; r < std::min(rows, rb + 32); ++r)
        for (uint64_t c = cb; c < std::min(cols, cb + 32); ++c)
          std::memcpy(transposed.data() + (c * rows + r) * width,
                      raw.data() + (r * cols + c) * width, width);
  auto value = DenseElementsAttr::getFromRawBuffer(out, transposed);
  rewriter.setInsertionPoint(op);
  auto folded = rewriter.create<arith::ConstantOp>(op.getLoc(), value);
  rewriter.replaceOp(op, folded.getResult());
  return true;
}
class FoldConstantTransposesPass
    : public PassWrapper<FoldConstantTransposesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConstantTransposesPass)
  StringRef getArgument() const final { return "hexagon-fold-constant-transposes"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    SmallVector<linalg::TransposeOp> work;
    getOperation().walk([&](linalg::TransposeOp op) { work.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (auto op : work) foldConstantTranspose(op, rewriter);
  }
};
} // namespace mlir::hexagon
#endif
