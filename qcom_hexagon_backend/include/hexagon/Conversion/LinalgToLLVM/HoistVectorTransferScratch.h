// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_LINALGTOLLVM_HOIST_VECTOR_TRANSFER_SCRATCH_H
#define HEXAGON_LINALGTOLLVM_HOIST_VECTOR_TRANSFER_SCRATCH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::hexagon {

// This late stage runs after heap deallocation insertion. Do not use general
// buffer-loop-hoisting here: it can also move heap allocations independently
// of their existing frees. Only local vector stack scratch is eligible.
inline bool isLocalVectorScratch(Value allocation, Operation *scope) {
  SmallVector<Value> pending{allocation};
  llvm::SmallPtrSet<Operation *, 16> visitedAliases;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (!scope->isProperAncestor(owner))
        return false;
      if (auto load = dyn_cast<memref::LoadOp>(owner)) {
        if (load.getMemRef() == value) continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(owner)) {
        if (store.getMemRef() == value) continue;
      }
      if (isa<memref::DimOp, vector::LoadOp, vector::StoreOp,
              vector::TransferReadOp, vector::TransferWriteOp>(owner))
        continue;
      if (isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp,
              vector::TypeCastOp>(owner)) {
        if (visitedAliases.insert(owner).second)
          for (Value result : owner->getResults())
            pending.push_back(result);
        continue;
      }
      // In particular: no calls, returned/yielded aliases, deallocation, or
      // pointer extraction. All data escaping the loop must have been copied.
      return false;
    }
  }
  return true;
}

inline unsigned hoistVectorTransferScratch(func::FuncOp function) {
  SmallVector<memref::AllocaOp> candidates;
  function.walk([&](memref::AllocaOp allocation) {
    auto type = allocation.getType();
    auto vector = dyn_cast<VectorType>(type.getElementType());
    if (!type.hasStaticShape() || allocation->getNumOperands() != 0 ||
        !vector || vector.isScalable())
      return;
    if (Attribute space = type.getMemorySpace())
      if (!isa<IntegerAttr>(space) || cast<IntegerAttr>(space).getInt() != 0)
        return;
    // The vector-to-SCF pass places scratch directly in an automatic scope.
    // Keep parallel/async scopes and conditional placement unchanged.
    auto loop = dyn_cast<scf::ForOp>(allocation->getParentOp());
    if (!loop || !isLocalVectorScratch(allocation.getMemref(), loop))
      return;
    candidates.push_back(allocation);
  });
  for (memref::AllocaOp allocation : candidates) {
    Operation *loop = allocation->getParentOp();
    while (isa<scf::ForOp>(loop->getParentOp()))
      loop = loop->getParentOp();
    allocation->moveBefore(loop);
  }
  return candidates.size();
}

struct HoistVectorTransferScratchPass
    : PassWrapper<HoistVectorTransferScratchPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistVectorTransferScratchPass)
  StringRef getArgument() const final { return "hexagon-hoist-vector-scratch"; }
  void runOnOperation() final { hoistVectorTransferScratch(getOperation()); }
};

inline std::unique_ptr<Pass> createHoistVectorTransferScratchPass() {
  return std::make_unique<HoistVectorTransferScratchPass>();
}
} // namespace mlir::hexagon
#endif
