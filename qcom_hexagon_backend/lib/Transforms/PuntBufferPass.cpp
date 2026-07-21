//===- PuntBufferPass.cpp - implementation of punt-buffer pass ---------======//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements removing of copying of kernel input arg buffer content
// to locally created buffers. Instead forwards it to user where valid.
//
//===----------------------------------------------------------------------===//
//
// e.g.
// ```
//   func.func @foo(%arg0: memref<*xf16>, ...) {
//      ...
//      %reinterpret_cast = memref.reinterpret_cast %arg0 to ...]
//                          : memref<*xElType> to memref<ShapeTyxElType>
//      %alloc = memref.alloc() : memref<ShapeTyxElType>
//      memref.copy %reinterpret_cast, %alloc : memref<...> to memref<...>
//      %... = bufferization.to_tensor %alloc restrict writable : memref<...>
//      ...
// ```
// is replaced by
// ```
//      %reinterpret_cast = memref.reinterpret_cast %arg0 to ...]
//                          : memref<*xElType> to memref<ShapeTyxElType>
//      %... = bufferization.to_tensor %reinterpret_cast restrict : memref<...>
// ```
//
// Things can quickly get complicated when interacting with memref views and
// loops. e.g.
// ```
//   func.func @foo(%arg0: memref<*xf16>, ...) {
//      ...
//      %reinterpret_cast = memref.reinterpret_cast %arg0 to ...]
//                          : memref<*xf16> to memref<ShapeTyxElType>
//      ... = scf.for %i = %start to %stop step %step
//                iter_args(..., %travelling_memref = %reinterpret_cast)
//                      -> (..., memref<...>)  : i32 {
//            ...
//            %alloc = memref.alloc() : memref<...>
//            memref.copy %travelling_memref, %alloc :  memref<...> to
//            memref<...>
//            ... =  bufferization.to_tensor %alloc restrict writable :
//            memref<...>
//            ...
//            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to ...
//           scf.yield ..., %reinterpret_cast_0 : ...
//        }
// ```
// We are able to handle such complicated cases using buffer alias analysis.

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "hexagon-punt-buffer"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONPUNTBUFFER
#include "hexagon/Transforms/Passes.h.inc"

namespace {

using PuntCandidates =
    std::vector<std::tuple<memref::CopyOp, bufferization::ToTensorOp, Value>>;

struct HexagonPuntBufferPass
    : public ::impl::HexagonPuntBufferBase<HexagonPuntBufferPass> {

  void runOnOperation() override;
};

// Use alias-analysis framework to work out if a particular memref value is
// a view of the kernel input memref arg. The check below is not most efficient
// but when reverse resolve works this can be reworked to make it efficient.
static bool isaViewOfFuncInputBuffer(Value value, FunctionOpInterface funcOp) {
  BufferViewFlowAnalysis analysis(funcOp);
  for (const auto &arg : funcOp.getArguments()) {
    if (isa<BaseMemRefType>(arg.getType())) {
      auto viewSet = analysis.resolve(arg);
      for (auto view : viewSet) {
        if (value == view)
          return true;
      }
    }
  }
  return false;
}

// Populate punting candidates iff copy involves `source` that is a view
// of kernel input arg (checked separately), and `target` is an alloc
// with no further use.
bool populatePuntCandidate(memref::CopyOp copyOp, PuntCandidates &candidates) {
  Value target = copyOp.getTarget();
  auto targetSubview = target.getDefiningOp<memref::SubViewOp>();
  Value allocValue = targetSubview ? targetSubview.getSource() : target;

  // Check we are copying to an allocation, directly or through a subview.
  auto alloc = allocValue.getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return false;

  bufferization::ToTensorOp toTensor;
  for (OpOperand &use : allocValue.getUses())
    if ((toTensor = llvm::dyn_cast<bufferization::ToTensorOp>(use.getOwner())))
      break;
  if (!toTensor)
    return false;

  if (!targetSubview) {
    auto uses = copyOp.getTarget().getUses();
    if (std::distance(uses.begin(), uses.end()) != 2)
      return false;

    auto firstUse = uses.begin()->getOwner();
    auto secondUse = std::next(uses.begin())->getOwner();
    if (!firstUse || !secondUse || (firstUse != copyOp && secondUse != copyOp))
      return false;

    auto otherUse = firstUse == copyOp ? secondUse : firstUse;
    toTensor = llvm::dyn_cast<bufferization::ToTensorOp>(otherUse);
    if (!toTensor)
      return false;
  } else {
    auto sourceSubview = copyOp.getSource().getDefiningOp<memref::SubViewOp>();
    if (!sourceSubview)
      return false;
    if (sourceSubview.getMixedOffsets() != targetSubview.getMixedOffsets() ||
        sourceSubview.getMixedSizes() != targetSubview.getMixedSizes() ||
        sourceSubview.getMixedStrides() != targetSubview.getMixedStrides())
      return false;

    for (OpOperand &use : allocValue.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == targetSubview || owner == toTensor ||
          isa<linalg::FillOp>(owner))
        continue;
      return false;
    }
    if (!llvm::hasSingleElement(targetSubview->getUses()))
      return false;
  }

  Value replacement = copyOp.getSource();
  candidates.emplace_back(copyOp, toTensor, replacement);
  DBG("___target is a valid single use alloc");
  return true;
}

Value cloneMemrefViewAt(IRRewriter &rewriter, Value value,
                        Operation *insertionPoint) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->isBeforeInBlock(insertionPoint))
    return value;

  rewriter.setInsertionPoint(insertionPoint);
  if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
    return rewriter
        .create<memref::ReinterpretCastOp>(
            castOp.getLoc(), castOp.getType(), castOp.getSource(),
            castOp.getOffsets(), castOp.getSizes(), castOp.getStrides(),
            castOp.getStaticOffsets(), castOp.getStaticSizes(),
            castOp.getStaticStrides())
        .getResult();
  }

  if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
    Value source =
        cloneMemrefViewAt(rewriter, subviewOp.getSource(), insertionPoint);
    if (!source)
      return nullptr;

    return rewriter
        .create<memref::SubViewOp>(
            subviewOp.getLoc(), cast<MemRefType>(subviewOp.getType()), source,
            subviewOp.getMixedOffsets(), subviewOp.getMixedSizes(),
            subviewOp.getMixedStrides())
        .getResult();
  }

  return nullptr;
}

bool genericChainDependsOn(Value value, Value root,
                           llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (value == root)
    return true;
  auto genericOp = value.getDefiningOp<linalg::GenericOp>();
  if (!genericOp || genericOp->getNumResults() != 1)
    return false;
  if (!visited.insert(genericOp).second)
    return false;
  for (OpOperand *input : genericOp.getDpsInputOperands())
    if (genericChainDependsOn(input->get(), root, visited))
      return true;
  return false;
}

Value cloneGenericChainToDynamicTensor(IRRewriter &rewriter, Value value,
                                       Value oldRoot, Value newRoot,
                                       Value outputTensor,
                                       Operation *insertionPoint,
                                       IRMapping &mapping) {
  if (value == oldRoot)
    return newRoot;
  if (mapping.contains(value))
    return mapping.lookup(value);
  llvm::SmallPtrSet<Operation *, 8> visited;
  if (!genericChainDependsOn(value, oldRoot, visited) &&
      value.getType() != oldRoot.getType())
    return value;

  auto genericOp = value.getDefiningOp<linalg::GenericOp>();
  if (!genericOp || genericOp->getNumResults() != 1) {
    auto fillOp = value.getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return value;
    if (value.getType() == newRoot.getType())
      return value;
    rewriter.setInsertionPoint(insertionPoint);
    auto newOp = rewriter.create<linalg::FillOp>(
        fillOp.getLoc(), fillOp.getInputs(), ValueRange{outputTensor});
    mapping.map(value, newOp->getResult(0));
    return newOp->getResult(0);
  }

  SmallVector<Value> newInputs;
  for (OpOperand *input : genericOp.getDpsInputOperands()) {
    Value newInput = cloneGenericChainToDynamicTensor(
        rewriter, input->get(), oldRoot, newRoot, outputTensor, insertionPoint,
        mapping);
    if (!newInput)
      return nullptr;
    newInputs.push_back(newInput);
  }

  SmallVector<Value> newOutputs;
  for (OpOperand &output : genericOp.getDpsInitsMutable()) {
    Value oldOutput = output.get();
    Value newOutput = cloneGenericChainToDynamicTensor(
        rewriter, oldOutput, oldRoot, newRoot, outputTensor, insertionPoint,
        mapping);
    if (!newOutput)
      return nullptr;
    newOutputs.push_back(oldOutput == oldRoot ||
                                 newOutput.getType() != outputTensor.getType()
                             ? outputTensor
                             : newOutput);
  }

  SmallVector<Type> resultTypes;
  for (Value output : newOutputs)
    resultTypes.push_back(output.getType());

  rewriter.setInsertionPoint(insertionPoint);
  auto newOp = rewriter.create<linalg::GenericOp>(
      genericOp.getLoc(), resultTypes, newInputs, newOutputs,
      genericOp.getIndexingMapsArray(), genericOp.getIteratorTypesArray(),
      /*bodyBuild=*/nullptr);
  rewriter.cloneRegionBefore(genericOp->getRegion(0), newOp.getRegion(),
                             newOp.getRegion().begin());
  mapping.map(value, newOp->getResult(0));
  return newOp->getResult(0);
}

bool rewriteMaterializeChainFromRoot(IRRewriter &rewriter, Value oldRoot,
                                     Value newRoot) {
  auto funcOp = oldRoot.getParentBlock()->getParentOp();
  bufferization::MaterializeInDestinationOp materialize;
  tensor::ExtractSliceOp resultSlice;
  linalg::GenericOp directMaterializeGeneric;

  funcOp->walk([&](bufferization::MaterializeInDestinationOp candidate) {
    auto slice = candidate.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    Value source = slice ? Value(slice.getSource()) : candidate.getSource();

    llvm::SmallPtrSet<Operation *, 8> visited;
    if (!slice) {
      if (auto genericOp = source.getDefiningOp<linalg::GenericOp>()) {
        if (llvm::is_contained(genericOp.getInputs(), oldRoot)) {
          directMaterializeGeneric = genericOp;
          materialize = candidate;
          return WalkResult::interrupt();
        }
      }
    }

    if (genericChainDependsOn(source, oldRoot, visited)) {
      resultSlice = slice;
      materialize = candidate;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!materialize)
    return false;

  auto destType = dyn_cast<MemRefType>(materialize.getDest().getType());
  auto tensorType = dyn_cast<RankedTensorType>(materialize.getSource().getType());
  if (!destType || !tensorType)
    return false;

  auto newTensorType = RankedTensorType::get(
      destType.getShape(), tensorType.getElementType(), tensorType.getEncoding());
  Operation *insertionPoint =
      resultSlice ? resultSlice.getOperation() : materialize.getOperation();
  Value dest = cloneMemrefViewAt(rewriter, materialize.getDest(), insertionPoint);
  if (!dest)
    return false;
  materialize.getDestMutable().assign(dest);
  materialize.setRestrict(true);

  rewriter.setInsertionPoint(insertionPoint);
  Value outputTensor = rewriter.create<bufferization::ToTensorOp>(
      materialize.getLoc(), newTensorType, dest, true /* restrict */,
      true /* writable */);

  if (directMaterializeGeneric) {
    SmallVector<Value> newInputs;
    for (OpOperand *input : directMaterializeGeneric.getDpsInputOperands()) {
      Value value = input->get();
      newInputs.push_back(value == oldRoot ? newRoot : value);
    }
    SmallVector<Value> newOutputs;
    for (OpOperand &output : directMaterializeGeneric.getDpsInitsMutable()) {
      Value value = output.get();
      newOutputs.push_back(value == oldRoot ? outputTensor : value);
    }
    SmallVector<Type> resultTypes;
    for (Value output : newOutputs)
      resultTypes.push_back(output.getType());
    rewriter.setInsertionPoint(insertionPoint);
    auto newOp = rewriter.create<linalg::GenericOp>(
        directMaterializeGeneric.getLoc(), resultTypes, newInputs, newOutputs,
        directMaterializeGeneric.getIndexingMapsArray(),
        directMaterializeGeneric.getIteratorTypesArray(),
        /*bodyBuild=*/nullptr);
    rewriter.cloneRegionBefore(directMaterializeGeneric->getRegion(0),
                               newOp.getRegion(), newOp.getRegion().begin());
    materialize->setOperand(0, newOp->getResult(0));
    if (directMaterializeGeneric->use_empty())
      rewriter.eraseOp(directMaterializeGeneric);
    return true;
  }

  IRMapping mapping;
  Value source =
      resultSlice ? Value(resultSlice.getSource()) : materialize.getSource();
  Value newSource = cloneGenericChainToDynamicTensor(
      rewriter, source, oldRoot, newRoot, outputTensor, insertionPoint,
      mapping);
  if (!newSource || newSource.getType() != materialize.getSource().getType())
    return false;

  if (newSource == newRoot && source == oldRoot) {
    auto sourceTensor = newRoot.getDefiningOp<bufferization::ToTensorOp>();
    if (!sourceTensor)
      return false;

    unsigned rank = destType.getRank();
    SmallVector<AffineMap> maps{
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    rewriter.setInsertionPoint(insertionPoint);
    rewriter.create<linalg::GenericOp>(
        materialize.getLoc(), TypeRange{}, ValueRange{sourceTensor.getBuffer()},
        ValueRange{dest}, maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          builder.create<linalg::YieldOp>(loc, args[0]);
        });
    rewriter.eraseOp(materialize);
    if (resultSlice && resultSlice.use_empty())
      rewriter.eraseOp(resultSlice);
  } else {
    materialize->setOperand(0, newSource);
    if (resultSlice && resultSlice.use_empty())
      rewriter.eraseOp(resultSlice);
  }

  SmallVector<Value> deadValues;
  llvm::SmallDenseSet<Value, 8> seenDeadValues;
  auto pushDeadValue = [&](Value value) {
    if (value && seenDeadValues.insert(value).second)
      deadValues.push_back(value);
  };
  pushDeadValue(source);
  for (unsigned i = 0; i < deadValues.size(); ++i) {
    auto genericOp = deadValues[i].getDefiningOp<linalg::GenericOp>();
    if (!genericOp || !genericOp->use_empty())
      continue;
    for (OpOperand *input : genericOp.getDpsInputOperands())
      pushDeadValue(input->get());
    for (OpOperand &output : genericOp.getDpsInitsMutable())
      pushDeadValue(output.get());
    rewriter.eraseOp(genericOp);
  }
  return true;
}

void eraseDeadAllocUsers(IRRewriter &rewriter, Value allocValue) {
  Operation *allocOp = allocValue.getDefiningOp();
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> users(allocValue.getUsers().begin(),
                                   allocValue.getUsers().end());
    for (Operation *op : users) {
      if (!op || op == allocOp)
        continue;

      if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
        if (!subviewOp.use_empty())
          continue;
        rewriter.eraseOp(subviewOp);
        changed = true;
        continue;
      }

      if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
        if (!fillOp->use_empty())
          continue;
        rewriter.eraseOp(fillOp);
        changed = true;
        continue;
      }
    }

    SmallVector<scf::IfOp> emptyIfs;
    allocOp->getParentOfType<func::FuncOp>().walk([&](scf::IfOp ifOp) {
      if (isOpTriviallyDead(ifOp))
        emptyIfs.push_back(ifOp);
    });
    for (scf::IfOp ifOp : emptyIfs) {
      rewriter.eraseOp(ifOp);
      changed = true;
    }
  }

  if (allocOp && allocOp->use_empty() && isOpTriviallyDead(allocOp))
    rewriter.eraseOp(allocOp);
}

linalg::GenericOp rebuildGenericWithReplacement(IRRewriter &rewriter,
                                                linalg::GenericOp op,
                                                Value oldValue,
                                                Value newValue) {
  tensor::ExtractSliceOp resultSlice;
  bufferization::MaterializeInDestinationOp materialize;
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!op->hasOneUse())
    return nullptr;
  if (op->getResults().size() == 1 && funcOp) {
    funcOp.walk([&](bufferization::MaterializeInDestinationOp candidate) {
      auto slice =
          candidate.getSource().getDefiningOp<tensor::ExtractSliceOp>();
      if (!slice || slice.getSource() != op->getResult(0))
        return WalkResult::advance();
      resultSlice = slice;
      materialize = candidate;
      return WalkResult::interrupt();
    });
  }

  Value outputTensor;
  auto destType =
      materialize ? dyn_cast<MemRefType>(materialize.getDest().getType())
                  : MemRefType();
  auto tensorType = dyn_cast<RankedTensorType>(newValue.getType());
  if (materialize && destType && tensorType) {
    auto newTensorType = RankedTensorType::get(
        destType.getShape(), tensorType.getElementType(),
        tensorType.getEncoding());
    Value dest = cloneMemrefViewAt(rewriter, materialize.getDest(), op);
    if (!dest)
      return nullptr;
    materialize.getDestMutable().assign(dest);
    materialize.setRestrict(true);

    rewriter.setInsertionPoint(op);
    outputTensor = rewriter.create<bufferization::ToTensorOp>(
        materialize.getLoc(), newTensorType, dest, true /* restrict */,
        true /* writable */);
  }

  SmallVector<Value> newInputs;
  SmallVector<Value> newOutputs;
  for (OpOperand *input : op.getDpsInputOperands()) {
    Value value = input->get();
    newInputs.push_back(value == oldValue ? newValue : value);
  }
  for (OpOperand &output : op.getDpsInitsMutable()) {
    Value value = output.get();
    if (value == oldValue)
      newOutputs.push_back(outputTensor ? outputTensor : newValue);
    else
      newOutputs.push_back(value);
  }

  SmallVector<Type> resultTypes;
  for (Value output : newOutputs)
    resultTypes.push_back(output.getType());

  rewriter.setInsertionPoint(op);
  auto newOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(), resultTypes, newInputs, newOutputs,
      op.getIndexingMapsArray(), op.getIteratorTypesArray(),
      /*bodyBuild=*/nullptr);
  rewriter.cloneRegionBefore(op->getRegion(0), newOp.getRegion(),
                             newOp.getRegion().begin());

  if (resultSlice && materialize && newOp->getResults().size() == 1 &&
      resultSlice.getResult().getType() == newOp->getResult(0).getType()) {
    materialize->setOperand(0, newOp->getResult(0));
    if (resultSlice.use_empty())
      rewriter.eraseOp(resultSlice);
  }
  if (op->use_empty())
    rewriter.eraseOp(op);
  return newOp;
}

bool rebuildGenericUsers(IRRewriter &rewriter, Value oldValue, Value newValue) {
  SmallVector<linalg::GenericOp> genericUsers;
  llvm::SmallPtrSet<Operation *, 4> seen;
  for (Operation *user : llvm::make_early_inc_range(oldValue.getUsers())) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
      if (!seen.insert(user).second)
        continue;
      genericUsers.push_back(genericOp);
    }
  }

  for (linalg::GenericOp genericOp : genericUsers) {
    if (!rebuildGenericWithReplacement(rewriter, genericOp, oldValue, newValue))
      return false;
  }
  return true;
}

bool rebuildExtractUsers(IRRewriter &rewriter, Value oldValue, Value newValue) {
  SmallVector<tensor::ExtractOp> extractUsers;
  for (Operation *user : llvm::make_early_inc_range(oldValue.getUsers()))
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(user))
      extractUsers.push_back(extractOp);

  for (tensor::ExtractOp extractOp : extractUsers) {
    rewriter.setInsertionPoint(extractOp);
    auto newExtract = rewriter.create<tensor::ExtractOp>(
        extractOp.getLoc(), extractOp.getType(), newValue,
        extractOp.getIndices());
    rewriter.replaceOp(extractOp, newExtract.getResult());
  }

  return !extractUsers.empty();
}

bool rewriteDirectSliceMaterializeUsers(IRRewriter &rewriter, Value oldValue,
                                        Value newValue) {
  SmallVector<tensor::ExtractSliceOp> sliceUsers;
  for (Operation *user : llvm::make_early_inc_range(oldValue.getUsers()))
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user))
      sliceUsers.push_back(sliceOp);

  bool changed = false;
  for (tensor::ExtractSliceOp sliceOp : sliceUsers) {
    if (!sliceOp->hasOneUse())
      continue;
    auto materialize =
        dyn_cast<bufferization::MaterializeInDestinationOp>(
            *sliceOp->getUsers().begin());
    if (!materialize)
      continue;

    auto destType = dyn_cast<MemRefType>(materialize.getDest().getType());
    if (!destType)
      continue;

    Operation *insertionPoint = sliceOp.getOperation();
    Value dest = cloneMemrefViewAt(rewriter, materialize.getDest(), insertionPoint);
    if (!dest)
      continue;
    materialize.getDestMutable().assign(dest);
    materialize.setRestrict(true);

    auto elementType =
        cast<RankedTensorType>(newValue.getType()).getElementType();
    auto tensorType = RankedTensorType::get(destType.getShape(), elementType);
    rewriter.setInsertionPoint(insertionPoint);
    Value outputTensor = rewriter.create<bufferization::ToTensorOp>(
        materialize.getLoc(), tensorType, dest, true /* restrict */,
        true /* writable */);

    unsigned rank = destType.getRank();
    SmallVector<AffineMap> maps{
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    auto identity = rewriter.create<linalg::GenericOp>(
        materialize.getLoc(), TypeRange{tensorType}, ValueRange{newValue},
        ValueRange{outputTensor}, maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          builder.create<linalg::YieldOp>(loc, args[0]);
        });

    materialize->setOperand(0, identity->getResult(0));
    rewriter.eraseOp(sliceOp);
    changed = true;
  }

  return changed;
}

void HexagonPuntBufferPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  PuntCandidates candidates;

  funcOp.walk([&](memref::CopyOp copyOp) {
    DBG("punt buffer candidate: " << copyOp);

    if (!isaViewOfFuncInputBuffer(copyOp.getSource(), funcOp)) {
      DBG("___src is not a view of memref func arg");
      return WalkResult::advance();
    }

    populatePuntCandidate(copyOp, candidates);
    return WalkResult::advance();
  });

  // Punt the buffer use and erase the copy.
  for (auto &candidate : candidates) {
    auto copyOp = std::get<0>(candidate);
    auto toTensor = std::get<1>(candidate);
    auto replacement = std::get<2>(candidate);
    auto targetSubview = copyOp.getTarget().getDefiningOp<memref::SubViewOp>();
    Value allocValue =
        targetSubview ? Value(targetSubview.getSource()) : copyOp.getTarget();
    DBG("punted : " << copyOp);

    auto replacementType = dyn_cast<MemRefType>(replacement.getType());
    auto tensorType = dyn_cast<RankedTensorType>(toTensor.getType());
    if (!replacementType || !tensorType)
      continue;

    auto newTensorType = RankedTensorType::get(
        replacementType.getShape(), tensorType.getElementType(),
        tensorType.getEncoding());
    rewriter.setInsertionPoint(toTensor);
    auto newToTensor = rewriter.create<bufferization::ToTensorOp>(
        toTensor.getLoc(), newTensorType, replacement,
        true /* restrict */, false /* writable */);
    bool rebuiltGenericUsers = false;
    bool dynamicReplacement =
        toTensor.getResult().getType() != newToTensor.getResult().getType();
    if (dynamicReplacement) {
      while (rewriteMaterializeChainFromRoot(
          rewriter, toTensor.getResult(), newToTensor.getResult()))
        rebuiltGenericUsers = true;
      rebuiltGenericUsers =
          rebuildExtractUsers(rewriter, toTensor.getResult(),
                              newToTensor.getResult()) ||
          rebuiltGenericUsers;
      rebuiltGenericUsers =
          rewriteDirectSliceMaterializeUsers(rewriter, toTensor.getResult(),
                                             newToTensor.getResult()) ||
          rebuiltGenericUsers;
      rebuiltGenericUsers = rebuiltGenericUsers || toTensor->use_empty();
    } else {
      rebuiltGenericUsers =
          rebuildGenericUsers(rewriter, toTensor.getResult(), newToTensor.getResult());
    }
    if (!rebuiltGenericUsers) {
      rewriter.eraseOp(newToTensor);
      continue;
    }
    if (!dynamicReplacement)
      rewriter.replaceAllUsesWith(toTensor.getResult(), newToTensor.getResult());
    if (!toTensor->use_empty())
      continue;

    // We don't need the copy anymore.
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(toTensor);
    if (newToTensor->use_empty())
      rewriter.eraseOp(newToTensor);

    eraseDeadAllocUsers(rewriter, allocValue);
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonPuntBufferPass() {
  return std::make_unique<HexagonPuntBufferPass>();
}
