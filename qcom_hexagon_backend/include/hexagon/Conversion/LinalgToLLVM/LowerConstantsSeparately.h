//===- LowerConstantsSeparately.h - Separate out large constants ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
//  Separates the large constant tensors (typically conv/matmul weights) by
//  lowering them into separate shared objects (.so).
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_LINALGTOLLVM_LOWERCONSTANTSSEPARATELY_H
#define HEXAGON_CONVERSION_LINALGTOLLVM_LOWERCONSTANTSSEPARATELY_H

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h" // For divideCeil
#include "llvm/ADT/SmallPtrSet.h"

#include "hexagon/Conversion/LinalgToLLVM/Passes.h" // For the base class LowerConstantsSeparatelyBase

#define DEBUG_TYPE "hexagon-lower-constants-separately"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LOWERCONSTANTSSEPARATELY
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

// Kept only for easier debugging to print the data of a blob (from the
// dialect_resources) that's in hexadecimal format with two characters per byte
void printValueOfConstant(ArrayRef<char> originalData) {
  for (char c : originalData) {
    // Print each byte as a two-digit hexadecimal number.
    // std::hex sets the output format to hexadecimal.
    // std::setw(2) ensures each byte is printed with at least 2 characters.
    // std::setfill('0') pads with '0' if the byte is less than 2 characters.
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << (int)(unsigned char)c;
  }
  std::cout << std::endl;
}

// This class implements the main logic of the constant splitting, which will
// move constant tensors to (potentially many) new separate modules, which will
// all be compiled and linked individually.
struct LowerConstantsSeparatelyPass
    : public ::impl::LowerConstantsSeparatelyBase<
          LowerConstantsSeparatelyPass> {

  // Maximum size above which a module is considered "full" and a new one will
  // be created (can happen several times, leading to several const-only mods)
  static constexpr uint64_t maximumBytesPerSharedObject =
      500'000'000; // 0.5 GB

  // Opt-in only: assertion strings belong to the code which references them,
  // not to a reusable model-weight DSO. LLVM's print-call helper can number
  // these private symbols differently for successive modules in one process.
  // Retaining their original definitions also retains all assertion behavior.
  static constexpr llvm::StringLiteral keepAssertMessagesAttr =
      "hexagon.keep_assert_messages_in_main";

  static bool isPrivateAssertMessage(LLVM::GlobalOp op) {
    if (!op.getConstant() || op.getLinkage() != LLVM::Linkage::Private)
      return false;
    llvm::StringRef name = op.getSymName();
    if (name != "assert_msg") {
      if (!name.consume_front("assert_msg_") || name.empty())
        return false;
      for (char c : name)
        if (c < '0' || c > '9')
          return false;
    }
    auto type = dyn_cast<LLVM::LLVMArrayType>(op.getType());
    auto value = dyn_cast_or_null<DenseIntElementsAttr>(op.getValueAttr());
    if (!type || !type.getElementType().isInteger(8) || !value ||
        value.getType().getRank() != 1 ||
        !value.getElementType().isInteger(8) ||
        type.getNumElements() != uint64_t(value.getNumElements()) ||
        !type.getNumElements() || type.getNumElements() > 65536)
      return false;
    return (*(value.getValues<llvm::APInt>().begin() +
              (value.getNumElements() - 1))).isZero();
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  // Utility function to compute the memory footprint (in bytes) of a constant
  // tensor op, which will be used to know when to create a new mlir module.
  // If passed an operation that's not a valid constant tensor (containing
  // int or float values) it returns 0.
  uint64_t estimateConstantTensorMemoryFootprint(mlir::LLVM::GlobalOp op) {
    // Ensure the global is marked constant
    if (!op.getConstant()) {
      return 0;
    }

    mlir::Type type = op.getType();
    mlir::Type previousType;
    uint64_t totalElements = 1;

    // Unwrap nested LLVM array types to get shape and element type
    while (auto arrayType = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
      uint64_t count = arrayType.getNumElements();
      if (count && totalElements > std::numeric_limits<uint64_t>::max() / count) {
        return std::numeric_limits<uint64_t>::max();
      }
      totalElements *= count;
      previousType = type;
      type = arrayType.getElementType();
      // Break if type doesn't change (safety against infinite loop)
      if (type == previousType) {
        return 0;
      }
    }

    // Now `type` should be the element type (int or float)
    if (!type.isIntOrFloat()) {
      std::cerr << "Element type is not int or float." << std::endl;
      return 0;
    }

    uint64_t elementBitWidth = type.getIntOrFloatBitWidth();
    uint64_t elementByteSize = llvm::divideCeil(elementBitWidth, uint64_t{8});
    if (elementByteSize &&
        totalElements > std::numeric_limits<uint64_t>::max() / elementByteSize)
      return std::numeric_limits<uint64_t>::max();
    return totalElements * elementByteSize;
  }

  // Main function of the pass, which goes through the module, looking for
  // constants to be moved to (potentially) many new mlir modules. Does so
  // while making sure that each produced module stays under the limit
  // defined by `maximumBytesPerSharedObject`. Takes care not only of
  // moving the constants, but also of moving the associated entried defined
  // in the dialect_resources
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext &newContext = *(mod.getContext());
    producedModules.clear();

    llvm::SmallPtrSet<Operation *, 16> retainedMessages;
    if (Attribute flag = mod->getAttr(keepAssertMessagesAttr)) {
      if (!isa<UnitAttr>(flag)) {
        mod.emitError() << keepAssertMessagesAttr << " must be a unit attribute";
        signalPassFailure();
        return;
      }
      // Validate before mutating anything. A reference from another global
      // initializer could cross the new DSO boundary; do not hide that case.
      for (auto global : mod.getOps<LLVM::GlobalOp>()) {
        if (!isPrivateAssertMessage(global))
          continue;
        auto uses = SymbolTable::getSymbolUses(global, mod);
        if (!uses || llvm::any_of(*uses, [](const SymbolTable::SymbolUse &use) {
              return !use.getUser()->getParentOfType<LLVM::LLVMFuncOp>();
            })) {
          global.emitError("cannot retain assertion message with non-function "
                           "or unknown symbol uses");
          signalPassFailure();
          return;
        }
        retainedMessages.insert(global.getOperation());
      }
    }

    // Create a new MLIR module in which we'll be putting the current op
    DBG("[LowerConstantsSeparately] Creating a new module for constants only"
        << "\n");
    mlir::ModuleOp newMod =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&newContext));
    newMod->setAttrs(mod->getAttrs());
    // The size of the new module will be the sum of the weights (in bytes) of
    // the constant tensors it contains. It is 0 for now since it's empty.
    uint64_t newModSize = 0;
    bool newModHasConstants = false;

    // Walking the original MLIR module, trying to find global constants to move
    // to the new MLIR module we just made
    mod.walk([&](mlir::Operation *op) {
      // If it's a GlobalOp...
      if (auto globalOp = llvm::dyn_cast<mlir::LLVM::GlobalOp>(op)) {
        // And if it's a constant...
        if (globalOp.getConstant() && !retainedMessages.contains(op)) {
          // 1 - Copy the constant definition to the new module being created
          // ------------------------------------------------------------------------
          // Then we clone the global constant
          mlir::LLVM::GlobalOp clonedOp = globalOp.clone();
          // We set its linkage attribute to External, to make sure it will be
          // accessible from other .o
          auto linkageAttrClonedOp =
              LLVM::LinkageAttr::get(&newContext, LLVM::Linkage::External);
          clonedOp.setLinkageAttr(linkageAttrClonedOp);

          // Compute the size (in bytes) needed by the current constant
          uint64_t currentOpSize =
              estimateConstantTensorMemoryFootprint(globalOp);

          // If the current op can't fit in the current module for constants
          bool exceedsModuleLimit =
              currentOpSize > maximumBytesPerSharedObject ||
              newModSize > maximumBytesPerSharedObject - currentOpSize;
          if (newModHasConstants && exceedsModuleLimit) {
            // Push the current mod into the collection of produced modules
            producedModules.push_back(newMod);
            // Create a new MLIR module in which we'll be putting the current op
            DBG("[LowerConstantsSeparately] Creating another new "
                "module for constants only"
                << "\n");
            newMod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&newContext));
            newMod->setAttrs(mod->getAttrs());
            // It's size is 0 for now
            newModSize = 0;
            newModHasConstants = false;
          }

          // Account for the constant even when it alone exceeds the preferred
          // module limit. Such a constant cannot be split here, but it must not
          // cause an empty module or make the following constants look free.
          newModSize = currentOpSize > maximumBytesPerSharedObject
                           ? currentOpSize
                           : newModSize + currentOpSize;
          newModHasConstants = true;

          // And we insert the cloned operation (constant definition) into the
          // new module
          newMod.push_back(clonedOp);

          // 2 - Replace the constant by an extern in the main module (for the
          // code)
          // ------------------------------------------------------------------------
          OpBuilder builder(globalOp.getContext());
          builder.setInsertionPoint(globalOp);

          // Create a new extern GlobalOp
          auto externGlobalOp =
              LLVM::GlobalOp::create(builder, globalOp.getLoc(),
                                     /*type=*/globalOp.getType(),
                                     /*isConstant=*/false,
                                     /*linkage=*/LLVM::Linkage::External,
                                     /*name=*/globalOp.getSymName(),
                                     /*value=*/Attribute{});

          // Replace the original GlobalOp that was the constant definition with
          // the new "extern" GlobalOp, which references a constant from the
          // outside
          globalOp.getOperation()->replaceAllUsesWith(
              externGlobalOp.getOperation());
          globalOp.erase();
        }
      }
    });

    // The original module and every constants module deliberately share the
    // same MLIRContext, and therefore the same BuiltinDialect resource blob
    // manager. Keep dense_resource blobs under their original keys: moving a
    // blob and reinserting it into that same manager creates a suffixed key,
    // while the cloned DenseResourceElementsAttr still references the old key.

    // Push the final non-empty constants module. With no constants, returning
    // no auxiliary module avoids generating and linking a useless empty DSO.
    if (newModHasConstants)
      producedModules.push_back(newMod);
    else
      newMod.erase();
  }

  // Method to access the newly produced modules
  std::vector<ModuleOp> getProducedModules() const { return producedModules; }

private:
  // Modules produced by the pass, which contain only constants (moved
  // from the original module)
  std::vector<ModuleOp> producedModules;

}; // end of class

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLowerConstantsSeparatelyPass();

#endif // HEXAGON_CONVERSION_LINALGTOLLVM_LOWERCONSTANTSSEPARATELY_H
