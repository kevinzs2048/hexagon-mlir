//===- HexKLToLLVMPass.cpp - HexKL to LLVM  conversion      ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass implements lowering of HexKL ops to LLVM.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLExternalFnNames.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLToLLVM.h"
#include "hexagon/Conversion/HexKLToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hexkl-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hexkl;

#define GEN_PASS_CLASSES
#include "hexagon/Conversion/HexKLToLLVM/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Existing lowering: hexkl::MatmulOp -> extern C matmul
// Keep this implementation intact per project guidance.
//===----------------------------------------------------------------------===//
static FailureOr<LLVM::LLVMFuncOp>
getMatmulRefFn(ModuleOp module, const LLVMTypeConverter &typeConverter,
               ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = module->getContext();

  // size_t → use the target index type (i32 on 32-bit, i64 on 64-bit)
  Type idxTy = typeConverter.getIndexType();

  // Pointers are opaque in modern LLVM dialect: use !llvm.ptr
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  SmallVector<Type, 6> argTys = {idxTy, idxTy, idxTy, ptrTy, ptrTy, ptrTy};
  return LLVM::lookupOrCreateFn(rewriter, module, hexkl::getMatmulMicroFnName(),
                                argTys, voidTy);
}

// Common helper: extract aligned pointer from lowered memref descriptor.
static Value alignedPtr(ConversionPatternRewriter &rewriter, Location loc,
                        Value memrefDesc) {
  MemRefDescriptor desc(memrefDesc);
  return desc.alignedPtr(rewriter, loc);
}

struct LowerMatmul : public ConvertOpToLLVMPattern<hexkl::MatmulOp> {
  using ConvertOpToLLVMPattern<hexkl::MatmulOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(hexkl::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    // Extract memref types
    auto lhsType = cast<mlir::MemRefType>(op.getLhs().getType());
    auto rhsType = cast<mlir::MemRefType>(op.getRhs().getType());
    auto resType = cast<mlir::MemRefType>(op.getOuts().getType());

    // --- Ensure the callee is declared ---
    ModuleOp module = op->getParentOfType<ModuleOp>();
    const auto *typeConverter = getTypeConverter();
    auto fn = getMatmulRefFn(module, *typeConverter, rewriter);
    if (failed(fn))
      return failure();

    // --- Extract aligned data pointers using helper function ---
    Value outPtr = alignedPtr(rewriter, loc, adaptor.getOuts());
    Value lhsPtr = alignedPtr(rewriter, loc, adaptor.getLhs());
    Value rhsPtr = alignedPtr(rewriter, loc, adaptor.getRhs());

    // --- Extract matrix extents as index-typed values ---
    // lhs: [n_row, n_inner], rhs: [n_inner, n_col], out: [n_row, n_col]
    MemRefDescriptor lhsDesc(adaptor.getLhs());
    MemRefDescriptor outDesc(adaptor.getOuts());
    Value nRow = lhsDesc.size(rewriter, loc, /*dim=*/0);
    Value nInner = lhsDesc.size(rewriter, loc, /*dim=*/1);
    Value nCol = outDesc.size(rewriter, loc, /*dim=*/1);

    // --- Build argument list and emit the call ---
    SmallVector<Value, 6> operands = {nRow,   nCol,   nInner,
                                      outPtr, lhsPtr, rhsPtr};
    auto fnCallee = FlatSymbolRefAttr::get((*fn).getOperation());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnCallee,
                                              operands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helpers: extern declaration and common utilities
//===----------------------------------------------------------------------===//
static Value castToI32(ConversionPatternRewriter &rewriter, Location loc,
                       Value v) {
  auto i32Ty = rewriter.getI32Type();
  if (v.getType() == i32Ty)
    return v;
  auto intTy = dyn_cast<IntegerType>(v.getType());
  assert(intTy && "castToI32 expects an integer-typed value");
  unsigned w = intTy.getWidth();
  if (w > 32)
    return rewriter.create<LLVM::TruncOp>(loc, i32Ty, v);
  if (w < 32)
    return rewriter.create<LLVM::ZExtOp>(loc, i32Ty, v);
  return v;
}
static FailureOr<LLVM::LLVMFuncOp>
getNoArgI32RetFn(ModuleOp module, ConversionPatternRewriter &rewriter,
                 StringRef name) {
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 0> argTys;
  auto fn = LLVM::lookupOrCreateFn(rewriter, module, name, argTys, i32Ty);
  if (succeeded(fn)) {
    auto funcOp = *fn;
    auto ctx = rewriter.getContext();
    funcOp->setAttr(StringAttr::get(ctx, "llvm.readnone"),
                    rewriter.getUnitAttr());
    funcOp->setAttr(StringAttr::get(ctx, "llvm.nounwind"),
                    rewriter.getUnitAttr());
  }
  return fn;
}

static FailureOr<LLVM::LLVMFuncOp>
getSetupAccReadF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  auto i32Vec = SmallVector<Type, 2>{ptrTy, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxSetupAccReadF16FnName(), i32Vec, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getAccClearF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  SmallVector<Type, 0> argTys;
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxAccClearF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getAccReadF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{ptrTy, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxAccReadF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getCopySubmatrixToF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 7> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxCopySubmatrixToF16FnName(), argTys,
                                voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getRmToAhF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{ptrTy, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxRmToAhF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getRmToWhF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 6> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxRmToWhF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getMmF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{ptrTy, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module, hexkl::getHmxMmF16FnName(),
                                argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getAhToRmF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{ptrTy, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxAhToRmF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getCopyF16ToF32SubmatrixFn(ModuleOp module,
                           ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 7> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxCopyF16ToF32SubmatrixFnName(),
                                argTys, voidTy);
}

//===----------------------------------------------------------------------===//
// Lower micro ops to extern calls
//===----------------------------------------------------------------------===//

struct LowerSetupAccReadF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXSetupAccReadF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXSetupAccReadF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXSetupAccReadF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op

    // vtcm_base
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    MemRefDescriptor hmxDesc(adaptor.getHmxBlock());

    // vtcm_size (element count of memref<?xi8>)
    Value vtcmSizeIdx = hmxDesc.size(rewriter, loc, 0);
    Value vtcmSizeI32 = castToI32(rewriter, loc, vtcmSizeIdx);

    auto cfgFn =
        getNoArgI32RetFn(module, rewriter, hexkl::getHmxConfigSizeFnName());
    if (failed(cfgFn))
      return failure();
    auto cfgCallee = FlatSymbolRefAttr::get((*cfgFn).getOperation());
    Value cfgSize = rewriter
                        .create<LLVM::CallOp>(loc, rewriter.getI32Type(),
                                              cfgCallee, ValueRange{})
                        .getResult();

    // hmx_config_offset = vtcm_size - config_size
    Value cfgOffset = rewriter.create<LLVM::SubOp>(loc, vtcmSizeI32, cfgSize);

    // call setup_acc_read_f16(vtcm_base, hmx_config_offset)
    auto setupFn = getSetupAccReadF16Fn(module, rewriter);
    if (failed(setupFn))
      return failure();
    auto setupCallee = FlatSymbolRefAttr::get((*setupFn).getOperation());
    SmallVector<Value> args{vtcmBase, cfgOffset};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, setupCallee,
                                              args);
    return success();
  }
};

struct LowerAccClearF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXAccClearF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXAccClearF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXAccClearF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op
    auto fn = getAccClearF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee,
                                              ValueRange{});
    return success();
  }
};

struct LowerAccReadF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXAccReadF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXAccReadF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXAccReadF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op

    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    MemRefDescriptor hmxDesc(adaptor.getHmxBlock());
    Value vtcmSizeIdx = hmxDesc.size(rewriter, loc, 0);
    Value vtcmSizeI32 = castToI32(rewriter, loc, vtcmSizeIdx);

    auto cfgFn =
        getNoArgI32RetFn(module, rewriter, hexkl::getHmxConfigSizeFnName());
    if (failed(cfgFn))
      return failure();
    auto cfgCallee = FlatSymbolRefAttr::get((*cfgFn).getOperation());
    Value cfgSize = rewriter
                        .create<LLVM::CallOp>(loc, rewriter.getI32Type(),
                                              cfgCallee, ValueRange{})
                        .getResult();

    Value cfgOffset = rewriter.create<LLVM::SubOp>(loc, vtcmSizeI32, cfgSize);

    Value outOffset = adaptor.getOutOffset();
    auto fn = getAccReadF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, cfgOffset, outOffset};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerCopySubmatrixToF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXCopySubmatrixToF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXCopySubmatrixToF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXCopySubmatrixToF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op

    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value srcPtr = alignedPtr(rewriter, loc, adaptor.getSrc());

    auto fn = getCopySubmatrixToF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase,
                            adaptor.getOutOffset(),
                            srcPtr,
                            adaptor.getTileRow(),
                            adaptor.getTileCol(),
                            adaptor.getInputRows(),
                            adaptor.getInputCols()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerRmToAhF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXRmToAhF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXRmToAhF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXRmToAhF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    auto fn = getRmToAhF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, adaptor.getActivationOutOffset(),
                            adaptor.getFlatInOffset()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerRmToWhF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXRmToWhF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXRmToWhF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXRmToWhF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value srcPtr = alignedPtr(rewriter, loc, adaptor.getSrc());
    auto fn = getRmToWhF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase,
                            adaptor.getWeightOffset(),
                            srcPtr,
                            adaptor.getTileRow(),
                            adaptor.getTileCol(),
                            adaptor.getWtCols()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerMmF16 : public ConvertOpToLLVMPattern<hexkl::MicroHMXMmF16Op> {
  using ConvertOpToLLVMPattern<hexkl::MicroHMXMmF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXMmF16Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    auto fn = getMmF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, adaptor.getActivationOffset(),
                            adaptor.getWeightOffset()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerAhToRmF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXAhToRmF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXAhToRmF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXAhToRmF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    auto fn = getAhToRmF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, adaptor.getFlatOutOffset(),
                            adaptor.getActivationInOffset()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerCopyF16ToF32Submatrix
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXCopyF16ToF32SubmatrixOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXCopyF16ToF32SubmatrixOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXCopyF16ToF32SubmatrixOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value dstPtr = alignedPtr(rewriter, loc, adaptor.getDst());
    auto fn = getCopyF16ToF32SubmatrixFn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase,
                            adaptor.getInOffset(),
                            dstPtr,
                            adaptor.getTileRow(),
                            adaptor.getTileCol(),
                            adaptor.getOutputRows(),
                            adaptor.getOutputCols()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexkl::HmxCoreDotOp -> inline HMX intrinsics (deep-streaming loop)
//===----------------------------------------------------------------------===//
//
// Reproduces spike_hmx_mm_fp16: a deep-tile-streaming HMX fp16 matmul over
// VTCM-resident activation-half/weight-half tiles. Emits the 5 HMX intrinsics
// as declared external llvm.func + llvm.call (the proven Task-1 recipe; do NOT
// use llvm.call_intrinsic, which the LLVM-22 translator cannot resolve). The
// enclosing llvm.func is tagged with target_features<["+hmx","+hmxv73"]> so the
// Hexagon backend's UseHMX isel predicate fires.

// Dotted LLVM intrinsic symbol names.
static constexpr StringRef kHmxClrAcc = "llvm.hexagon.M8.mxclracc.hf";
static constexpr StringRef kHmxActDeep = "llvm.hexagon.M8.mxmem.sm.act.hf";
static constexpr StringRef kHmxWei = "llvm.hexagon.M8.mxmem.wei.hf";
static constexpr StringRef kHmxCvtr = "llvm.hexagon.M8.mxcvtr.sat.hf";
static constexpr StringRef kHmxBias = "llvm.hexagon.M8.mxmem2.bias";

// HMX tile constants (mirror the spike): 1024 fp16 elems * 2 bytes = 2048 bytes
// per tile; deep-streaming blocks of up to 32 tiles per mxmem call.
static constexpr int64_t kHmxTileBytes = 2048;
static constexpr int64_t kHmxMaxKBlock = 32;

// Lazily declare an HMX intrinsic as an external llvm.func with the given
// signature, returning a callee symbol ref. lookupOrCreateFn reuses an existing
// declaration if already present.
static FailureOr<FlatSymbolRefAttr>
declareHmxIntrinsic(ModuleOp module, ConversionPatternRewriter &rewriter,
                    StringRef name, ArrayRef<Type> argTys, Type retTy) {
  SmallVector<Type> args(argTys.begin(), argTys.end());
  auto fn = LLVM::lookupOrCreateFn(rewriter, module, name, args, retTy);
  if (failed(fn))
    return failure();
  return FlatSymbolRefAttr::get((*fn).getOperation());
}

struct LowerHmxCoreDot
    : public ConvertOpToLLVMPattern<hexkl::HmxCoreDotOp> {
  using ConvertOpToLLVMPattern<hexkl::HmxCoreDotOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(hexkl::HmxCoreDotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    auto i32Ty = rewriter.getI32Type();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    // --- Tag the enclosing llvm.func with the HMX target features. By the time
    // this pass runs, func.func has already been lowered to llvm.func (see the
    // LinalgToLLVM pipeline: ConvertFuncToLLVM precedes HexKLToLLVM). The attr
    // translates to "target-features"="+hmx,+hmxv73", which is load-bearing for
    // HMX instruction selection.
    if (auto enclosing = op->getParentOfType<LLVM::LLVMFuncOp>()) {
      auto tf = LLVM::TargetFeaturesAttr::get(ctx, {"+hmx", "+hmxv73"});
      enclosing.setTargetFeaturesAttr(tf);
    }

    // --- Declare the 5 HMX intrinsics (lazily). ---
    auto clrAccFn = declareHmxIntrinsic(module, rewriter, kHmxClrAcc, {}, voidTy);
    auto actFn = declareHmxIntrinsic(module, rewriter, kHmxActDeep,
                                     {i32Ty, i32Ty}, voidTy);
    auto weiFn = declareHmxIntrinsic(module, rewriter, kHmxWei,
                                     {i32Ty, i32Ty}, voidTy);
    auto cvtrFn = declareHmxIntrinsic(module, rewriter, kHmxCvtr,
                                      {ptrTy, i32Ty}, voidTy);
    auto biasFn = declareHmxIntrinsic(module, rewriter, kHmxBias, {ptrTy},
                                      voidTy);
    if (failed(clrAccFn) || failed(actFn) || failed(weiFn) || failed(cvtrFn) ||
        failed(biasFn))
      return failure();

    // --- VTCM base pointer + byte offsets -> typed pointers. ---
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    auto i8Ty = rewriter.getI8Type();

    auto gepI8 = [&](Value base, Value byteOff) -> Value {
      return rewriter.create<LLVM::GEPOp>(loc, ptrTy, i8Ty, base,
                                          ValueRange{byteOff});
    };
    auto cst = [&](int64_t v) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                               rewriter.getI32IntegerAttr(v));
    };

    Value outOff = adaptor.getOutOff();
    Value actOff = adaptor.getActOff();
    Value wtOff = adaptor.getWtOff();
    Value biasOff = adaptor.getBiasOff();
    Value nRowTiles = adaptor.getNRowTiles();
    Value nColTiles = adaptor.getNColTiles();
    Value nDotTiles = adaptor.getNDotTiles();

    Value outBase = gepI8(vtcmBase, outOff);
    Value actBase = gepI8(vtcmBase, actOff);
    Value wtBase = gepI8(vtcmBase, wtOff);
    Value biasPtr = gepI8(vtcmBase, biasOff);

    Value c0 = cst(0);
    Value c1 = cst(1);
    Value tileBytes = cst(kHmxTileBytes);
    Value maxKBlock = cst(kHmxMaxKBlock);
    // dot stride in bytes = n_dot_tiles * 2048
    Value dotStride = rewriter.create<LLVM::MulOp>(loc, nDotTiles, tileBytes);

    // bias load once.
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, *biasFn,
                                  ValueRange{biasPtr});

    // Build a 3-level LLVM-IR loop nest with explicit basic blocks:
    //   for r in [0, n_row_tiles):
    //     for c in [0, n_col_tiles):
    //       mxclracc
    //       for k in [0, n_dot_tiles) step kb=min(rem,32):
    //         mxmem.sm.act.hf / mxmem.wei.hf
    //       mxcvtr.sat.hf
    //
    // We split the current block, then create header/body/latch blocks per
    // level. The conversion rewriter requires manual block/branch construction
    // here because scf has already been lowered upstream.
    Block *origBlock = rewriter.getInsertionBlock();
    Block *contBlock =
        rewriter.splitBlock(origBlock, rewriter.getInsertionPoint());

    // ---- Outer (row) loop ----
    Block *rHeader = rewriter.createBlock(contBlock, {i32Ty}, {loc});
    Block *rBody = rewriter.createBlock(contBlock);

    rewriter.setInsertionPointToEnd(origBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0}, rHeader);

    // row header: cond r < n_row_tiles
    rewriter.setInsertionPointToEnd(rHeader);
    Value r = rHeader->getArgument(0);
    Value rCond = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                                r, nRowTiles);
    // rBody filled below; contBlock is loop exit.
    rewriter.create<LLVM::CondBrOp>(loc, rCond, rBody, ValueRange{}, contBlock,
                                    ValueRange{});

    // row body
    rewriter.setInsertionPointToEnd(rBody);
    // row_act_base = act_base + r * dotStride
    Value rDot = rewriter.create<LLVM::MulOp>(loc, r, dotStride);
    Value rowActBase = gepI8(actBase, rDot);

    // ---- Column loop ----
    Block *cHeader = rewriter.createBlock(contBlock, {i32Ty}, {loc});
    Block *cBody = rewriter.createBlock(contBlock);
    Block *rLatch = rewriter.createBlock(contBlock);

    rewriter.setInsertionPointToEnd(rBody);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0}, cHeader);

    // col header
    rewriter.setInsertionPointToEnd(cHeader);
    Value c = cHeader->getArgument(0);
    Value cCond = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                                c, nColTiles);
    rewriter.create<LLVM::CondBrOp>(loc, cCond, cBody, ValueRange{}, rLatch,
                                    ValueRange{});

    // row latch: r++ ; back to row header
    rewriter.setInsertionPointToEnd(rLatch);
    Value rNext = rewriter.create<LLVM::AddOp>(loc, r, c1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{rNext}, rHeader);

    // col body
    rewriter.setInsertionPointToEnd(cBody);
    // clear accumulator
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, *clrAccFn, ValueRange{});
    // col_wt_base = wt_base + c * dotStride
    Value cDot = rewriter.create<LLVM::MulOp>(loc, c, dotStride);
    Value colWtBase = gepI8(wtBase, cDot);

    // ---- K deep-streaming loop ----
    // carries: (k, row_tiles_ptr, col_tiles_ptr)
    Block *kHeader =
        rewriter.createBlock(contBlock, {i32Ty, ptrTy, ptrTy}, {loc, loc, loc});
    Block *kBody = rewriter.createBlock(contBlock);
    Block *cLatch = rewriter.createBlock(contBlock);

    rewriter.setInsertionPointToEnd(cBody);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0, rowActBase, colWtBase},
                               kHeader);

    // k header
    rewriter.setInsertionPointToEnd(kHeader);
    Value k = kHeader->getArgument(0);
    Value rowTilesPtr = kHeader->getArgument(1);
    Value colTilesPtr = kHeader->getArgument(2);
    Value kCond = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                                k, nDotTiles);
    rewriter.create<LLVM::CondBrOp>(loc, kCond, kBody, ValueRange{}, cLatch,
                                    ValueRange{});

    // col latch: read out acc, c++, back to col header
    rewriter.setInsertionPointToEnd(cLatch);
    // out_tile = out_base + (r*n_col_tiles + c) * 2048
    Value rTimesNcol = rewriter.create<LLVM::MulOp>(loc, r, nColTiles);
    Value outTileIdx = rewriter.create<LLVM::AddOp>(loc, rTimesNcol, c);
    Value outTileByte =
        rewriter.create<LLVM::MulOp>(loc, outTileIdx, tileBytes);
    Value outTilePtr = gepI8(outBase, outTileByte);
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, *cvtrFn,
                                  ValueRange{outTilePtr, c0});
    Value cNext = rewriter.create<LLVM::AddOp>(loc, c, c1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{cNext}, cHeader);

    // k body
    rewriter.setInsertionPointToEnd(kBody);
    // rem = n_dot_tiles - k ; kb = min(rem, 32)
    Value rem = rewriter.create<LLVM::SubOp>(loc, nDotTiles, k);
    Value remLtMax = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::slt, rem, maxKBlock);
    Value kb = rewriter.create<LLVM::SelectOp>(loc, remLtMax, rem, maxKBlock);
    // range = 2048*kb - 1
    Value kbBytes = rewriter.create<LLVM::MulOp>(loc, kb, tileBytes);
    Value range = rewriter.create<LLVM::SubOp>(loc, kbBytes, c1);
    // act/wei take i32 addresses (ptrtoint the VTCM pointer; VTCM is sub-4GB).
    Value rowAddrI32 =
        rewriter.create<LLVM::PtrToIntOp>(loc, i32Ty, rowTilesPtr);
    Value colAddrI32 =
        rewriter.create<LLVM::PtrToIntOp>(loc, i32Ty, colTilesPtr);
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, *actFn,
                                  ValueRange{rowAddrI32, range});
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, *weiFn,
                                  ValueRange{colAddrI32, range});
    // advance row/col tile pointers by kb*2048; k += kb
    Value rowNext = gepI8(rowTilesPtr, kbBytes);
    Value colNext = gepI8(colTilesPtr, kbBytes);
    Value kNext = rewriter.create<LLVM::AddOp>(loc, k, kb);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{kNext, rowNext, colNext},
                               kHeader);

    // Erase the original op; control falls through contBlock.
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Setup the Lowering Pass and patterns
//===----------------------------------------------------------------------===//

void populateHexKLToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  // Keep existing matmul lowering.
  patterns.add<LowerMatmul>(converter);
  // Lower micro ops to externs.
  patterns.add<LowerSetupAccReadF16, LowerAccClearF16, LowerAccReadF16,
               LowerCopySubmatrixToF16, LowerRmToAhF16, LowerRmToWhF16,
               LowerMmF16, LowerAhToRmF16, LowerCopyF16ToF32Submatrix>(
      converter);
  // Lower the deep-streaming HMX core-dot to inline HMX intrinsics.
  patterns.add<LowerHmxCoreDot>(converter);
}

struct HexKLToLLVMPass : public HexKLToLLVMBase<HexKLToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, LLVM::LLVMDialect, HexKLDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);
    LowerToLLVMOptions options(context,
                               dataLayoutAnalysis.getAtOrAbove(moduleOp));
    LLVMTypeConverter typeConverter(context, options);

    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<HexKLDialect>();

    hexagon::addTypeConversions(context, typeConverter);
    populateHexKLToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexkl::createHexKLToLLVMPass() {
  return std::make_unique<HexKLToLLVMPass>();
}
