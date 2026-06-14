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
// Lower hexkl::HvxLayoutWeightOp -> inline HVX vscattermhw weight-half layout
//===----------------------------------------------------------------------===//
//
// Emits the fast HVX weight-half tile layout proven in Task 1, replacing the
// per-tile vendor rm_to_wh loop. Consumes the weight in the [K][N] row-major
// orientation the matmul provides (NO transpose): for each K-row it loads the
// contiguous N values and scatters each fp16 to its weight-half byte offset via
// V6.vscattermhw (halfword scatter with word offsets), which lets every fp16 be
// placed independently — exactly what reading [K][N] directly requires.
//
// Weight-half within-tile byte offset (empirically derived, byte-identical to
// rm_to_wh):  off(ki,ni) = (ki/2)*128 + (ki%2)*2 + ni*4   for ki,ni in 0..31,
// tile(kt,nt) = nt*n_k_tiles + kt at 2048 bytes each.
//
// vscattermhw's <64 x i32> offset operand is DEINTERLEAVED: pair-lane m holds
// the offset for the even halfword (data fp16 lane 2m), pair-lane 32+m for the
// odd halfword (2m+1). The constant column tables below are built in that order.

static constexpr StringRef kHvxScatterMhw = "llvm.hexagon.V6.vscattermhw.128B";

struct LowerHvxLayoutWeight
    : public ConvertOpToLLVMPattern<hexkl::HvxLayoutWeightOp> {
  using ConvertOpToLLVMPattern<
      hexkl::HvxLayoutWeightOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(hexkl::HvxLayoutWeightOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    auto i32Ty = rewriter.getI32Type();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i8Ty = rewriter.getI8Type();
    auto v64i32 = VectorType::get({64}, i32Ty);
    auto v32i32 = VectorType::get({32}, i32Ty);

    // --- Tag the enclosing llvm.func with the HVX target features (load-bearing
    // for HVX scatter isel, mirrors the HMX core-dot's +hmx tagging). Preserve
    // any features already present (e.g. +hmx,+hmxv73 from a sibling core-dot in
    // the same func) by merging rather than overwriting. ---
    if (auto enclosing = op->getParentOfType<LLVM::LLVMFuncOp>()) {
      llvm::SmallVector<StringRef> feats;
      llvm::SmallVector<std::string> owned;
      if (auto existing = enclosing.getTargetFeaturesAttr())
        for (StringRef f : existing.getFeatures())
          feats.push_back(f);
      for (StringRef want : {StringRef("+hvxv73"), StringRef("+hvx-length128b")})
        if (!llvm::is_contained(feats, want))
          feats.push_back(want);
      enclosing.setTargetFeaturesAttr(
          LLVM::TargetFeaturesAttr::get(ctx, feats));
    }

    // --- Declare V6.vscattermhw.128B(i32 Rt, i32 Mu, <64xi32> off,
    //     <32xi32> val) -> void as an external llvm.func (the proven recipe). ---
    auto scatFn = declareHmxIntrinsic(module, rewriter, kHvxScatterMhw,
                                      {i32Ty, i32Ty, v64i32, v32i32}, voidTy);
    if (failed(scatFn))
      return failure();

    auto cst = [&](int64_t v) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                               rewriter.getI32IntegerAttr(v));
    };
    auto gepI8 = [&](Value base, Value byteOff) -> Value {
      return rewriter.create<LLVM::GEPOp>(loc, ptrTy, i8Ty, base,
                                          ValueRange{byteOff});
    };

    // --- Build the two compile-time-constant <64xi32> column tables. ---
    //   colOff[lane]   = (ni % 32) * 4        (column byte offset within a tile)
    //   upperLane[lane]= 1 if ni >= 32 else 0 (selects the 2nd tile of the pair)
    // where ni is the N-within-chunk for the deinterleaved lane (m -> 2m,
    // 32+m -> 2m+1). The upper-tile lanes get an extra n_k_tiles*2048 added at
    // runtime.
    SmallVector<int32_t, 64> colOffVals(64), upperVals(64);
    for (int m = 0; m < 32; ++m) {
      int ne = 2 * m, no = 2 * m + 1;
      colOffVals[m] = (ne % 32) * 4;
      upperVals[m] = (ne / 32);
      colOffVals[32 + m] = (no % 32) * 4;
      upperVals[32 + m] = (no / 32);
    }
    Value colOff = rewriter.create<LLVM::ConstantOp>(
        loc, v64i32,
        DenseElementsAttr::get(v64i32, ArrayRef<int32_t>(colOffVals)));
    Value upperLane = rewriter.create<LLVM::ConstantOp>(
        loc, v64i32,
        DenseElementsAttr::get(v64i32, ArrayRef<int32_t>(upperVals)));

    // --- Operands / derived scalars. ---
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value srcPtr = alignedPtr(rewriter, loc, adaptor.getSrc());
    Value wtOff = adaptor.getWtOff();
    Value K = adaptor.getK();
    Value N = adaptor.getN();

    Value c0 = cst(0);
    Value c1 = cst(1);
    Value c2 = cst(2);
    Value c32 = cst(32);
    Value c64 = cst(64);
    Value c128 = cst(128);
    Value tileBytes = cst(kHmxTileBytes);   // 2048
    // n_k_tiles = K / 32
    Value nKTiles = rewriter.create<LLVM::SDivOp>(loc, K, c32);
    // pair stride = n_k_tiles * 2048 (byte distance between adjacent N-tiles)
    Value pairStride = rewriter.create<LLVM::MulOp>(loc, nKTiles, tileBytes);
    // region = pairStride + 2048 - 1
    Value regionTmp = rewriter.create<LLVM::AddOp>(loc, pairStride, tileBytes);
    Value region = rewriter.create<LLVM::SubOp>(loc, regionTmp, c1);
    // splat(pairStride) and the upper-lane runtime offset = upperLane*pairStride
    Value pairStrideSplat =
        rewriter.create<LLVM::ConstantOp>(loc, v64i32,
                                          rewriter.getZeroAttr(v64i32));
    {
      // build a real splat: insert into lane 0 then shufflevector broadcast.
      Value undef = rewriter.create<LLVM::UndefOp>(loc, v64i32);
      Value ins = rewriter.create<LLVM::InsertElementOp>(loc, v64i32, undef,
                                                         pairStride, c0);
      SmallVector<int32_t, 64> zeros(64, 0);
      pairStrideSplat = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, ins, undef, ArrayRef<int32_t>(zeros));
    }
    Value upperOff =
        rewriter.create<LLVM::MulOp>(loc, upperLane, pairStrideSplat);
    // baseOff64 = colOff + upperOff  (the part independent of k)
    Value baseOff64 = rewriter.create<LLVM::AddOp>(loc, colOff, upperOff);

    // VTCM dst base for the weight region = vtcmBase + wt_off.
    Value wtBase = gepI8(vtcmBase, wtOff);

    // ---- Loop nest (explicit blocks; scf already lowered upstream) ----
    //   for k in [0, K):
    //     ki = k % 32 ; kt = k / 32
    //     whK = (ki/2)*128 + (ki%2)*2  ; voffK = baseOff64 + splat(whK)
    //     rowPtr = src + k*N*2
    //     for c in [0, N) step 64:
    //       nt = c / 32
    //       tileBase = wtBase + (nt*n_k_tiles + kt) * 2048
    //       vval = load <32xi32> rowPtr   (advance rowPtr by 128 bytes)
    //       vscattermhw(ptrtoint(tileBase), region, voffK, vval)
    Block *origBlock = rewriter.getInsertionBlock();
    Block *contBlock =
        rewriter.splitBlock(origBlock, rewriter.getInsertionPoint());

    Block *kHeader = rewriter.createBlock(contBlock, {i32Ty}, {loc});
    Block *kBody = rewriter.createBlock(contBlock);

    rewriter.setInsertionPointToEnd(origBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0}, kHeader);

    // k header
    rewriter.setInsertionPointToEnd(kHeader);
    Value k = kHeader->getArgument(0);
    Value kCond = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                                k, K);
    rewriter.create<LLVM::CondBrOp>(loc, kCond, kBody, ValueRange{}, contBlock,
                                    ValueRange{});

    // k body: compute per-k offset vector + row pointer, then enter c-loop.
    rewriter.setInsertionPointToEnd(kBody);
    Value kt = rewriter.create<LLVM::SDivOp>(loc, k, c32);
    Value ki = rewriter.create<LLVM::SRemOp>(loc, k, c32);
    Value kiHi = rewriter.create<LLVM::SDivOp>(loc, ki, c2);
    Value kiLo = rewriter.create<LLVM::SRemOp>(loc, ki, c2);
    Value whK0 = rewriter.create<LLVM::MulOp>(loc, kiHi, c128);
    Value whK1 = rewriter.create<LLVM::MulOp>(loc, kiLo, c2);
    Value whK = rewriter.create<LLVM::AddOp>(loc, whK0, whK1);
    // splat(whK) into <64xi32>
    Value whUndef = rewriter.create<LLVM::UndefOp>(loc, v64i32);
    Value whIns = rewriter.create<LLVM::InsertElementOp>(loc, v64i32, whUndef,
                                                        whK, c0);
    SmallVector<int32_t, 64> zeros64(64, 0);
    Value whSplat = rewriter.create<LLVM::ShuffleVectorOp>(
        loc, whIns, whUndef, ArrayRef<int32_t>(zeros64));
    Value voffK = rewriter.create<LLVM::AddOp>(loc, baseOff64, whSplat);
    // rowPtr = src + k * N * 2 bytes  (fp16)
    Value kN = rewriter.create<LLVM::MulOp>(loc, k, N);
    Value kNBytes = rewriter.create<LLVM::MulOp>(loc, kN, c2);
    Value rowPtr = gepI8(srcPtr, kNBytes);
    // kt * 2048 (byte contribution of the k-tile to the tile base, constant
    // across the c loop): tileBase = wtBase + (nt*n_k_tiles + kt)*2048
    //                              = wtBase + kt*2048 + nt*pairStride/...
    // Note tile(nt,kt) and tile(nt+1,kt) differ by pairStride; tile index along
    // nt advances by n_k_tiles tiles == pairStride bytes. Per c (+=64 cols ==
    // 2 N-tiles) we advance the tile base by 2*pairStride and the row pointer by
    // 128 bytes.
    Value ktBytes = rewriter.create<LLVM::MulOp>(loc, kt, tileBytes);
    Value rowTileBase = gepI8(wtBase, ktBytes);

    // ---- c loop ----  carries (c, rowPtr, tileBasePtr)
    Block *cHeader =
        rewriter.createBlock(contBlock, {i32Ty, ptrTy, ptrTy}, {loc, loc, loc});
    Block *cBody = rewriter.createBlock(contBlock);
    Block *kLatch = rewriter.createBlock(contBlock);

    rewriter.setInsertionPointToEnd(kBody);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0, rowPtr, rowTileBase},
                               cHeader);

    // c header
    rewriter.setInsertionPointToEnd(cHeader);
    Value c = cHeader->getArgument(0);
    Value curRowPtr = cHeader->getArgument(1);
    Value curTilePtr = cHeader->getArgument(2);
    Value cCond = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                                c, N);
    rewriter.create<LLVM::CondBrOp>(loc, cCond, cBody, ValueRange{}, kLatch,
                                    ValueRange{});

    // k latch: k++ ; back to k header
    rewriter.setInsertionPointToEnd(kLatch);
    Value kNext = rewriter.create<LLVM::AddOp>(loc, k, c1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{kNext}, kHeader);

    // c body: load row chunk, scatter, advance pointers.
    rewriter.setInsertionPointToEnd(cBody);
    // vval = load <32xi32> from curRowPtr (unaligned; align 2 == fp16 vmemu).
    auto loadOp =
        rewriter.create<LLVM::LoadOp>(loc, v32i32, curRowPtr, /*alignment=*/2);
    Value vval = loadOp.getResult();
    Value tileAddrI32 =
        rewriter.create<LLVM::PtrToIntOp>(loc, i32Ty, curTilePtr);
    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, *scatFn,
        ValueRange{tileAddrI32, region, voffK, vval});
    // advance: rowPtr += 128 bytes (64 fp16) ; tileBase += 2*pairStride bytes
    Value rowNext = gepI8(curRowPtr, c128);
    Value twoPair = rewriter.create<LLVM::MulOp>(loc, pairStride, c2);
    Value tileNext = gepI8(curTilePtr, twoPair);
    Value cNext = rewriter.create<LLVM::AddOp>(loc, c, c64);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{cNext, rowNext, tileNext},
                               cHeader);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexkl::HvxLayoutActivationOp -> inline HVX vscattermhw activation-half
//===----------------------------------------------------------------------===//
//
// Structural clone of LowerHvxLayoutWeight, applied along the K axis instead of
// the N axis. Reads the activation in [M][K] row-major (DDR or VTCM) directly —
// folding the vendor copy_submatrix_to_f16 — and scatters each fp16 to its
// activation-half byte offset, replacing the vendor copy_submatrix + rm_to_ah
// per-tile loop.
//
// Activation-half within-tile byte offset (empirically derived, byte-identical
// to copy_submatrix + rm_to_ah):
//     off(r,c) = (r/2)*128 + (r%2)*2 + c*4   for r,c in 0..31
//   (identical *form* to the weight-half map, but here r is the in-tile ROW and
//    c the in-tile column of a flat 32x32 fp16 tile).
// Tile (mt,kt) = (mt*n_k_tiles + kt) at 2048 bytes each.
//
// Each DDR row (global row R = mt*32 + r) is contiguous in K, so we load 64
// contiguous fp16 (cols of two adjacent k-tiles kt,kt+1) per vscattermhw and
// scatter them to the two adjacent activation-half tiles — mirroring the weight
// op's N-tile pair scatter. K must be a multiple of 64.

struct LowerHvxLayoutActivation
    : public ConvertOpToLLVMPattern<hexkl::HvxLayoutActivationOp> {
  using ConvertOpToLLVMPattern<
      hexkl::HvxLayoutActivationOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(hexkl::HvxLayoutActivationOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    auto i32Ty = rewriter.getI32Type();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i8Ty = rewriter.getI8Type();
    auto v64i32 = VectorType::get({64}, i32Ty);
    auto v32i32 = VectorType::get({32}, i32Ty);

    // Tag the enclosing llvm.func with HVX features (merge, preserve siblings).
    if (auto enclosing = op->getParentOfType<LLVM::LLVMFuncOp>()) {
      llvm::SmallVector<StringRef> feats;
      if (auto existing = enclosing.getTargetFeaturesAttr())
        for (StringRef f : existing.getFeatures())
          feats.push_back(f);
      for (StringRef want : {StringRef("+hvxv73"), StringRef("+hvx-length128b")})
        if (!llvm::is_contained(feats, want))
          feats.push_back(want);
      enclosing.setTargetFeaturesAttr(
          LLVM::TargetFeaturesAttr::get(ctx, feats));
    }

    auto scatFn = declareHmxIntrinsic(module, rewriter, kHvxScatterMhw,
                                      {i32Ty, i32Ty, v64i32, v32i32}, voidTy);
    if (failed(scatFn))
      return failure();

    auto cst = [&](int64_t v) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                               rewriter.getI32IntegerAttr(v));
    };
    auto gepI8 = [&](Value base, Value byteOff) -> Value {
      return rewriter.create<LLVM::GEPOp>(loc, ptrTy, i8Ty, base,
                                          ValueRange{byteOff});
    };
    auto splat64 = [&](Value s) -> Value {
      Value undef = rewriter.create<LLVM::UndefOp>(loc, v64i32);
      Value c0 = cst(0);
      Value ins =
          rewriter.create<LLVM::InsertElementOp>(loc, v64i32, undef, s, c0);
      SmallVector<int32_t, 64> zeros(64, 0);
      return rewriter.create<LLVM::ShuffleVectorOp>(loc, ins, undef,
                                                    ArrayRef<int32_t>(zeros));
    };

    // colOff[lane] = (col%32)*4 ; upper[lane] = col/32 (which tile of the pair),
    // where the deinterleaved lane m -> fp16 2m, 32+m -> fp16 2m+1.
    SmallVector<int32_t, 64> colOffVals(64), upperVals(64);
    for (int m = 0; m < 32; ++m) {
      int ce = 2 * m, co = 2 * m + 1;
      colOffVals[m] = (ce % 32) * 4;
      upperVals[m] = (ce / 32);
      colOffVals[32 + m] = (co % 32) * 4;
      upperVals[32 + m] = (co / 32);
    }
    Value colOff = rewriter.create<LLVM::ConstantOp>(
        loc, v64i32,
        DenseElementsAttr::get(v64i32, ArrayRef<int32_t>(colOffVals)));
    Value upperLane = rewriter.create<LLVM::ConstantOp>(
        loc, v64i32,
        DenseElementsAttr::get(v64i32, ArrayRef<int32_t>(upperVals)));

    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value srcPtr = alignedPtr(rewriter, loc, adaptor.getSrc());
    Value actOff = adaptor.getActOff();
    Value M = adaptor.getM();
    Value K = adaptor.getK();

    Value c0 = cst(0);
    Value c1 = cst(1);
    Value c2 = cst(2);
    Value c32 = cst(32);
    Value c64 = cst(64);
    Value c128 = cst(128);
    Value tileBytes = cst(kHmxTileBytes); // 2048
    // n_k_tiles = K / 32
    Value nKTiles = rewriter.create<LLVM::SDivOp>(loc, K, c32);
    // For activation, adjacent k-tiles within a row are 2048 bytes apart, so the
    // "upper" tile of a scatter pair is +2048 (one tile). region spans 2 tiles.
    Value upperOff =
        rewriter.create<LLVM::MulOp>(loc, upperLane, splat64(tileBytes));
    Value baseOff64 = rewriter.create<LLVM::AddOp>(loc, colOff, upperOff);
    // region = 2*2048 - 1 (a scatter pair touches two adjacent tiles).
    Value twoTiles = rewriter.create<LLVM::MulOp>(loc, tileBytes, c2);
    Value region = rewriter.create<LLVM::SubOp>(loc, twoTiles, c1);

    // VTCM dst base for the activation region.
    Value actBase = gepI8(vtcmBase, actOff);

    // for R in [0, M):              (global activation row)
    //   mt = R / 32 ; r = R % 32
    //   whR = (r/2)*128 + (r%2)*2 ; voffR = baseOff64 + splat(whR)
    //   rowPtr = src + R*K*2
    //   mtTileBase = actBase + mt*n_k_tiles*2048     (tile (mt,0))
    //   for kc in [0, K) step 64:   (kt = kc/32, scatters tiles kt,kt+1)
    //     tileBase = mtTileBase + kt*2048
    //     vval = load <32xi32> rowPtr (advance 128 bytes)
    //     vscattermhw(ptrtoint(tileBase), region, voffR, vval)
    Block *origBlock = rewriter.getInsertionBlock();
    Block *contBlock =
        rewriter.splitBlock(origBlock, rewriter.getInsertionPoint());

    Block *rHeader = rewriter.createBlock(contBlock, {i32Ty}, {loc});
    Block *rBody = rewriter.createBlock(contBlock);

    rewriter.setInsertionPointToEnd(origBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0}, rHeader);

    rewriter.setInsertionPointToEnd(rHeader);
    Value R = rHeader->getArgument(0);
    Value rCond =
        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, R, M);
    rewriter.create<LLVM::CondBrOp>(loc, rCond, rBody, ValueRange{}, contBlock,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(rBody);
    Value mt = rewriter.create<LLVM::SDivOp>(loc, R, c32);
    Value r = rewriter.create<LLVM::SRemOp>(loc, R, c32);
    Value rHi = rewriter.create<LLVM::SDivOp>(loc, r, c2);
    Value rLo = rewriter.create<LLVM::SRemOp>(loc, r, c2);
    Value whR0 = rewriter.create<LLVM::MulOp>(loc, rHi, c128);
    Value whR1 = rewriter.create<LLVM::MulOp>(loc, rLo, c2);
    Value whR = rewriter.create<LLVM::AddOp>(loc, whR0, whR1);
    Value voffR = rewriter.create<LLVM::AddOp>(loc, baseOff64, splat64(whR));
    // rowPtr = src + R*K*2 bytes
    Value RK = rewriter.create<LLVM::MulOp>(loc, R, K);
    Value RKBytes = rewriter.create<LLVM::MulOp>(loc, RK, c2);
    Value rowPtr = gepI8(srcPtr, RKBytes);
    // mtTileBase = actBase + mt*n_k_tiles*2048
    Value mtTiles = rewriter.create<LLVM::MulOp>(loc, mt, nKTiles);
    Value mtBytes = rewriter.create<LLVM::MulOp>(loc, mtTiles, tileBytes);
    Value mtTileBase = gepI8(actBase, mtBytes);

    Block *kHeader =
        rewriter.createBlock(contBlock, {i32Ty, ptrTy, ptrTy}, {loc, loc, loc});
    Block *kBody = rewriter.createBlock(contBlock);
    Block *rLatch = rewriter.createBlock(contBlock);

    rewriter.setInsertionPointToEnd(rBody);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0, rowPtr, mtTileBase},
                               kHeader);

    rewriter.setInsertionPointToEnd(kHeader);
    Value kc = kHeader->getArgument(0);
    Value curRowPtr = kHeader->getArgument(1);
    Value curTilePtr = kHeader->getArgument(2);
    Value kCond =
        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, kc, K);
    rewriter.create<LLVM::CondBrOp>(loc, kCond, kBody, ValueRange{}, rLatch,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(rLatch);
    Value RNext = rewriter.create<LLVM::AddOp>(loc, R, c1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{RNext}, rHeader);

    rewriter.setInsertionPointToEnd(kBody);
    auto loadOp =
        rewriter.create<LLVM::LoadOp>(loc, v32i32, curRowPtr, /*alignment=*/2);
    Value vval = loadOp.getResult();
    Value tileAddrI32 =
        rewriter.create<LLVM::PtrToIntOp>(loc, i32Ty, curTilePtr);
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, *scatFn,
                                  ValueRange{tileAddrI32, region, voffR, vval});
    // advance: rowPtr += 128 bytes (64 fp16) ; tileBase += 2*2048 (2 k-tiles)
    Value rowNext = gepI8(curRowPtr, c128);
    Value tileNext = gepI8(curTilePtr, twoTiles);
    Value kcNext = rewriter.create<LLVM::AddOp>(loc, kc, c64);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{kcNext, rowNext, tileNext},
                               kHeader);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexkl::HvxReadoutOp -> inline HVX vgathermhw + vcvt (ah->rm + f16->f32)
//===----------------------------------------------------------------------===//
//
// Reads activation-half accumulator tiles, gathers each into flat row-major
// fp16 via V6.vgathermhw (the inverse of the activation-half scatter, using the
// identical off(r,c) map), then widens fp16->fp32 via V6.vcvt_sf_hf and stores
// row-major to DDR [M][N]. Replaces vendor ah_to_rm + copy_f16_to_f32.
//
// vgathermhw(Rs, Rt, Mu, Voff): gathers 64 halfwords from base Rt (region Mu+1)
// at the <64xi32> byte offsets Voff into the VTCM vector at Rs, in the SAME
// deinterleaved lane order the scatter uses. We gather a full 32x32 tile (1024
// fp16 = 16 gathers of 64) into a flat row-major scratch tile, then widen+store.
//
// vcvt_sf_hf: <64 x f16> -> {<32 x f32>, <32 x f32>} (a vector PAIR, lo=lanes
// 0..31, hi=lanes 32..63).

static constexpr StringRef kHvxGatherMhw = "llvm.hexagon.V6.vgathermhw.128B";

struct LowerHvxReadout
    : public ConvertOpToLLVMPattern<hexkl::HvxReadoutOp> {
  using ConvertOpToLLVMPattern<hexkl::HvxReadoutOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(hexkl::HvxReadoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    auto i32Ty = rewriter.getI32Type();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i8Ty = rewriter.getI8Type();
    auto f16Ty = rewriter.getF16Type();
    auto f32Ty = rewriter.getF32Type();
    auto v64i32 = VectorType::get({64}, i32Ty);
    auto v32i32 = VectorType::get({32}, i32Ty); // = <64xf16> bit-reinterpret
    auto v32f32 = VectorType::get({32}, f32Ty);

    // Tag the enclosing llvm.func with HVX features (merge, preserve siblings).
    if (auto enclosing = op->getParentOfType<LLVM::LLVMFuncOp>()) {
      llvm::SmallVector<StringRef> feats;
      if (auto existing = enclosing.getTargetFeaturesAttr())
        for (StringRef f : existing.getFeatures())
          feats.push_back(f);
      for (StringRef want : {StringRef("+hvxv73"), StringRef("+hvx-length128b")})
        if (!llvm::is_contained(feats, want))
          feats.push_back(want);
      enclosing.setTargetFeaturesAttr(
          LLVM::TargetFeaturesAttr::get(ctx, feats));
    }

    // vgathermhw(ptr Rs, i32 Rt, i32 Mu, <64xi32> off) -> void
    auto gatherFn = declareHmxIntrinsic(module, rewriter, kHvxGatherMhw,
                                        {ptrTy, i32Ty, i32Ty, v64i32}, voidTy);
    if (failed(gatherFn))
      return failure();
    (void)v32i32;

    auto cst = [&](int64_t v) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                               rewriter.getI32IntegerAttr(v));
    };
    auto gepI8 = [&](Value base, Value byteOff) -> Value {
      return rewriter.create<LLVM::GEPOp>(loc, ptrTy, i8Ty, base,
                                          ValueRange{byteOff});
    };

    // Build the gather offset table. We gather 64 fp16 per call. Lay them out so
    // the gathered scratch is flat row-major: gather group g (0..15) covers the
    // 64 fp16 of rows (2g, 2g+1) — these occupy the contiguous 128-byte AH block
    // at base (g*128). The deinterleaved lane m -> output fp16 2m, 32+m -> 2m+1.
    // We want scratch[ (2g)*32 .. ] = row 2g cols0..31, then row 2g+1.
    // So for this group, output position p (0..63): row = 2g + p/32, col = p%32,
    // and AH source offset = off(row_in_tile, col) where row_in_tile = (p/32 for
    // the 2-row sub-block)*? Simpler: gather offsets within the 128-byte block.
    //   Within an AH block (rows 2g,2g+1): off(2g+sub, c) - off(2g,0)
    //     = (sub)*2 + c*4   (sub in {0,1}, c in 0..31), since (r/2) constant.
    // For deinterleaved lane order: lane m (even halfword) -> fp16 2m; we want
    // gathered output to be row-major [row0 c0..c31, row1 c0..c31]. Map output
    // position p -> (sub=p/32, c=p%32). Even halfword fp16 index 2m, odd 2m+1.
    // So output even positions {0,2,..} hold fp16 2m at lane m, etc. We instead
    // set Voff[lane] to the AH offset of the desired output element so that the
    // gather lands row-major. Output fp16 index q: lane(q) = (q%2==0)? q/2 :
    // 32 + q/2. desired element at output q = (sub=q/32, c=q%32) -> ah within
    // block = sub*2 + c*4.
    SmallVector<int32_t, 64> offVals(64);
    for (int q = 0; q < 64; ++q) {
      int sub = q / 32, c = q % 32;
      int ahWithinBlock = sub * 2 + c * 4;
      int lane = (q % 2 == 0) ? (q / 2) : (32 + q / 2);
      offVals[lane] = ahWithinBlock;
    }
    Value offBase = rewriter.create<LLVM::ConstantOp>(
        loc, v64i32, DenseElementsAttr::get(v64i32, ArrayRef<int32_t>(offVals)));

    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value dstPtr = alignedPtr(rewriter, loc, adaptor.getDst());
    Value outOff = adaptor.getOutOff();
    Value scratchOff = adaptor.getScratchOff();
    Value M = adaptor.getM();
    Value N = adaptor.getN();

    Value c0 = cst(0);
    Value c1 = cst(1);
    Value c2 = cst(2);
    Value c4 = cst(4);
    Value c32 = cst(32);
    Value c128 = cst(128);
    Value c2047 = cst(2047);
    Value tileBytes = cst(kHmxTileBytes);
    Value scratchPtr = gepI8(vtcmBase, scratchOff);
    Value outBase = gepI8(vtcmBase, outOff);
    // n_n_tiles = N / 32
    Value nNTiles = rewriter.create<LLVM::SDivOp>(loc, N, c32);
    // n_m_tiles = M / 32
    Value nMTiles = rewriter.create<LLVM::SDivOp>(loc, M, c32);

    // Loop over output tiles (mt, nt). For each: gather its AH tile (16 groups)
    // into scratch flat row-major, then widen+store to dst row-major.
    Block *origBlock = rewriter.getInsertionBlock();
    Block *contBlock =
        rewriter.splitBlock(origBlock, rewriter.getInsertionPoint());

    // ---- mt loop ----
    Block *mtHeader = rewriter.createBlock(contBlock, {i32Ty}, {loc});
    Block *mtBody = rewriter.createBlock(contBlock);
    rewriter.setInsertionPointToEnd(origBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0}, mtHeader);
    rewriter.setInsertionPointToEnd(mtHeader);
    Value mt = mtHeader->getArgument(0);
    Value mtCond =
        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, mt, nMTiles);
    rewriter.create<LLVM::CondBrOp>(loc, mtCond, mtBody, ValueRange{}, contBlock,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(mtBody);
    // ---- nt loop ----
    Block *ntHeader = rewriter.createBlock(contBlock, {i32Ty}, {loc});
    Block *ntBody = rewriter.createBlock(contBlock);
    Block *ntLatch = rewriter.createBlock(contBlock);
    Block *mtLatch = rewriter.createBlock(contBlock);
    rewriter.setInsertionPointToEnd(mtBody);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0}, ntHeader);
    rewriter.setInsertionPointToEnd(ntHeader);
    Value nt = ntHeader->getArgument(0);
    Value ntCond =
        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, nt, nNTiles);
    rewriter.create<LLVM::CondBrOp>(loc, ntCond, ntBody, ValueRange{}, mtLatch,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(mtLatch);
    Value mtNext = rewriter.create<LLVM::AddOp>(loc, mt, c1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{mtNext}, mtHeader);

    rewriter.setInsertionPointToEnd(ntLatch);
    Value ntNext = rewriter.create<LLVM::AddOp>(loc, nt, c1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{ntNext}, ntHeader);

    rewriter.setInsertionPointToEnd(ntBody);
    // tile (mt,nt) byte base = outBase + (mt*n_n_tiles + nt)*2048
    Value tileIdx = rewriter.create<LLVM::MulOp>(loc, mt, nNTiles);
    tileIdx = rewriter.create<LLVM::AddOp>(loc, tileIdx, nt);
    Value tileByte = rewriter.create<LLVM::MulOp>(loc, tileIdx, tileBytes);
    Value ahTilePtr = gepI8(outBase, tileByte);
    Value ahTileAddrI32 =
        rewriter.create<LLVM::PtrToIntOp>(loc, i32Ty, ahTilePtr);

    // ---- fused per-group loop g in [0,16): gather rows (2g,2g+1) into a
    //      private scratch vector, then IMMEDIATELY load it back, widen to f32,
    //      and store. Fusing (vs two separate loops) gives the gather store ->
    //      load a real same-iteration data dependency so isel keeps them ordered
    //      (the HVX gather engine writes asynchronously; the consuming load must
    //      follow it — see llama.cpp hmx-utils gather pattern). ----
    //
    // Widening: the HVX qfloat vcvt.sf.hf is non-functional on this v73 part
    // (returns 0 even via the C builtin), so the f16->f32 conversion is emitted
    // as a vector fpext (<32xf16> -> <32xf32>) per row. The vgather (the
    // ah->rm tile reshuffle) is the HVX-vector part replacing the vendor
    // ah_to_rm; fpext is a plain numeric widen (compiler-emitted, no vendor
    // call), exact for f16->f32. Two rows (2w, 2w+1) per gathered 128B block.
    Value c16 = cst(16);
    auto v32f16 = VectorType::get({32}, f16Ty);
    Block *wHeader = rewriter.createBlock(contBlock, {i32Ty}, {loc});
    Block *wBody = rewriter.createBlock(contBlock);
    rewriter.setInsertionPointToEnd(ntBody);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{c0}, wHeader);
    rewriter.setInsertionPointToEnd(wHeader);
    Value w = wHeader->getArgument(0); // group of 2 rows, 0..15
    Value wCond =
        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, w, c16);
    rewriter.create<LLVM::CondBrOp>(loc, wCond, wBody, ValueRange{}, ntLatch,
                                    ValueRange{});
    rewriter.setInsertionPointToEnd(wBody);
    Value wBytes = rewriter.create<LLVM::MulOp>(loc, w, c128);
    // gather: off_w = offBase + splat(w*128); Rt = ahTile base; Mu=2047.
    Value wSplat;
    {
      Value undef = rewriter.create<LLVM::UndefOp>(loc, v64i32);
      Value ins =
          rewriter.create<LLVM::InsertElementOp>(loc, v64i32, undef, wBytes, c0);
      SmallVector<int32_t, 64> zeros(64, 0);
      wSplat = rewriter.create<LLVM::ShuffleVectorOp>(loc, ins, undef,
                                                      ArrayRef<int32_t>(zeros));
    }
    Value offW = rewriter.create<LLVM::AddOp>(loc, offBase, wSplat);
    Value scratchWPtr = gepI8(scratchPtr, wBytes);
    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, *gatherFn,
        ValueRange{scratchWPtr, ahTileAddrI32, c2047, offW});
    // dst row index for the two rows: row0 = mt*32 + 2w, row1 = row0+1.
    Value mtRow = rewriter.create<LLVM::MulOp>(loc, mt, c32);
    Value twoW = rewriter.create<LLVM::MulOp>(loc, w, c2);
    Value row0 = rewriter.create<LLVM::AddOp>(loc, mtRow, twoW);
    Value row1 = rewriter.create<LLVM::AddOp>(loc, row0, c1);
    Value ntCol = rewriter.create<LLVM::MulOp>(loc, nt, c32);
    auto dstRowPtr = [&](Value row) -> Value {
      Value rn = rewriter.create<LLVM::MulOp>(loc, row, N);
      Value idx = rewriter.create<LLVM::AddOp>(loc, rn, ntCol);
      Value byte = rewriter.create<LLVM::MulOp>(loc, idx, c4);
      return gepI8(dstPtr, byte);
    };
    // load the just-gathered 2 rows (row r at scratch+r*64), widen, store.
    Value c64b = cst(64);
    // row0: scratch+w*128 ; row1: scratch+w*128+64
    Value sRow0 = scratchWPtr;
    Value sRow1 = gepI8(scratchWPtr, c64b);
    auto widenStore = [&](Value sptr, Value drow) {
      auto ld = rewriter.create<LLVM::LoadOp>(loc, v32f16, sptr,
                                              /*alignment=*/2);
      Value f32v =
          rewriter.create<LLVM::FPExtOp>(loc, v32f32, ld.getResult());
      rewriter.create<LLVM::StoreOp>(loc, f32v, dstRowPtr(drow),
                                     /*alignment=*/4);
    };
    widenStore(sRow0, row0);
    widenStore(sRow1, row1);
    Value wNext = rewriter.create<LLVM::AddOp>(loc, w, c1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{wNext}, wHeader);

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
  // Lower the fast HVX weight-half layout to inline vscatter.
  patterns.add<LowerHvxLayoutWeight>(converter);
  // Lower the fast HVX activation-half layout + accumulator readout.
  patterns.add<LowerHvxLayoutActivation, LowerHvxReadout>(converter);
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
