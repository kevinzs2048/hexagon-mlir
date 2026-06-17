//===-- DecomposeHexKLMatmulPass.cpp - Decompose hexkl.matmul to micro ops ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to decompose hexkl::MatmulOp into hexkl micro HMX operations.
// This pass implements the tiling strategy from hexkl_matmul_f16f16_f32.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "decompose-hexkl-matmul"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_DECOMPOSEHEXKLMATMUL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct DecomposeHexKLMatmul final : public OpRewritePattern<hexkl::MatmulOp> {
  DecomposeHexKLMatmul(MLIRContext *ctx, bool useInlineHMX, bool enableQ8Weight)
      : OpRewritePattern(ctx), useInlineHMX(useInlineHMX),
        enableQ8Weight(enableQ8Weight) {}

  bool useInlineHMX = false;
  bool enableQ8Weight = false;

  LogicalResult matchAndRewrite(hexkl::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value result = op.getOuts();

    auto lhsType = cast<MemRefType>(lhs.getType());
    auto rhsType = cast<MemRefType>(rhs.getType());
    auto resultType = cast<MemRefType>(result.getType());

    // Validate rank (must be 2D)
    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    if (lhsShape.size() != 2 || rhsShape.size() != 2 ||
        resultShape.size() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matmul supported");
    }

    // Validate static shape compatibility if available
    if (lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
        resultType.hasStaticShape()) {
      int64_t M = lhsShape[0];
      int64_t K_lhs = lhsShape[1];
      int64_t K_rhs = rhsShape[0];
      int64_t N = rhsShape[1];
      int64_t M_out = resultShape[0];
      int64_t N_out = resultShape[1];

      // Validate dimensions match: lhs(M×K) × rhs(K×N) = result(M×N)
      if (K_lhs != K_rhs) {
        return rewriter.notifyMatchFailure(
            op, "inner dimensions mismatch: lhs K != rhs K");
      }
      if (M != M_out || N != N_out) {
        return rewriter.notifyMatchFailure(op, "output dimensions mismatch");
      }
    }

    // Create constants
    auto i32Ty = rewriter.getI32Type();

    Value idx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value idx32 = rewriter.create<arith::ConstantIndexOp>(loc, 32);
    Value idx31 = rewriter.create<arith::ConstantIndexOp>(loc, 31);

    Value i32_0 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 0);
    Value i32_1 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 1);
    Value i32_32 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 32);
    Value i32_4096 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 4096);
    Value idx4096 = rewriter.create<arith::ConstantIndexOp>(loc, 4096);

    // Get dimensions dynamically
    Value dimM = rewriter.create<memref::DimOp>(loc, lhs, idx0);
    Value dimK = rewriter.create<memref::DimOp>(loc, lhs, idx1);
    Value dimN = rewriter.create<memref::DimOp>(loc, rhs, idx1);

    Value M = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimM);
    Value K = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimK);
    Value N = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimN);

    // Calculate numKTiles = (k + 31) / 32
    Value kPlus31 = rewriter.create<arith::AddIOp>(loc, dimK, idx31);
    Value kTiles = rewriter.create<arith::DivUIOp>(loc, kPlus31, idx32);
    Value kTilesI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, kTiles);

    // ----------------------------------------------------------------------
    // Inline-HMX path (mirrors the proven hmx-spike pipeline). When enabled,
    // replace the per-output-tile micro compute with ONE hexkl.hmx_core_dot
    // over an all-resident VTCM arena. Layout + readout reuse the existing
    // micro ops. Default-off path (below) is left byte-identical.
    // ----------------------------------------------------------------------
    if (useInlineHMX) {
      if (failed(rewriteInlineHMX(op, rewriter, loc, lhs, rhs, result, i32Ty,
                                  idx0, idx1, idx32, dimM, dimN, M, K, N, kTiles,
                                  kTilesI32)))
        return failure();
      return success();
    }

    // Calculate VTCM size: (numKTiles*2 + 2 + 3) * 4096
    // Layout: [act_tiles | scratch_tiles | flat_out | acc_read | extra]
    Value twoKTiles = rewriter.create<arith::MulIOp>(
        loc, kTiles, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value dataTiles = rewriter.create<arith::AddIOp>(
        loc, twoKTiles, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value vtcmTiles = rewriter.create<arith::AddIOp>(
        loc, dataTiles, rewriter.create<arith::ConstantIndexOp>(loc, 3));
    Value vtcmBytes = rewriter.create<arith::MulIOp>(loc, vtcmTiles, idx4096);

    // Manual VTCM alloc; tag for bufferization and dealloc after outer loop.
    auto vtcmType =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                        MemRefLayoutAttrInterface{},
                        IntegerAttr::get(rewriter.getI32Type(), 1));
    auto vtcmAlloc =
        rewriter.create<hexagonmem::AllocOp>(loc, vtcmType, vtcmBytes);
    vtcmAlloc->setAttr("bufferization.manual_deallocation",
                       rewriter.getUnitAttr());
    Value vtcm = vtcmAlloc.getResult();

    // Setup HMX accumulator
    rewriter.create<hexkl::MicroHMXSetupAccReadF16Op>(loc, vtcm);

    // Hoist loop-invariant offset calculations
    // weightOffset = numKTiles * 4096 (weight buffer starts after activation
    // tiles)
    Value wOff = rewriter.create<arith::MulIOp>(loc, kTilesI32, i32_4096);

    // flatOffset = numKTiles * 4096 (flat output buffer location)
    Value flatOff = wOff;

    // accReadOffset = (numKTiles + 1) * 4096 (accumulator readback location)
    Value kTilesPlus1 = rewriter.create<arith::AddIOp>(loc, kTilesI32, i32_1);
    Value accOff = rewriter.create<arith::MulIOp>(loc, kTilesPlus1, i32_4096);

    // Outer loop: iterate over rows (M dimension) in 32-row tiles
    auto outerFor = rewriter.create<scf::ForOp>(
        loc, idx0, dimM, idx32, ValueRange{},
        [&](OpBuilder &b, Location loc, Value row, ValueRange) {
          Value rowI32 = b.create<arith::IndexCastOp>(loc, i32Ty, row);
          Value rowTile = b.create<arith::DivUIOp>(loc, rowI32, i32_32);

          // Load and layout activation tiles for this row
          b.create<scf::ForOp>(
              loc, idx0, kTiles, idx1, ValueRange{},
              [&](OpBuilder &bb, Location loc, Value ktIdx, ValueRange) {
                Value kt = bb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);

                // scratchOffset = (numKTiles + kTile) * 4096
                Value scrIdx = bb.create<arith::AddIOp>(loc, kTilesI32, kt);
                Value scrOff = bb.create<arith::MulIOp>(loc, scrIdx, i32_4096);

                // Copy activation tile to scratch
                bb.create<hexkl::MicroHMXCopySubmatrixToF16Op>(
                    loc, vtcm, scrOff, lhs, rowTile, kt, M, K);

                // Layout activation from scratch to activation buffer
                Value actOff = bb.create<arith::MulIOp>(loc, kt, i32_4096);
                bb.create<hexkl::MicroHMXRmToAhF16Op>(loc, vtcm, actOff,
                                                      scrOff);

                bb.create<scf::YieldOp>(loc);
              });

          // Inner loop: iterate over columns (N dimension) in 32-col tiles
          b.create<scf::ForOp>(
              loc, idx0, dimN, idx32, ValueRange{},
              [&](OpBuilder &bb, Location loc, Value col, ValueRange) {
                Value colI32 = bb.create<arith::IndexCastOp>(loc, i32Ty, col);
                Value colTile = bb.create<arith::DivUIOp>(loc, colI32, i32_32);

                // Clear accumulator
                bb.create<hexkl::MicroHMXAccClearF16Op>(loc);

                // Innermost loop: iterate over K tiles for accumulation
                bb.create<scf::ForOp>(
                    loc, idx0, kTiles, idx1, ValueRange{},
                    [&](OpBuilder &bbb, Location loc, Value ktIdx, ValueRange) {
                      Value kt =
                          bbb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);

                      // Load weight tile
                      bbb.create<hexkl::MicroHMXRmToWhF16Op>(
                          loc, vtcm, wOff, rhs, kt, colTile, N);

                      // Perform matrix multiplication (accumulates)
                      Value actOff2 =
                          bbb.create<arith::MulIOp>(loc, kt, i32_4096);
                      bbb.create<hexkl::MicroHMXMmF16Op>(loc, vtcm, actOff2,
                                                         wOff);

                      bbb.create<scf::YieldOp>(loc);
                    });

                // Read accumulator
                bb.create<hexkl::MicroHMXAccReadF16Op>(loc, vtcm, accOff);

                // Convert from activation layout to flat layout
                bb.create<hexkl::MicroHMXAhToRmF16Op>(loc, vtcm, flatOff,
                                                      accOff);

                // Copy result to output (f16 -> f32)
                bb.create<hexkl::MicroHMXCopyF16ToF32SubmatrixOp>(
                    loc, vtcm, flatOff, result, rowTile, colTile, M, N);

                bb.create<scf::YieldOp>(loc);
              });

          b.create<scf::YieldOp>(loc);
        });

    // Explicitly deallocate VTCM buffer to avoid relying on ConvertToHexagonmem
    // rewriting of generic memref.dealloc for dynamic VTCM types.
    rewriter.setInsertionPointAfter(outerFor);
    rewriter.create<hexagonmem::DeallocOp>(loc, vtcm);

    rewriter.eraseOp(op);
    return success();
  }

  // Find the Q8 weight buffer for the enableQ8Weight path: the i8-element
  // ranked memref argument of the enclosing func. The fp16 matmul rhs only
  // carries the K/N dims; the actual quantized weight data is passed as a
  // separate i8 [N][row_stride] kernel argument (the matmul op type system
  // requires a fp16 rhs, so the Q8 buffer cannot be the rhs itself).
  static Value findQ8WeightArg(hexkl::MatmulOp op) {
    auto func = op->getParentOfType<FunctionOpInterface>();
    if (!func)
      return nullptr;
    Value found;
    for (Value arg : func.getArguments()) {
      auto mt = dyn_cast<MemRefType>(arg.getType());
      if (!mt)
        continue;
      // The Q8 weight is the DDR (memory space 0) i8 memref arg. In the split
      // (HEXKL_SPLIT_DEQUANT_M8) path the func ALSO carries an external VTCM i8
      // arg (rank-1, memory space 1) -- skip it so the Q8 lookup stays
      // unambiguous.
      if (mt.getElementType().isInteger(8) &&
          mt.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE) {
        if (found) // ambiguous: more than one DDR i8 memref arg
          return nullptr;
        found = arg;
      }
    }
    return found;
  }

  // Split (HEXKL_SPLIT_DEQUANT_M8) path: locate the trailing scalar `index`
  // runtime arg carrying the wt-tile base byte offset into the shared VTCM
  // arena. When present, the split halves use it as wtOff instead of the
  // compile-derived act_off+act_bytes, so the shim can double-buffer the weight
  // region. Returns nullptr if the func has no scalar index arg (legacy
  // single-buffer kernels) so callers fall back to the computed offset.
  static Value findWtOffArg(hexkl::MatmulOp op) {
    auto func = op->getParentOfType<FunctionOpInterface>();
    if (!func)
      return nullptr;
    for (Value arg : func.getArguments())
      if (arg.getType().isIndex())
        return arg;
    return nullptr;
  }

  // Split (HEXKL_SPLIT_DEQUANT_M8) path: locate the external VTCM arena passed
  // in as a func argument -- a rank-1 i8 memref in VTCM address space (memory
  // space 1). Both split funcs (compiler_dequant / compiler_m8) take this as
  // their first arg so they share the same arena base + offset formula.
  static Value findExternalVtcmArg(hexkl::MatmulOp op) {
    auto func = op->getParentOfType<FunctionOpInterface>();
    if (!func)
      return nullptr;
    for (Value arg : func.getArguments()) {
      auto mt = dyn_cast<MemRefType>(arg.getType());
      if (!mt)
        continue;
      if (mt.hasRank() && mt.getRank() == 1 &&
          mt.getElementType().isInteger(8) &&
          mt.getMemorySpaceAsInt() == hexagon::VTCM_ADDRESS_SPACE)
        return arg;
    }
    return nullptr;
  }

  // Inline-HMX decomposition: all-resident VTCM arena + one hexkl.hmx_core_dot.
  // Mirrors the proven hmx-spike pipeline (spike_main.cpp / spike_layout.c /
  // spike_dequant.c / spike_hmx_mm.c):
  //   1. arena holding act/wt/out/bias/scratch tiles simultaneously
  //   2. lay out ALL activation tiles -> act[mt][kt]
  //   3. lay out ALL weight tiles     -> wt[nt][kt]  (N outer, K inner)
  //   4. init bias struct (0x00003c00 per-column-scale=1.0,bias=0 in first 32
  //      i32 lanes; rest zero)  -- spike T6 finding
  //   5. ONE hexkl.hmx_core_dot(vtcm, out, act, wt, bias, m_t, n_t, k_t)
  //   6. readout per (mt,nt) -> ah_to_rm + copy_f16_to_f32 -> DDR
  LogicalResult rewriteInlineHMX(hexkl::MatmulOp op, PatternRewriter &rewriter,
                                 Location loc, Value lhs, Value rhs,
                                 Value result, Type i32Ty, Value idx0,
                                 Value idx1, Value idx32, Value dimM, Value dimN,
                                 Value M, Value K, Value N, Value kTiles,
                                 Value kTilesI32) const {
    if (::getenv("HEXKL_VRMPY")) {
      // int8-vrmpy 1-touch path: no VTCM dequant arena, no HMX. Pull the two
      // q8x4x2 i8 args (activation %Aq8 first, weight %Wq8 second) positionally
      // -- findQ8WeightArg returns nullptr with >1 i8 DDR arg.
      auto func = op->getParentOfType<FunctionOpInterface>();
      Value actI8, wtI8;
      for (Value arg : func.getArguments()) {
        auto mt = dyn_cast<MemRefType>(arg.getType());
        if (mt && mt.getElementType().isInteger(8) &&
            mt.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE) {
          if (!actI8)
            actI8 = arg;
          else if (!wtI8)
            wtI8 = arg;
        }
      }
      if (!actI8 || !wtI8)
        return failure();
      rewriter.create<hexkl::MatmulVrmpyQ8Op>(loc, /*out=*/result,
                                              /*act=*/actI8, /*wt=*/wtI8,
                                              /*k=*/K);
      rewriter.eraseOp(op);
      return success();
    }

    // Tile counts (index + i32).
    Value mTiles = rewriter.create<arith::DivUIOp>(loc, dimM, idx32);
    Value nTiles = rewriter.create<arith::DivUIOp>(loc, dimN, idx32);
    Value mTilesI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, mTiles);
    Value nTilesI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, nTiles);

    // Tile stride is 2048 bytes (32x32 fp16), matching the spike kernel's
    // contiguous deep-streaming reads (HMX_FP16_TILE_N_ELMS = 1024 elms).
    Value i32_2048 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 2048);
    Value idx2048 = rewriter.create<arith::ConstantIndexOp>(loc, 2048);

    // Region byte sizes (index).
    Value mkTiles = rewriter.create<arith::MulIOp>(loc, mTiles, kTiles);
    Value nkTiles = rewriter.create<arith::MulIOp>(loc, nTiles, kTiles);
    Value mnTiles = rewriter.create<arith::MulIOp>(loc, mTiles, nTiles);
    Value actBytes = rewriter.create<arith::MulIOp>(loc, mkTiles, idx2048);
    Value wtBytes = rewriter.create<arith::MulIOp>(loc, nkTiles, idx2048);
    Value outBytes = rewriter.create<arith::MulIOp>(loc, mnTiles, idx2048);

    // Arena offsets (i32 bytes). act|wt|out|bias|flat|scratch, all 2048-aligned.
    Value actOffI32 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 0);
    Value actBytesI32 =
        rewriter.create<arith::IndexCastOp>(loc, i32Ty, actBytes);
    Value wtBytesI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, wtBytes);
    Value outBytesI32 =
        rewriter.create<arith::IndexCastOp>(loc, i32Ty, outBytes);
    Value wtOffI32 =
        rewriter.create<arith::AddIOp>(loc, actOffI32, actBytesI32);
    Value outOffI32 = rewriter.create<arith::AddIOp>(loc, wtOffI32, wtBytesI32);
    Value biasOffI32 =
        rewriter.create<arith::AddIOp>(loc, outOffI32, outBytesI32);
    // One bias tile, then a flat readout tile, then a layout scratch tile.
    Value flatOffI32 =
        rewriter.create<arith::AddIOp>(loc, biasOffI32, i32_2048);
    Value scratchOffI32 =
        rewriter.create<arith::AddIOp>(loc, flatOffI32, i32_2048);

    // Total arena bytes = scratchOff + one tile (index).
    Value scratchOffIdx =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                            scratchOffI32);
    Value vtcmBytes =
        rewriter.create<arith::AddIOp>(loc, scratchOffIdx, idx2048);

    // ----------------------------------------------------------------------
    // SPLIT path (HEXKL_SPLIT_DEQUANT_M8, default OFF): emit ONLY half of the
    // fused 6-op pipeline depending on which func encloses this matmul, sharing
    // an EXTERNAL VTCM arena passed as a func arg (no internal Alloc/Dealloc).
    // The arena offsets above are computed identically from M/K/N in BOTH
    // funcs, so dequant's writes and M8's reads land on the same VTCM tiles.
    //   func "compiler_dequant": act layout + weight dequant (HVX, no HMX) --
    //       parallelizable across threads (weight dequant can be tile-ranged).
    //   func "compiler_m8":      HMX setup + bias init + M8 + readout (single
    //       thread; owns the HMX unit).
    // Anything else: fall through to the original single-func fused path.
    // ----------------------------------------------------------------------
    if (::getenv("HEXKL_SPLIT_DEQUANT_M8")) {
      Value extVtcm = findExternalVtcmArg(op);
      if (!extVtcm)
        return rewriter.notifyMatchFailure(
            op, "HEXKL_SPLIT_DEQUANT_M8: no external VTCM i8 memref (rank-1, "
                "space 1) argument found on the enclosing func");
      auto func = op->getParentOfType<FunctionOpInterface>();
      StringRef fname = func ? func.getName() : StringRef();
      bool isDequant = fname.contains("dequant");
      bool isM8 = fname.contains("m8") || fname.contains("M8");
      if (!isDequant && !isM8)
        return rewriter.notifyMatchFailure(
            op, "HEXKL_SPLIT_DEQUANT_M8: enclosing func name must contain "
                "'dequant' or 'm8' to select the split half");

      // Runtime wt_off: when the func carries a trailing scalar index arg, the
      // weight tiles live at a CALLER-CHOSEN base (the shim double-buffers the
      // wt region), so the fixed regions must NOT include wt in their chain --
      // otherwise out/bias/scratch would shift with the runtime wt_off. Re-lay
      // the fixed regions as act|out|bias|flat|scratch (wt excluded) and take
      // wtOff from the arg. Without the arg (legacy single-buffer kernels), keep
      // the original act|wt|out|... layout computed above.
      Value wtOffArg = findWtOffArg(op);
      if (wtOffArg) {
        wtOffI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, wtOffArg);
        outOffI32 = rewriter.create<arith::AddIOp>(loc, actOffI32, actBytesI32);
        biasOffI32 =
            rewriter.create<arith::AddIOp>(loc, outOffI32, outBytesI32);
        flatOffI32 = rewriter.create<arith::AddIOp>(loc, biasOffI32, i32_2048);
        scratchOffI32 =
            rewriter.create<arith::AddIOp>(loc, flatOffI32, i32_2048);
      }

      if (isDequant) {
        // --- dequant half: act layout + weight dequant only ---
        rewriter.create<hexkl::HvxLayoutActivationOp>(loc, extVtcm, actOffI32,
                                                      lhs, M, K);
        if (enableQ8Weight) {
          Value q8Src = findQ8WeightArg(op);
          if (!q8Src)
            return rewriter.notifyMatchFailure(
                op, "HEXKL_SPLIT_DEQUANT_M8 dequant: no DDR i8 weight memref "
                    "argument found on the enclosing func");
          rewriter.create<hexkl::HvxLayoutWeightQ8Op>(loc, extVtcm, wtOffI32,
                                                      q8Src, K, N);
        } else {
          rewriter.create<hexkl::HvxLayoutWeightOp>(loc, extVtcm, wtOffI32, rhs,
                                                    K, N);
        }
        rewriter.eraseOp(op);
        return success();
      }

      // --- M8 half: HMX setup + bias init + M8 + readout only ---
      rewriter.create<hexkl::MicroHMXSetupAccReadF16Op>(loc, extVtcm);
      {
        // Match the i32 bias-view memory space to the external VTCM arg's own
        // space attr (func args parse "1" as a plain index attr, not "1 : i32"),
        // else memref.view rejects the space mismatch.
        Attribute vtcmSpace =
            cast<MemRefType>(extVtcm.getType()).getMemorySpace();
        auto i32MemTy = MemRefType::get({512}, i32Ty,
                                        MemRefLayoutAttrInterface{}, vtcmSpace);
        Value biasOffIdx = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), biasOffI32);
        Value biasView = rewriter.create<memref::ViewOp>(
            loc, i32MemTy, extVtcm, biasOffIdx, ValueRange{});
        Value cScale =
            rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 0x00003c00);
        Value cZero = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 0);
        Value idx512 = rewriter.create<arith::ConstantIndexOp>(loc, 512);
        rewriter.create<scf::ForOp>(
            loc, idx0, idx512, idx1, ValueRange{},
            [&](OpBuilder &b, Location loc, Value i, ValueRange) {
              Value isScale = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::ult, i, idx32);
              Value v = b.create<arith::SelectOp>(loc, isScale, cScale, cZero);
              b.create<memref::StoreOp>(loc, v, biasView, ValueRange{i});
              b.create<scf::YieldOp>(loc);
            });
      }
      rewriter.create<hexkl::HmxCoreDotOp>(loc, extVtcm, outOffI32, actOffI32,
                                           wtOffI32, biasOffI32, mTilesI32,
                                           nTilesI32, kTilesI32);
      rewriter.create<hexkl::HvxReadoutOp>(loc, extVtcm, outOffI32, scratchOffI32,
                                           result, M, N);
      rewriter.eraseOp(op);
      return success();
    }

    // Manual VTCM alloc; tag for bufferization and dealloc after the readout.
    auto vtcmType =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                        MemRefLayoutAttrInterface{},
                        IntegerAttr::get(rewriter.getI32Type(), 1));
    auto vtcmAlloc =
        rewriter.create<hexagonmem::AllocOp>(loc, vtcmType, vtcmBytes);
    vtcmAlloc->setAttr("bufferization.manual_deallocation",
                       rewriter.getUnitAttr());
    Value vtcm = vtcmAlloc.getResult();

    // Setup HMX accumulator for acc-read output.
    rewriter.create<hexkl::MicroHMXSetupAccReadF16Op>(loc, vtcm);

    // --- 2. lay out ALL activation tiles: act[mt][kt] ---
    // ONE compiler-emitted HVX vscattermhw layout over the whole [M][K] DDR
    // activation, replacing the per-(mt,kt) copy_submatrix + rm_to_ah loop.
    // The op reads lhs [M][K] row-major directly (folds the submatrix copy) and
    // writes tile (mt,kt) at actOff + (mt*n_k_tiles+kt)*2048 -- the same order
    // hmx_core_dot's act loop expects. Byte-identical output, ~8x faster.
    rewriter.create<hexkl::HvxLayoutActivationOp>(loc, vtcm, actOffI32, lhs, M,
                                                  K);

    // --- 3. lay out ALL weight tiles: wt[nt][kt] (N outer, K inner) ---
    // ONE compiler-emitted HVX vscattermhw layout over the whole [K][N] DDR
    // weight, replacing the per-(kt,nt) rm_to_wh loop. The op reads rhs [K][N]
    // row-major directly (NO transpose) and writes tile (nt,kt) at
    // wtOff + (nt*n_k_tiles+kt)*2048 -- the order hmx_core_dot's col loop
    // expects (spike T8 weight-tile-order fix). Byte-identical output.
    //
    // Q8-weight path (flag-gated, default off): instead of laying out a
    // pre-dequantized fp16 [K][N] weight, fuse the int8->fp16 dequant into the
    // layout by emitting hvx_layout_weight_q8 over the q8x4x2 weight in its
    // natural [N][row_stride] orientation -- eliminating the slow scalar
    // [N][K]->[K][N] dequant+transpose. The Q8 i8 weight is supplied as an extra
    // i8 memref argument on the kernel func (the fp16 rhs still carries K/N for
    // hmx_core_dot's tile counts). act layout / hmx_core_dot / readout are
    // unchanged, so the default fp16 path is byte-identical when the flag is off.
    if (enableQ8Weight) {
      Value q8Src = findQ8WeightArg(op);
      if (!q8Src)
        return rewriter.notifyMatchFailure(
            op, "enableQ8Weight: no i8 weight memref argument found on the "
                "enclosing func");
      rewriter.create<hexkl::HvxLayoutWeightQ8Op>(loc, vtcm, wtOffI32, q8Src, K,
                                                  N);
    } else {
      rewriter.create<hexkl::HvxLayoutWeightOp>(loc, vtcm, wtOffI32, rhs, K, N);
    }

    // --- 4. init bias struct in the bias tile ---
    // HMX scale/bias structure for Q6_bias_mxmem2_A: first 32 i32 lanes =
    // 0x00003c00 (per-column scale 1.0, bias 0), rest of the 2048-byte tile 0.
    // View the i8 VTCM at byte offset biasOff as 512 x i32 and fill.
    {
      auto i32MemTy =
          MemRefType::get({512}, i32Ty, MemRefLayoutAttrInterface{},
                          IntegerAttr::get(rewriter.getI32Type(), 1));
      Value biasOffIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), biasOffI32);
      Value biasView = rewriter.create<memref::ViewOp>(
          loc, i32MemTy, vtcm, biasOffIdx, ValueRange{});
      Value cScale =
          rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 0x00003c00);
      Value cZero = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 0);
      Value idx512 = rewriter.create<arith::ConstantIndexOp>(loc, 512);
      rewriter.create<scf::ForOp>(
          loc, idx0, idx512, idx1, ValueRange{},
          [&](OpBuilder &b, Location loc, Value i, ValueRange) {
            Value isScale = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ult, i, idx32);
            Value v = b.create<arith::SelectOp>(loc, isScale, cScale, cZero);
            b.create<memref::StoreOp>(loc, v, biasView, ValueRange{i});
            b.create<scf::YieldOp>(loc);
          });
    }

    // --- 5. ONE hexkl.hmx_core_dot ---
    rewriter.create<hexkl::HmxCoreDotOp>(
        loc, vtcm, outOffI32, actOffI32, wtOffI32, biasOffI32, mTilesI32,
        nTilesI32, kTilesI32);

    // --- 6. readout all (mt,nt) output tiles -> DDR fp32 ---
    // ONE compiler-emitted HVX vgathermhw + fpext readout over the whole acc
    // region, replacing the per-(mt,nt) ah_to_rm + copy_f16_to_f32 loop. The op
    // reads acc tile (mt,nt) from outOff + (mt*n_n_tiles+nt)*2048 (matches
    // hmx_core_dot's (r*n_col_tiles+c) write order), gathers to flat row-major
    // fp16 in the scratch tile, widens to fp32, and writes result [M][N] DDR.
    // fpext is exact (vendor was lossy) -> <=1 f32 ULP, << fp16 rounding.
    rewriter.create<hexkl::HvxReadoutOp>(loc, vtcm, outOffI32, scratchOffI32,
                                         result, M, N);

    // Explicitly deallocate VTCM buffer.
    rewriter.create<hexagonmem::DeallocOp>(loc, vtcm);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateDecomposeHexKLMatmulPatterns(RewritePatternSet &patterns,
                                          bool useInlineHMX,
                                          bool enableQ8Weight) {
  patterns.add<DecomposeHexKLMatmul>(patterns.getContext(), useInlineHMX,
                                     enableQ8Weight);
}

struct DecomposeHexKLMatmulPass
    : public ::impl::DecomposeHexKLMatmulBase<DecomposeHexKLMatmulPass> {
  DecomposeHexKLMatmulPass() = default;
  explicit DecomposeHexKLMatmulPass(const DecomposeHexKLMatmulOptions &options)
      : DecomposeHexKLMatmulBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<hexkl::HexKLDialect, hexagonmem::HexagonMemDialect,
                arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateDecomposeHexKLMatmulPatterns(patterns, useInlineHMX, enableQ8Weight);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createDecomposeHexKLMatmulPass() {
  return std::make_unique<DecomposeHexKLMatmulPass>();
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createDecomposeHexKLMatmulPass(
    const DecomposeHexKLMatmulOptions &options) {
  return std::make_unique<DecomposeHexKLMatmulPass>(options);
}
