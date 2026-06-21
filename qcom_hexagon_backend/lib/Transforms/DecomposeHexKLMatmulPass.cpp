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
      // The Q8 weight is the DDR (memory space 0) i8 memref arg. Skip any VTCM
      // i8 arg (memory space 1) so the Q8 lookup stays unambiguous.
      if (mt.getElementType().isInteger(8) &&
          mt.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE) {
        if (found) // ambiguous: more than one DDR i8 memref arg
          return nullptr;
        found = arg;
      }
    }
    return found;
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

    // Region tile counts (index).
    Value mkTiles = rewriter.create<arith::MulIOp>(loc, mTiles, kTiles);
    Value mnTiles = rewriter.create<arith::MulIOp>(loc, mTiles, nTiles);

    // ----------------------------------------------------------------------
    // N-chunking (Q8/CROUTON path): the weight (N*K) is the VTCM-overflow
    // source for large prefill shapes (e.g. N6144*K8192 weight = 96 MiB >> 8
    // MiB VTCM). We chunk the OUTPUT-COLUMN (N) dimension: each N_chunk is an
    // INDEPENDENT, complete matmul over the FULL K (clear -> stream full K ->
    // readout), exactly like the all-resident path, just over fewer output
    // columns. NO cross-call accumulation. The activation (M*K, full-K) is laid
    // out ONCE outside the N-loop and reused; only the weight region is sized
    // per-chunk. The output region stays full-N (cheap: m_tiles*n_tiles tiles)
    // so the single full-N readout consumes the concatenation untouched.
    //
    //   arena(chunk) = (mkTiles + chunk_nt*kTiles + mnTiles)*2048 + 3*2048
    //                                 ^^wt^^                       ^bias/flat/scratch
    // Solve chunk_nt under a 7 MiB budget (1 MiB VTCM slack):
    //   avail_wt_tiles = BUDGET/2048 - mkTiles - mnTiles - 3
    //   chunk_nt       = clamp(avail_wt_tiles / kTiles, 1, nTiles)
    // chunk_nt==nTiles (single chunk) reproduces the all-resident path exactly.
    constexpr int64_t kVtcmBudgetTiles = (7 * 1024 * 1024) / 2048; // 3584
    Value budgetTilesC =
        rewriter.create<arith::ConstantIndexOp>(loc, kVtcmBudgetTiles);
    Value idx3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    Value fixedTiles = rewriter.create<arith::AddIOp>(loc, mkTiles, mnTiles);
    fixedTiles = rewriter.create<arith::AddIOp>(loc, fixedTiles, idx3);
    // availWt = budget - fixed; guard non-negative (clamp to >=kTiles so the
    // div yields >=1). If even one column doesn't fit, chunk_nt clamps to 1 and
    // the alloc may still exceed VTCM -> the DSP would fault; for Qwen3
    // (K<=8192) one column + full-K act always fits, so this is a guard only.
    Value availWt = rewriter.create<arith::SubIOp>(loc, budgetTilesC, fixedTiles);
    // max(availWt, kTiles) so chunk_nt >= 1 even under pressure.
    Value availClamped = rewriter.create<arith::MaxSIOp>(loc, availWt, kTiles);
    Value chunkNtRaw =
        rewriter.create<arith::DivUIOp>(loc, availClamped, kTiles);
    // min(chunkNtRaw, nTiles): never chunk wider than the matrix.
    Value chunkNt = rewriter.create<arith::MinUIOp>(loc, chunkNtRaw, nTiles);
    // The fp16 (non-Q8) weight layout op does not support a tile range, so it
    // can only run all-resident: force a single chunk (chunk_nt = nTiles). The
    // N-chunking is exercised only on the Q8/CROUTON prefill path.
    if (!enableQ8Weight)
      chunkNt = nTiles;
    Value chunkNtI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, chunkNt);

    // wt region is sized for ONE chunk (chunk_nt * kTiles tiles).
    Value chunkNkTiles = rewriter.create<arith::MulIOp>(loc, chunkNt, kTiles);
    Value actBytes = rewriter.create<arith::MulIOp>(loc, mkTiles, idx2048);
    Value wtBytes = rewriter.create<arith::MulIOp>(loc, chunkNkTiles, idx2048);
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

    // --- 4. init bias struct in the bias tile (ONCE, reused per N_chunk) ---
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

    // --- 3+5. N-loop: per-chunk weight layout + hmx_core_dot ---
    // Each iteration handles a contiguous block of chunk_nt output-column tiles
    // [ntStart, ntStart+thisNt). thisNt = min(chunk_nt, nTiles-ntStart) handles
    // the tail. Per chunk:
    //   (a) lay out THIS chunk's weight tiles (linear tile range
    //       [ntStart*kTiles, (ntStart+thisNt)*kTiles)) into the chunk-sized wt
    //       region at its BASE (the q8 lowering writes dst chunk-local).
    //   (b) ONE hmx_core_dot over full K, thisNt columns, writing into the
    //       FULL-N output region at column offset ntStart with full-N row stride
    //       (out_col_stride = nTiles, out_col_off = ntStart*mTiles?  -- NO: the
    //       output tile index is r*nTiles + ntStart + c, so out_col_off =
    //       ntStart, out_col_stride = nTiles).
    // Single full-N readout AFTER the loop consumes the [mt][nt] row-major
    // output region untouched.
    //
    // The default fp16 weight path (non-Q8) cannot do tile ranges, so chunkNt
    // was forced to nTiles above -> exactly one iteration, all-resident, and we
    // emit the plain (non-range, non-col-offset) ops = byte-identical legacy.
    Value q8Src;
    if (enableQ8Weight) {
      q8Src = findQ8WeightArg(op);
      if (!q8Src)
        return rewriter.notifyMatchFailure(
            op, "enableQ8Weight: no i8 weight memref argument found on the "
                "enclosing func");
    }
    {
      // tile_start_dyn / out_col_off are linear-tile / column-tile indices.
      rewriter.create<scf::ForOp>(
          loc, idx0, nTiles, chunkNt, ValueRange{},
          [&](OpBuilder &b, Location loc, Value ntStart, ValueRange) {
            // thisNt = min(chunkNt, nTiles - ntStart)
            Value rem = b.create<arith::SubIOp>(loc, nTiles, ntStart);
            Value thisNt = b.create<arith::MinUIOp>(loc, chunkNt, rem);
            Value thisNtI32 =
                b.create<arith::IndexCastOp>(loc, i32Ty, thisNt);
            Value ntStartI32 =
                b.create<arith::IndexCastOp>(loc, i32Ty, ntStart);

            if (enableQ8Weight) {
              // Weight tile range [tileStart, tileStart+tileCount) (linear,
              // [nt][kt] order, nt outer): tileStart = ntStart*kTiles,
              // tileCount = thisNt*kTiles.
              Value tsIdx = b.create<arith::MulIOp>(loc, ntStart, kTiles);
              Value tcIdx = b.create<arith::MulIOp>(loc, thisNt, kTiles);
              Value tsI32 = b.create<arith::IndexCastOp>(loc, i32Ty, tsIdx);
              Value tcI32 = b.create<arith::IndexCastOp>(loc, i32Ty, tcIdx);
              b.create<hexkl::HvxLayoutWeightQ8Op>(
                  loc, vtcm, wtOffI32, q8Src, K, N,
                  /*tile_start=*/b.getI32IntegerAttr(0),
                  /*tile_count=*/b.getI32IntegerAttr(0),
                  /*tile_start_dyn=*/tsI32, /*tile_count_dyn=*/tcI32);
              // hmx_core_dot: thisNt columns over full K, output into full-N
              // region at column offset ntStart (row stride = full nTiles).
              b.create<hexkl::HmxCoreDotOp>(
                  loc, vtcm, outOffI32, actOffI32, wtOffI32, biasOffI32,
                  mTilesI32, thisNtI32, kTilesI32,
                  /*out_col_stride=*/nTilesI32, /*out_col_off=*/ntStartI32);
            } else {
              // Single all-resident chunk (chunkNt==nTiles): legacy ops.
              b.create<hexkl::HvxLayoutWeightOp>(loc, vtcm, wtOffI32, rhs, K, N);
              b.create<hexkl::HmxCoreDotOp>(
                  loc, vtcm, outOffI32, actOffI32, wtOffI32, biasOffI32,
                  mTilesI32, nTilesI32, kTilesI32,
                  /*out_col_stride=*/Value(), /*out_col_off=*/Value());
            }
            b.create<scf::YieldOp>(loc);
          });
    }

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
