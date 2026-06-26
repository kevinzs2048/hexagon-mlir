//===- HexKLOps.td - HexKL Ops -----------------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Describes the hexkl dialect ops.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HEXKL_IR_HEXKL_OPS_TD
#define HEXAGON_DIALECT_HEXKL_IR_HEXKL_OPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hexagon/Dialect/HexKL/IR/HexKLDialect.td"

// Base class for HexKL dialect ops.
class HexKL_Op<string mnemonic, list<Trait> traits> :
    Op<HexKL_Dialect, mnemonic, traits> {

}

def HexKL_MatmulOp
  : HexKL_Op<"matmul",
      [DestinationStyleOpInterface,MemoryEffects<[MemRead, MemWrite]>]> {

  let summary = "HexKL Matmul op";
  let description = [{
    Performs matrix multiplication on F16 inputs and F32 output using HexKL APIs.
  }];

  let arguments = (ins
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$lhs,
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$rhs,
    AnyTypeOf<[Non0RankedTensorOf<[F32]>, Non0RankedMemRefOf<[F32]>]>:$outs
  );

  let results = (outs Variadic<Non0RankedTensorOf<[F32]>>:$result);

  let assemblyFormat = [{
    `ins` `(`  $lhs `,` $rhs`:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    // Implemented as part of DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() { return getOutsMutable(); }
  }];
}

def HexKL_MicroHMXSetupAccReadF16Op
  : HexKL_Op<"micro_hmx_setup_acc_read_f16",
      [MemoryEffects<[MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Setup accumulator for reading";
  let description = [{
    Sets up the accumulator for reading F16 values. The configuration offset
    is derived from the VTCM block during lowering and is not modeled as an
    operand in the IR.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `)` `:` type($hmx_block) attr-dict";
}

def HexKL_MicroHMXCopySubmatrixToF16Op
  : HexKL_Op<"micro_hmx_copy_submatrix_to_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Copy submatrix to F16";
  let description = [{
    Copies a 32x32 tile from a flat F16 matrix in DDR to VTCM at the given
    destination offset.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$out_offset,
    AnyNon0RankedMemRef:$src,
    I32:$tile_row,
    I32:$tile_col,
    I32:$input_rows,
    I32:$input_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $out_offset `,` $src `,` $tile_row `,` $tile_col `,` $input_rows `,` $input_cols `)` `:` type($hmx_block) `,` type($out_offset) `,` type($src) `,` type($tile_row) `,` type($tile_col) `,` type($input_rows) `,` type($input_cols) attr-dict";
}

def HexKL_MicroHMXRmToAhF16Op
  : HexKL_Op<"micro_hmx_rm_to_ah_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Transform row-major to activation-horizontal F16";
  let description = [{
    Transforms a 32x32 fp16 tile from flat row-major layout in VTCM to
    activation layout at the destination offset.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$activation_out_offset,
    I32:$flat_in_offset
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $activation_out_offset `,` $flat_in_offset `)` `:` type($hmx_block) `,` type($activation_out_offset) `,` type($flat_in_offset) attr-dict";
}

def HexKL_MicroHMXAccClearF16Op
  : HexKL_Op<"micro_hmx_acc_clear_f16",
      [MemoryEffects<[MemWrite]>]> {

  let summary = "Clear accumulator";
  let description = [{
    Clears the F16 accumulator.
  }];

  let arguments = (ins  );

  let results = (outs);

  let assemblyFormat =
    "attr-dict";
}

def HexKL_MicroHMXRmToWhF16Op
  : HexKL_Op<"micro_hmx_rm_to_wh_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Transform row-major to weight-horizontal F16";
  let description = [{
    Converts a flat FP16 weight tile in DDR to the HMX weight layout and writes
    it to VTCM at the given destination offset.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$weight_offset,
    AnyNon0RankedMemRef:$src,
    I32:$tile_row,
    I32:$tile_col,
    I32:$wt_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $weight_offset `,` $src `,` $tile_row `,` $tile_col `,` $wt_cols `)` `:` type($hmx_block) `,` type($weight_offset) `,` type($src) `,` type($tile_row) `,` type($tile_col) `,` type($wt_cols) attr-dict";
}

def HexKL_MicroHMXMmF16Op
  : HexKL_Op<"micro_hmx_mm_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Micro HMX matrix multiplication F16";
  let description = [{
    Performs micro GEMM on activation and weight tiles in VTCM using F16 inputs.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$activation_offset,
    I32:$weight_offset
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $activation_offset `,` $weight_offset `)` `:` type($hmx_block) `,` type($activation_offset) `,` type($weight_offset) attr-dict";
}

def HexKL_MicroHMXAccReadF16Op
  : HexKL_Op<"micro_hmx_acc_read_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Read accumulator F16";
  let description = [{
    Reads the F16 accumulator and stores it in VTCM at the given destination offset.
    The configuration offset is derived from the VTCM block during lowering.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$out_offset
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $out_offset `)` `:` type($hmx_block) `,` type($out_offset) attr-dict";
}

def HexKL_MicroHMXAhToRmF16Op
  : HexKL_Op<"micro_hmx_ah_to_rm_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Transform activation-horizontal to row-major F16";
  let description = [{
    Transforms a 32x32 fp16 tile from activation layout in VTCM to flat row-major
    layout at the destination offset.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$flat_out_offset,
    I32:$activation_in_offset
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $flat_out_offset `,` $activation_in_offset `)` `:` type($hmx_block) `,` type($flat_out_offset) `,` type($activation_in_offset) attr-dict";
}

def HexKL_MicroHMXCopyF16ToF32SubmatrixOp
  : HexKL_Op<"micro_hmx_copy_f16_to_f32_submatrix",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Copy F16 to F32 submatrix";
  let description = [{
    Copies a 32x32 tile in flat F16 layout from VTCM to DDR as F32.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$in_offset,
    AnyNon0RankedMemRef:$dst,
    I32:$tile_row,
    I32:$tile_col,
    I32:$output_rows,
    I32:$output_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $in_offset `,` $dst `,` $tile_row `,` $tile_col `,` $output_rows `,` $output_cols `)` `:` type($hmx_block) `,` type($in_offset) `,` type($dst) `,` type($tile_row) `,` type($tile_col) `,` type($output_rows) `,` type($output_cols) attr-dict";
}

def HexKL_HvxLayoutWeightOp
  : HexKL_Op<"hvx_layout_weight",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Fast HVX weight-half tile layout via vscatter";
  let description = [{
    Lays out a row-major [K][N] FP16 weight (resident in VTCM) into the HMX
    weight-half tile format, writing to the VTCM `hmx_block` at `wt_off`. Emits
    inline HVX `V6.vscattermhw` halfword scatters instead of per-tile vendor
    `rm_to_wh` calls — byte-identical output, ~8x faster.

    The weight is consumed in the orientation the matmul provides ([K][N]
    row-major): for each K-row, the contiguous N values are scattered to their
    weight-half byte offsets directly, with NO separate transpose. K and N must
    be multiples of 32; N must be a multiple of 64 (pair-tile scatter).
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$wt_off,
    AnyNon0RankedMemRef:$src,
    I32:$k,
    I32:$n
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $wt_off `,` $src `,` $k `,` $n `)` `:` type($hmx_block) `,` type($wt_off) `,` type($src) `,` type($k) `,` type($n) attr-dict";
}

def HexKL_HvxLayoutWeightQ8Op
  : HexKL_Op<"hvx_layout_weight_q8",
      [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
  let hasVerifier = 1;

  let summary = "Fast HVX Q8_0 weight-half tile layout via fused dequant+vscatter";
  let description = [{
    Lays out a Q8_0 (q8x4x2) quantized weight `[N][row_stride]` (resident in
    DDR or VTCM) into the HMX weight-half tile format, writing to the VTCM
    `hmx_block` at `wt_off`, fusing the int8->fp16 dequantization (inline HVX)
    with the transposing vscatter. Mirrors the hand-written baseline
    `dequantize_x4x2_weight_to_fp16_tiles_task_q8_0`: it reads the weight in its
    NATURAL `[N][K]` orientation (N rows, K contiguous quants per row) — NO
    separate `[N][K]->[K][N]` transpose, which is the slow scalar phase this op
    eliminates.

    `q8_src` is the q8x4x2 weight `[N][row_stride]` where each row is
    `K` int8 quants (natural order) followed by `K/32` fp16 group scales
    (natural order); `row_stride = K + (K/32)*2` bytes. For each output tile the
    32 int8 quants of a K-group are widened to int16, converted to fp16, and
    multiplied by the group's fp16 scale (all HVX), then word-scattered (masked,
    region 2047) into the weight-half tile using the `i*128` transpose offsets.

    K must be a multiple of 32; N must be a multiple of 64.

    `tile_start` / `tile_count` (default 0 / 0) restrict the CROUTON-path
    layout to the half-open output-tile range `[tile_start, tile_start +
    tile_count)` over the `(N/32)*(K/32)` weight tiles (tile order `[nt][kt]`).
    `tile_count == 0` means "all tiles from tile_start to the end" — with the
    default `0/0` this is the full range, preserving backward compatibility.
    Each crouton tile is independent (src `t*1152`, dst `t*2048`), so a thread
    shim can call this op N times with disjoint `[tile_start, tile_count)`
    ranges to parallelize the dequant. Only the CROUTON path honors these; the
    default / WIDE paths ignore them.

    `tile_start_dyn` / `tile_count_dyn` (optional, runtime I32 operands) are the
    dynamic-shape counterpart of the `tile_start`/`tile_count` attrs, used by the
    N-chunked decompose where the chunk range is only known at runtime. When
    present they OVERRIDE the attrs. The src tile offset stays GLOBAL (`t*1152`,
    selecting the right weight columns) but the dst is written CHUNK-LOCAL
    (`(t - tile_start)*2048`) so the chunk's weight lands at the base of a
    chunk-sized wt region.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$wt_off,
    AnyNon0RankedMemRef:$q8_src,
    I32:$k,
    I32:$n,
    DefaultValuedAttr<I32Attr, "0">:$tile_start,
    DefaultValuedAttr<I32Attr, "0">:$tile_count,
    Optional<I32>:$tile_start_dyn,
    Optional<I32>:$tile_count_dyn
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $wt_off `,` $q8_src `,` $k `,` $n (`,` `range` $tile_start_dyn^ `,` $tile_count_dyn)? `)` `:` type($hmx_block) `,` type($wt_off) `,` type($q8_src) `,` type($k) `,` type($n) attr-dict";
}

def HexKL_MatmulVrmpyQ8Op
  : HexKL_Op<"matmul_vrmpy_q8", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "HVX int8-vrmpy 1-touch matmul over q8x4x2 weight+activation";
  let description = [{
    Computes out[M][N] f32 = act[M][K] (q8x4x2 int8) * weightT[N][K] (q8x4x2 int8),
    via Q6_Vw_vrmpy_VbVb (llvm.hexagon.V6.vrmpybv.128B) int8 x int8 -> int32, with the
    combined fp16 weight-scale x activation-scale applied AFTER int32 accumulation
    (no fp16 weight materialization). Both `act` and `wt` are q8x4x2-packed:
    `[K int8 quants][ (K/32) fp16 scales]`, row_stride = K + (K/32)*2 (the caller
    passes the padded/128-aligned stride via the memref). `k` is the runtime K (mult of 32).
  }];
  let arguments = (ins
    AnyNon0RankedMemRef:$out,   // f32 [M][N]
    AnyNon0RankedMemRef:$act,   // i8  [M][a_stride]
    AnyNon0RankedMemRef:$wt,    // i8  [N][w_stride]
    I32:$k
  );
  let results = (outs);
  let assemblyFormat =
    "`(` $out `,` $act `,` $wt `,` $k `)` `:` type($out) `,` type($act) `,` type($wt) `,` type($k) attr-dict";
}

def HexKL_MatmulVrmpyCroutonOp
  : HexKL_Op<"matmul_vrmpy_crouton", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "HVX int8-vrmpy 1-touch matmul: q8x4x2 activation x CROUTON weight";
  let description = [{
    Decode-path (small M) sibling of `hexkl.matmul_vrmpy_q8`. Computes
    out[M][N] f32 = act[M][K] (q8x4x2 int8) * weight (CROUTON 1152B-tile int8),
    via Q6_Vw_vrmpy_VbVb int8 x int8 -> int32, scale applied AFTER accumulation.

    The activation is the SAME q8x4x2 layout as matmul_vrmpy_q8:
    `[K int8 quants][(K/32) fp16 scales]`, row_stride = K + (K/32)*2.

    The weight is the OmniOp CROUTON tile layout (host `repack_q8_0_crouton`):
    per (nt,kt) tile of 1152 bytes, tile order [nt][kt] with nt=N/32, kt=K/32:
      bytes   0..127 : `sv` = 64 fp16 dup-scales, sv[r*2]=sv[r*2+1]=blk[nt*32+r].d
      bytes 128..1151 : `qz` = int8, qz[i*64 + r*2 + p] for colpair i=0..15,
                        row r=0..31, pair-elem p=0,1.
    For output row n (nt=n/32, r=n%32) and K-block kt, the 32 K int8 are SCATTERED
    at stride 64: qz[i*64 + r*2 + p]. The lowering GATHERS them K-contiguous
    (krow[2*i+p] = qz[i*64+r*2+p]) before the vrmpy, and pulls the row's fp16
    scale from sv[r*2] (the dup-scale drops in directly, no reshuffle). This
    mirrors PATH B of hmx-spike-compiler/spike_vrmpy_crouton.c (max_abs 1.7e-5).

    `k` is the runtime K (multiple of 32). N is derived from out's size1.
  }];
  let arguments = (ins
    AnyNon0RankedMemRef:$out,   // f32 [M][N]
    AnyNon0RankedMemRef:$act,   // i8  [M][a_stride]  q8x4x2
    AnyNon0RankedMemRef:$wt,    // i8  crouton tiles (nt*kt)*1152, flat
    I32:$k
  );
  let results = (outs);
  let assemblyFormat =
    "`(` $out `,` $act `,` $wt `,` $k `)` `:` type($out) `,` type($act) `,` type($wt) `,` type($k) attr-dict";
}

def HexKL_TLERawCallOp
  : HexKL_Op<"tle_raw_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Lower a recognized whole-kernel op to ONE call into a hand-written TLE-Raw core";
  let description = [{
    Generic leaf op for the TLE-Raw framework-lowering path. A recognition pass
    (LinalgToTLERawPass) rewrites a whole recognized kernel (e.g. the elementwise
    add kernel) into a single tle_raw_call whose `callee` names the hand-written
    core to invoke. By convention `operands[0]` is the OUTPUT memref and the rest
    are INPUT memrefs, in the core's C-ABI order. The lowering (LowerTLERawCall)
    extracts each memref's aligned base pointer, derives the element count `n` from
    the output memref descriptor (product of dim sizes), and emits exactly one
    `llvm.call @callee(ptr..., i32 n)`. No threading/tiling is introduced here;
    orchestration stays in the caller (llama.cpp shim).
  }];
  let arguments = (ins
    Variadic<AnyType>:$operands,   // leading memrefs (operands[0]=out, rest=inputs),
                                   // then optional trailing scalar operands (e.g. eps)
    StrAttr:$callee                // hand-written core symbol name
  );
  let results = (outs);
  let assemblyFormat =
    "$callee `(` $operands `)` `:` type($operands) attr-dict";
}

def HexKL_HvxLayoutActivationOp
  : HexKL_Op<"hvx_layout_activation",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Fast HVX activation-half tile layout via vscatter";
  let description = [{
    Lays out a row-major [M][K] FP16 activation (in DDR or VTCM) into the HMX
    activation-half tile format, writing to the VTCM `hmx_block` at `act_off`.
    Emits inline HVX `V6.vscattermhw` halfword scatters, replacing the vendor
    `copy_submatrix_to_f16` + `rm_to_ah_f16` per-tile loop — byte-identical.

    The activation is consumed directly in [M][K] row-major (no separate
    submatrix copy): for each (mt,kt) 32x32 tile, each fp16 of a tile row is
    scattered to its activation-half byte offset
        off(r,c) = (r/2)*128 + (r%2)*2 + c*4   (r,c in 0..31)
    Tile (mt,kt) lands at act_off + (mt*n_k_tiles + kt)*2048. M and K must be
    multiples of 32.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$act_off,
    AnyNon0RankedMemRef:$src,
    I32:$m,
    I32:$k
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $act_off `,` $src `,` $m `,` $k `)` `:` type($hmx_block) `,` type($act_off) `,` type($src) `,` type($m) `,` type($k) attr-dict";
}

def HexKL_HvxReadoutOp
  : HexKL_Op<"hvx_readout",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Fast HVX accumulator readout (ah->rm + f16->f32) via vgather";
  let description = [{
    Reads the HMX accumulator output tiles (activation-half FP16, in VTCM at
    `out_off`), converts to row-major FP32, and writes a row-major [M][N] FP32
    matrix `dst` (in DDR). Emits inline HVX `V6.vgathermhw` (the inverse of the
    activation-half scatter) to gather each tile into flat row-major FP16, then
    `V6.vcvt_sf_hf` to widen to FP32 — replacing the vendor `ah_to_rm_f16` +
    `copy_f16_to_f32_submatrix` per-tile loop. Byte-identical FP32 output.

    Output tile (mt,nt) read from out_off + (mt*n_n_tiles + nt)*2048; each fp16
    at activation-half byte offset off(r,c)=(r/2)*128+(r%2)*2+c*4 is placed at
    dst[mt*32+r][nt*32+c]. M and N must be multiples of 32.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$out_off,
    I32:$scratch_off,
    AnyNon0RankedMemRef:$dst,
    I32:$m,
    I32:$n
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $out_off `,` $scratch_off `,` $dst `,` $m `,` $n `)` `:` type($hmx_block) `,` type($out_off) `,` type($scratch_off) `,` type($dst) `,` type($m) `,` type($n) attr-dict";
}

def HexKL_HmxCoreDotOp
  : HexKL_Op<"hmx_core_dot",
      [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
  let hasVerifier = 1;

  let summary = "Deep-streaming HMX fp16 matmul over VTCM-resident tiles";
  let description = [{
    Deep-streaming HMX fp16 matmul over VTCM-resident activation-half/weight-half
    tiles; lowers to inline HMX intrinsics.

    The `c` (column) loop runs `[0, n_col_tiles)` and the weight half is read
    chunk-local (`wt_base + c*dot_stride`). By default each readout tile lands at
    `out_base + (r*n_col_tiles + c)*2048` — i.e. the VTCM output region is laid
    out `[mt][nt]` row-major with row stride = `n_col_tiles`.

    `out_col_stride` / `out_col_off` (optional, runtime I32 operands) support
    N-chunked matmul, where a chunk computes only `n_col_tiles` output columns
    but writes into a FULL-N output region so the single readout can consume it.
    The two operands are checked independently: `out_col_stride`, when present,
    replaces the row stride (`n_col_tiles` → `out_col_stride`); `out_col_off`,
    when present, adds the chunk's column offset. So the readout tile lands at
    `out_base + (r*<row_stride> + <col_off> + c)*2048`, where `<row_stride>`
    defaults to `n_col_tiles` and `<col_off>` defaults to 0. The only producer
    (DecomposeHexKLMatmulPass) always passes both together for an N-chunk and
    neither for a single-chunk matmul; the latter is byte-identical to the
    legacy `(r*n_col_tiles + c)` mapping.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$out_off,
    I32:$act_off,
    I32:$wt_off,
    I32:$bias_off,
    I32:$n_row_tiles,
    I32:$n_col_tiles,
    I32:$n_dot_tiles,
    Optional<I32>:$out_col_stride,
    Optional<I32>:$out_col_off
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $out_off `,` $act_off `,` $wt_off `,` $bias_off `,` $n_row_tiles `,` $n_col_tiles `,` $n_dot_tiles (`,` `cols` $out_col_stride^ `,` $out_col_off)? `)` `:` type($hmx_block) `,` type($out_off) `,` type($act_off) `,` type($wt_off) `,` type($bias_off) `,` type($n_row_tiles) `,` type($n_col_tiles) `,` type($n_dot_tiles) attr-dict";
}

#endif // HEXAGON_DIALECT_HEXKL_IR_HEXKL_OPS_TD
