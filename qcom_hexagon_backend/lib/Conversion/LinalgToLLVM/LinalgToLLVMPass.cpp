//===- LinalgToLLVMPass.cpp - Linalg to LLVM  conversion       ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements optimization and lowering of MLIR IR to LLVM.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Conversion/HexKLToLLVM/Passes.h"
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/FoldConstantTransposes.h"
#include "hexagon/Conversion/LinalgToLLVM/HoistVectorTransferScratch.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Dialect/Crouton/IR/CroutonDialect.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"

#include <chrono>
#include <cstdlib>
#include <mutex>

#define DEBUG_TYPE "linalg-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LINALGTOLLVM
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

// Opt-in progress without dumping model constants or entire operations.
class LoweringProgress : public PassInstrumentation {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::mutex mutex;

  void report(StringRef event, Pass *pass, Operation *op) {
    std::lock_guard<std::mutex> lock(mutex);
    double elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start).count();
    llvm::errs() << "[hexmlir-pass] " << elapsed << "s: " << event << " "
                 << pass->getName() << " on " << op->getName().getStringRef();
    if (auto name = op->getAttrOfType<StringAttr>("sym_name"))
      llvm::errs() << " @" << name.getValue();
    llvm::errs() << "\n";
  }

public:
  void runBeforePass(Pass *pass, Operation *op) override {
    report("begin", pass, op);
  }
  void runAfterPass(Pass *pass, Operation *op) override {
    report("end", pass, op);
  }
  void runAfterPassFailed(Pass *pass, Operation *op) override {
    report("failed", pass, op);
  }
};

struct LinalgToLLVMPass : public ::impl::LinalgToLLVMBase<LinalgToLLVMPass> {
public:
  explicit LinalgToLLVMPass(const LinalgToLLVMOptions &options)
      : Base(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, async::AsyncDialect, tensor::TensorDialect,
                    cf::ControlFlowDialect, bufferization::BufferizationDialect,
                    vector::VectorDialect, memref::MemRefDialect,
                    LLVM::LLVMDialect, crouton::CroutonDialect, ttx::TTXDialect,
                    tptr::HexagonTPtrDialect, hexagonmem::HexagonMemDialect,
                    hexkl::HexKLDialect, quant::QuantDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    setTargetTriple(moduleOp);
    setDataLayout(moduleOp);

    auto setIndexBitwidth = [&](auto passOption) {
      passOption.indexBitwidth = 32;
      return passOption;
    };

    auto setFusion = [&](auto passOption) {
      passOption.fusionAllowRecompute = fusionAllowRecompute;
      passOption.fusionDoMultiUse = fusionDoMultiUse;
      return passOption;
    };

    auto setExtendPack = [&](auto passOption) {
      passOption.upperFrontier = extendPackUpperFrontier;
      passOption.lowerFrontier = extendPackLowerFrontier;
      passOption.parallelsOnly = extendPackParallelsOnly;
      return passOption;
    };

    auto setVTCMTiling = [&](auto passOption) {
      passOption.tileSizes = tileSizes;
      passOption.vtcmBudget = scratch > 0 ? scratch : 0;
      return passOption;
    };

    auto setuseInterchangeVector = [&](auto passOption) {
      passOption.useInterchangeVector = useInterchangeVector;
      return passOption;
    };

    auto setOpSlicingFactor = [&](auto passOption) {
      passOption.slicingFactor = slicingFactor;
      return passOption;
    };

    auto setsplitTilingRange = [&](auto passOption) {
      passOption.splitTilingRange = splitTilingRange;
      return passOption;
    };

    auto setenableSplitReduction = [&](auto passOption) {
      passOption.enableSplitReduction = enableSplitReduction;
      return passOption;
    };

    auto setAllowReturnAllocs = [&](auto passOption) {
      passOption.allowReturnAllocsFromLoops = true;
      return passOption;
    };

    auto setBufferizeFunctionBoundaries = [&](auto passOption) {
      passOption.bufferizeFunctionBoundaries = true;
      return passOption;
    };

    auto setLWP = [&](auto passOption) {
      passOption.disableLWPLoop = disableLWPLoop;
      passOption.LWPloopDepth = LWPloopDepth;
      return passOption;
    };

    auto setDeviceType = [&](auto passOption) {
      passOption.device_type = device_type;
      return passOption;
    };

    // Set ConvTiling flags
    auto setConvTiling = [&](auto passOption) {
      passOption.convTileSizes = convTileSizes;
      return passOption;
    };

    auto setHexKLMode = [&](auto passOption) {
      passOption.mode = hexKLMode;
      return passOption;
    };

    PassManager pm(&getContext(), moduleOp.getOperationName());
    const char *trace = std::getenv("HEXAGON_MLIR_PROGRESS");
    if (trace && (StringRef(trace) == "true" || StringRef(trace) == "1")) {
      // runPipeline uses the parent instrumentor, not the local PM's.
      if (auto *instrumentor = getAnalysisManager().getPassInstrumentor())
        instrumentor->addInstrumentation(std::make_unique<LoweringProgress>());
    }

    // RequestCWrappersPass adds an attribute to a function if it has a return
    // value which would generate a c-wrapper function during the
    // FuncToLLVMPass. See here for more information:
    // https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
    //
    // For example given the following function definition:
    // func.func @foobar(%arg0: memref<128x128xf32>)
    // -> memref<128x128xf32>
    //
    // After the RequestCWrappersPass it is converted to,
    // func.func @foobar(%arg0: memref<128x128xf32>)
    // -> memref<128x128xf32> attributes {llvm.emit_c_interface}
    //
    // After FuncToLLVMPass we get an additional function,
    // llvm.func @_mlir_ciface_foobar(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
    // attributes {llvm.emit_c_interface} {
    //   %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64,
    //   array<2 x i64>, array<2 x i64>)>
    //   %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x
    //   i64>, array<2 x i64>)>
    //   ...
    //   %8 = llvm.call @foobar(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr,
    //   !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64,
    //   array<2 x i64>, array<2 x i64>)>
    //
    //   llvm.store %8, %arg0 :
    //   !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>,
    //   !llvm.ptr
    //
    //   llvm.return
    // }
    //
    if (doesFuncReturnValue(moduleOp))
      pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());

    pm.addNestedPass<func::FuncOp>(createLowerTTXPass());
    pm.addPass(createLowerLibdevicePass());
    pm.addNestedPass<func::FuncOp>(createLowerTPtrPass());
    pm.addNestedPass<func::FuncOp>(createHexagonLowerTmTensorPass());
    pm.addNestedPass<func::FuncOp>(createReduceContractionRankPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (puntBuffer)
      pm.addNestedPass<func::FuncOp>(createHexagonPuntBufferPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung allocs

    if (enableConversionToFp16)
      pm.addNestedPass<func::FuncOp>(createConversionToFp16Pass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(createOptimizeExtfTruncfOpPass());

    // Reciprocal multiplication changes rounding even without fast-math
    // attributes on the new operations. Honor the pipeline opt-out.
    if (addFastMath)
      pm.addNestedPass<func::FuncOp>(createDivToMulOptimizationPass());

    // Quantization related passes in this block
    // Lower quant.qcast and quant.dcast ops to arith dialect
    pm.addNestedPass<func::FuncOp>(quant::createLowerQuantOps());
    // Convert arith ops to linalg elementwise ops
    pm.addPass(createConvertElementwiseToLinalgPass());
    // Remove quant.scast ops
    pm.addPass(createCSEPass());
    if (enableHexKL) {
      pm.addNestedPass<func::FuncOp>(
          mlir::hexagon::createFoldResourceTransposePass());
      pm.addNestedPass<func::FuncOp>(
          mlir::hexagon::createFoldCastsIntoMatmulPass());
      pm.addNestedPass<func::FuncOp>(
          createMatmulToHexKLPass(setHexKLMode(MatmulToHexKLOptions{})));
      if (hexKLMode == "macro") {
        pm.addNestedPass<func::FuncOp>(
            hexagon::createPreprocessWeightsForHMXPass());
      }
      pm.addPass(createCanonicalizerPass());
    }

    // enableMatmulToConv and enableSeedLayoutConversions are supposed to be set
    // for unit test only. They are not supposed to run on Full models
    if (enableMatmulToConv && enableSeedLayoutConversions) {
      pm.addNestedPass<func::FuncOp>(createMatmulToConvPass());
      pm.addNestedPass<func::FuncOp>(createSeedLayoutConversionsPass());
      pm.addNestedPass<func::FuncOp>(createHexagonExtendPackPass(
          setExtendPack(HexagonExtendPackOptions{})));
      pm.addPass(createCSEPass());
    }

    if (enableConvTiling) {
      pm.addNestedPass<func::FuncOp>(
          createConvTilingPass(setConvTiling(ConvTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    if (enableSeedLayoutConversions) {
      pm.addNestedPass<func::FuncOp>(createPreprocessTiledConv2DPass());
    }

    pm.addPass(std::make_unique<mlir::hexagon::FoldConstantTransposesPass>());
    pm.addPass(createCanonicalizerPass());
    if (tileOutlinedMatmul && !outlineMatmul) {
      moduleOp.emitError("tile-outlined-matmul requires outline-matmul");
      return signalPassFailure();
    }
    if (outlineMatmul)
      pm.addPass(createOutlineMatmulPass(tileOutlinedMatmul));
    pm.addNestedPass<func::FuncOp>(createScheduleMatmulForHVXPass());
    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizePass());

    if (returnValueOptimization)
      pm.addNestedPass<func::FuncOp>(createHexagonRVOPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung re-interprets
    pm.addPass(createCSEPass());

    if (enableSCFThreading) {
      assert(!enableMultiThreading && !enableVTCMTiling && scratch == 0 &&
             "currently scf-threading can be enabled only if"
             " linalg multi-threading and vtcm tiling are off");
      pm.addNestedPass<func::FuncOp>(createFormSCFThreadsPass());
    }

    if (fusion)
      pm.addNestedPass<func::FuncOp>(
          createHexagonFusionPass(setFusion(HexagonFusionOptions{})));
    pm.addPass(createEraseUnusedLinalgOperands());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (enableSlicing)
      pm.addPass(createHexagonSlicingPass(
          setOpSlicingFactor(HexagonSlicingOptions{})));

    pm.addNestedPass<func::FuncOp>(createDecomposeTensorConcatPass());
    if (forceHVXCroutonization) {
      pm.addNestedPass<func::FuncOp>(createForceHVXCroutonPass());
      pm.addNestedPass<func::FuncOp>(
          createHexagonExtendPackPass(setExtendPack(HexagonExtendPackOptions{
              .upperFrontier = false,
          })));
    }

    pm.addNestedPass<func::FuncOp>(createLowerPackPass());
    pm.addPass(createCSEPass());

    // VTCMTilingPass must run when scratch > 0 to create the VTCM allocs
    // that MemoryOffsetsPass will replace with views into the scratch buffer.
    // When scratch > 0, pass it as vtcmBudget so tile sizes respect the
    // per-instance budget rather than the hardcoded 2 MB default.
    if (enableVTCMTiling || scratch > 0) {
      pm.addNestedPass<func::FuncOp>(
          createVTCMTilingPass(setVTCMTiling(VTCMTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    // split linalg.reduce into [parallel,reduce] followed by smaller [reduce].
    if (enableSplitReduceGeneric) {
      pm.addNestedPass<func::FuncOp>(createSplitReduceGenericPass());
    }

    if (enableMultiThreading) {
      pm.addNestedPass<func::FuncOp>(
          createFormVirtualThreadsPass(FormVirtualThreadsOptions{}));
    }

    pm.addPass(removeMLProgramPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    // Fused RMSNorm still contains fpowi(x, 2) here. Normalize it before
    // split-reduction admission, not only after tiling. Keep the default
    // pipeline unchanged when the explicitly opt-in reduction policy is off.
    if (enableVectorization && enableSplitReduction)
      pm.addPass(
          createSmallExponentToMultiplyPass(SmallExponentToMultiplyOptions{}));
    if (enableVectorization) {
      pm.addPass(
          createHexagonTilingPass(setsplitTilingRange(setuseInterchangeVector(
              setenableSplitReduction(HexagonTilingOptions{})))));
    }

    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(
        createSmallExponentToMultiplyPass(SmallExponentToMultiplyOptions{}));
    // ===== STEP 1: HOIST SCALAR OPS =====
    // Run before vectorization to expose scalar invariants
    pm.addNestedPass<func::FuncOp>(createHoistScalarOpsPass());
    pm.addPass(createEraseUnusedLinalgOperands());
    pm.addPass(createCSEPass());

    // ===== STEP 1.5: LOOP INVARIANT CODE MOTION =====
    // Move hoisted scalars further up the loop nest
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Run LICM again after canonicalization to catch newly exposed
    // opportunities
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());

    // ===== STEP 2: VECTORIZATION =====
    // Vectorizer now sees cleaner IR with hoisted scalars
    if (enableVectorization) {
      pm.addPass(createHexagonVectorizationPass());
    }
    pm.addPass(createRewriteUBPoisonToZeroPass());
    pm.addPass(createHexagonVectorLoweringPass());
    pm.addPass(createCanonicalizerPass());

    if (addFastMath) {
      pm.addPass(createHexagonAddFastMathPass());
      pm.addNestedPass<func::FuncOp>(createFoldMulFByZeroPass());
      pm.addPass(createCanonicalizerPass());
    }
    pm.addPass(memref::createResolveShapedTypeResultDimsPass());

    if (enableBufferization) {
      pm.addPass(bufferization::createEmptyTensorEliminationPass());

      // Erase unnecessary vector-to-tensor writeback in loops before
      // bufferization.
      pm.addNestedPass<func::FuncOp>(createEraseVectorToTensorWritebackPass());

      mlir::bufferization::OneShotBufferizePassOptions passOpts;
      passOpts.bufferizeFunctionBoundaries = true;
      passOpts.allowReturnAllocsFromLoops = true;
      if (outlineMatmul) {
        // Preserve contiguous transfers inside DSP-local helpers. Dynamic
        // signature strides otherwise force lane-by-lane vector construction.
        // Bufferization inserts a copy when an actual argument is noncontiguous.
        passOpts.functionBoundaryTypeConversion =
            bufferization::LayoutMapOption::IdentityLayoutMap;
      }
      pm.addPass(bufferization::createOneShotBufferizePass(passOpts));
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());

      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            createHexagonDoubleBufferGenericS1Pass());
      }

      pm.addNestedPass<func::FuncOp>(
          bufferization::createBufferLoopHoistingPass());

      pm.addNestedPass<func::FuncOp>(createCopyCanonicalizationPass());
      pm.addPass(createCanonicalizerPass());

      bufferization::buildBufferDeallocationPipeline(
          pm, bufferization::BufferDeallocationPipelineOptions{});

      pm.addPass(createCSEPass());
      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            createHexagonDoubleBufferGenericS2Pass());
      }

      // SCF Loop Unrolling of innermost loop after vectorization.
      if (enableSCFLoopUnroll) {
        pm.addNestedPass<func::FuncOp>(createSCFLoopUnrollPass());
      }

      pm.addNestedPass<func::FuncOp>(createConvertZeroSizeMemrefPass());
      pm.addPass(createConvertBufferizationToMemRefPass());
    }

    if (enableConvertToHexagonmem)
      pm.addNestedPass<func::FuncOp>(createConvertToHexagonmemPass());

    // External VTCM scratch mode: inject scratch arg and replace VTCM allocs
    // with views into the per-instance scratch buffer (hexagon.scratch).
    if (scratch > 0) {
      InsertScratchArgOptions scratchOpts;
      scratchOpts.scratch = scratch;
      pm.addNestedPass<func::FuncOp>(createInsertScratchArgPass(scratchOpts));
      pm.addNestedPass<func::FuncOp>(createMemoryOffsetsPass());
    }

    if (enableHexKL) {
      if (hexKLMode == "macro") {
        // Lower to HexKL macro API
        pm.addNestedPass<func::FuncOp>(createLowerHexKLMatmulToMacroPass());
      } else {
        // Decompose hexkl.matmul to micro ops
        pm.addNestedPass<func::FuncOp>(createDecomposeHexKLMatmulPass());
      }
    }

    // Lower linalg ops with library_call attribute set to custom fns.
    pm.addPass(createHexagonReplaceWithLibraryCallsPass());
    if (enableHexagonmemCopyToDMA)
      pm.addNestedPass<func::FuncOp>(createHexmemCpyToDMAPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertLinalgToLoopsPass());

    pm.addNestedPass<func::FuncOp>(createFormAsyncThreadsPass());
    pm.addPass(createAsyncFuncToAsyncRuntimePass());
    pm.addPass(createAsyncToAsyncRuntimePass());

    pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
    // Vector transfer lowering creates new stack temporaries after the earlier
    // buffer-hoisting stage. Move eligible vector stack temporaries out of sequential loops
    // before SCF scopes disappear: otherwise LLVM alloca executes on every
    // iteration and accumulates until function return (6 MiB per Qwen MLP
    // projection in the observed batch-one BF16 decode).
    pm.addNestedPass<func::FuncOp>(createHoistVectorTransferScratchPass());

    if (enableLWP)
      pm.addNestedPass<func::FuncOp>(
          createHexagonLWPPass(setLWP(mlir::hexagon::HexagonLWPPassOptions{})));

    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addNestedPass<func::FuncOp>(createExpandMathOpsPass());

    if (expandBoolVec)
      pm.addNestedPass<func::FuncOp>(createExpandBoolVecPass());

    // Omitting fast-math attributes does not undo an approximate reciprocal.
    if (addFastMath)
      pm.addPass(createFastInversePass());
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createConvertIndexToLLVMPass(
        setIndexBitwidth(ConvertIndexToLLVMPassOptions{})));

    pm.addPass(createConvertAsyncToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass(ConvertFuncToLLVMPassOptions{}));

    pm.addPass(hexagon::createDMAToLLVMPass());
    pm.addPass(hexagonmem::createHexagonMemToLLVMPass(
        setDeviceType(hexagonmem::HexagonMemToLLVMOptions{})));
    pm.addPass(hexkl::createHexKLToLLVMPass());

    if (enableCollapseAddressSpace) {
      pm.addPass(createCollapseAddressSpacePass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    }

    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createArithToLLVMConversionPass());
    // Vector/shape lowering may retain poison values for inactive lanes (e.g.
    // cached single-token decode). Preserve poison semantics while converting
    // the remaining UB dialect; LLVM IR export cannot translate ub.poison.
    pm.addPass(createUBToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (enableHexagonRoutines)
      pm.addPass(createHexagonLLVMEnableHexagonRoutinesPass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLinalgToLLVMPass(const LinalgToLLVMOptions &options) {
  return std::make_unique<LinalgToLLVMPass>(options);
}
