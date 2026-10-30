add_mlir_conversion_library(LinalgToLLVM
  Common.cpp
  ExtendPackUnpackFrontier.cpp
  FormSCFThreads.cpp
  FormVirtualThreadsPass.cpp
  FormAsyncThreadsPass.cpp
  LinalgToLLVM.cpp
  LinalgToLLVMPass.cpp
  HexmemCpyToDMA.cpp
  FusionPass.cpp
  SplitReduceGenericPass.cpp
  EraseVectorToTensorWritebackPass.cpp
  SlicingPass.cpp
  TilingPass.cpp
  VectorizationPass.cpp
  RewriteUBPoisonToZeroPass.cpp
  VTCMTiling.cpp
  LowerPack.cpp
  VTCMTilingOptions.cpp
  LowerConstantsSeparately.cpp
  SplitReductionUtils.cpp
  ConversionToFp16Pass.cpp
  OptimizeExtfTruncfOpPass.cpp
  
  DEPENDS
  LinalgToLLVMConversionPassIncGen
  TransformsPassIncGen
  DMAToLLVMConversionPassIncGen
  HexKLToLLVMConversionPassIncGen
  HexagonMemToLLVMConversionPassIncGen
  HexagonMemTableGen
  HexagonTPtrTableGen
  HexKLTableGen
  TmTensorTableGen
  TTXTableGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRMathTransforms
  MLIRSupport
  MLIRUBToLLVM
  MLIRRegisterAllExtensions

  Transforms
)
