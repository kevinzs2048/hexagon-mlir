add_mlir_conversion_library(HexagonMemToLLVM
  HexagonMemToLLVMPass.cpp
  HexagonMemExternalFnNames.cpp

  DEPENDS
  HexagonMemToLLVMConversionPassIncGen
  LinalgToLLVMConversionPassIncGen
  HexagonMemTableGen

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
  MLIRSupport
  HexagonCommon
)
