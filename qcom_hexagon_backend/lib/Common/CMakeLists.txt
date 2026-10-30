add_library(HexagonCommon STATIC
     Common.cpp
     DMATransferUtil.cpp
)
target_link_libraries(HexagonCommon PUBLIC MLIRMemRefUtils)
add_dependencies(HexagonCommon HexagonMemTableGen LinalgToLLVMConversionPassIncGen)
