add_mlir_translation_library(TritonLinalgLLVM
        MLLVMIRTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTargetLLVMIRExport
        # MLLVMIRTranslation.cpp calls hexagon::createLinalgToLLVMPass(), defined
        # in the LinalgToLLVM conversion lib. Declare the dependency so CMake
        # orders libLinalgToLLVM.a AFTER libTritonLinalgLLVM.a on the link line;
        # without it the single-pass linker drops the pass member (undefined
        # symbol createLinalgToLLVMPass in libtriton.so).
        LinalgToLLVM
        )
