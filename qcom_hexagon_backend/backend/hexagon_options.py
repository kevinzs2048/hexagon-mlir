# ===- hexagon_options.py ---------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
from dataclasses import dataclass
from typing import Tuple
import hashlib


@dataclass(frozen=True)
class HexagonOptions:
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    arch_triple: str = "hexagon"
    arch_features: str = f'+hvxv{os.getenv("HEXAGON_ARCH_VERSION")},+hvx-length128b'
    device_type: str = "hexagon"
    vectorize: int = 1
    vector_length: int = 32
    num_threads: int = 4
    data_layout: str = (
        "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:"
        "32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
    )
    # TODO: Next 7 options not used currently but kept to be consistent with other backends, remove later if not required
    num_warps: int = 1
    num_stages: int = 1
    num_ctas: int = 1
    shared: bool = False
    cluster_dims: tuple = (1, 1, 1)
    supported_fp8_dtypes: Tuple[str, ...] = ()
    sanitize_overflow: bool = True
    debug: bool = False
    # TODO: Currently set name here temporarily. Need to set correctly to the name of the compiled kernel
    # when driver code is completed.
    name: str = "Hexagon"
    instrumentation_mode: str = ""
    htp_kernel_gen: bool = False
    target_artifact: str = "o"
    iterations: int = 10  # Triton specific benchmarking iteration count

    # Hexagon Linalg Options
    # Experimental DSP-internal calls, not additional host/DSP RPC calls.
    # Off by default: call boundaries can inhibit fusion and affect runtime.
    outlineMatmul: bool = False
    tileOutlinedMatmul: bool = False  # Experimental DDR cache tiling for outlined W8A8 matmuls
    fusion: bool = True
    fusionAllowRecompute: bool = False
    fusionDoMultiUse: bool = True
    # Disable unrequested reassociation and other LLVM `fast` assumptions.
    # HVX instruction precision is a separate policy: LLVM translation selects
    # IEEE by default on v73 unless arch_features explicitly selects FP mode.
    addFastMath: bool = False

    enableBufferization: bool = True  # Used to disable for some dma testing
    enableCollapseAddressSpace: bool = True  # lower llvm.ptr<1> if hexagonmem did not
    enableConvTiling: bool = False
    enableDoubleBuffering: bool = False  # enable double buffering optimization
    convTileSizes: str = ""
    enableConvertToHexagonmem: bool = True  # rewrites memref.alloc/copy to hexagonmem.*
    enableHexagonmemCopyToDMA: bool = False  # rewrites hexmem.copy to memref.dma_*
    enableHexKL: bool = False  # use HexKL to lower matmul and convolutions
    enableHexagonRoutines: bool = True  # use optimized f16/f32 math routines
    # Disable only the affected LLVM BitSimplify subphase when compiling a
    # model function that triggers Hexagon CopyGeneration miscompilation.
    enableHexagonBitCopyGeneration: bool = True
    # Opt-in DSP-local reuse of the existing checked vector-exp scalarization.
    # Does not change exp math or add a host/device dispatch boundary.
    outlineCheckedVectorExp: bool = False
    # Optional reuse of the exact LLVM FP32 vector min/max reduction intrinsic.
    outlineVectorMinMax: bool = False
    # Optional compile-time tradeoff for large functions; does not disable HVX
    # vector insertion or change numerical precision. Default LLVM policy stays on.
    enableHexagonGenInsert: bool = True
    hexKLMode: str = "micro"  # possible options "macro", "micro"
    enableMultiThreading: bool = (
        False  # linalg-generic based multi-threading (FormVirtualThreadsPass)
    )
    enableThreadedDispatch: bool = (
        False  # use tm.exec() (real qurt threads) for SPMD grid dispatch; implicitly enabled when enableMultiThreading=True; when False, tm.exec_serial() is used
    )
    enableSCFThreading: bool = False  # scf based multi-threading
    enableSplitReduction: bool = False  # split-reduction optimization
    enableSplitReduceGeneric: bool = False  # split-reduce-generic optimization
    enableVectorization: bool = True  # enable HVX vectorization.
    enableVTCMTiling: bool = (
        True  # tile linalg-generic and introduce vtcm address-space
    )
    # External VTCM scratch buffer size in bytes per program instance.
    # scratch=0 (default): disabled, each instance allocates VTCM internally.
    # scratch=N (N>0): enables external VTCM flow. The wrapper allocates
    # N * prod(grid) bytes total, passes each instance a memref<Nxi8, 1>
    # {hexagon.scratch} slice. The compiler tiles and plans within N bytes.
    scratch: int = 0
    enableHVXInlining: bool = False
    enableSCFLoopUnroll: bool = False
    enableConversionToFp16: bool = False

    # This option enables 'seeding' of layout conversion ops around conv2d ops.
    # This introduces some builtin.unrealized_conversion_cast ops, that are expected
    # to be eliminated by conv -> hmx.conv2d pass. Till then, this pass could be
    # disabled (or there will be errors).
    enableSeedLayoutConversions: bool = False

    tileSizes: str = ""  # User defined tile sizes - for debugging purposes

    # Separate out constants (dense_resource) into separate shared objects
    lowerConstantsInSeparateSharedObjects: bool = False
    # Opt-in packaging for the torch-mlir multi-object path. Keep private
    # assertion diagnostics in the code module so they do not perturb shared
    # model-weight libraries. Does not remove assertions or change arithmetic.
    keepAssertMessagesInMain: bool = False

    # light weight profiling using hardware instructions
    enableLWP: bool = False

    # By default, loops are instrumented along with the function body.
    # To turn off loop level instrumentation, set it to True.
    disableLWPLoop: bool = False
    # By default, outer loop and nested sibling loops at LWPloopDepth 1 are instrumented.
    # Increasing LWPloopDepth may cause overhead.
    LWPloopDepth: int = 1

    # By default, delete all artifacts pushed to device for this kernel's execution after it runs.
    # This solely applies to execution on the standalone launcher.
    deviceCleanup: bool = True

    def __post_init__(self):
        if type(self.keepAssertMessagesInMain) is not bool:
            raise ValueError("keepAssertMessagesInMain must be a bool")
        if self.keepAssertMessagesInMain and self.lowerConstantsInSeparateSharedObjects is not True:
            raise ValueError(
                "keepAssertMessagesInMain requires lowerConstantsInSeparateSharedObjects"
            )
        # Validate target_artifact
        valid_artifacts = {"ttir", "ttsharedir", "llir", "o", "so"}
        if self.target_artifact not in valid_artifacts:
            raise ValueError(
                f"Invalid target_artifact '{self.target_artifact}'. "
                f"Must be one of: {', '.join(sorted(valid_artifacts))}"
            )

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()
