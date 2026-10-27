# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.20)
project(HexagonConstantPartitionRegression LANGUAGES C CXX)
find_package(MLIR REQUIRED CONFIG)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(HEXAGON_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(HEXAGON_INCLUDE "${HEXAGON_SOURCE_ROOT}/qcom_hexagon_backend/include")
set(GENERATED_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/include")
set(PASS_INCLUDE "${GENERATED_INCLUDE}/hexagon/Conversion/LinalgToLLVM/Passes.h.inc")
file(MAKE_DIRECTORY "${GENERATED_INCLUDE}/hexagon/Conversion/LinalgToLLVM")
find_program(MLIR_TABLEGEN_EXE mlir-tblgen HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
set(TABLEGEN_INCLUDES)
foreach(INCLUDE_DIR IN LISTS MLIR_INCLUDE_DIRS)
  list(APPEND TABLEGEN_INCLUDES -I "${INCLUDE_DIR}")
endforeach()
add_custom_command(OUTPUT "${PASS_INCLUDE}"
  COMMAND "${MLIR_TABLEGEN_EXE}" -gen-pass-decls --name LinalgToLLVM
    ${TABLEGEN_INCLUDES} -I "${HEXAGON_INCLUDE}"
    "${HEXAGON_INCLUDE}/hexagon/Conversion/LinalgToLLVM/Passes.td"
    -o "${PASS_INCLUDE}"
  DEPENDS "${HEXAGON_INCLUDE}/hexagon/Conversion/LinalgToLLVM/Passes.td"
  COMMAND_EXPAND_LISTS VERBATIM)
add_executable(constant-partition-probe constant_partition_probe.cpp "${PASS_INCLUDE}")
target_include_directories(constant-partition-probe PRIVATE
  "${GENERATED_INCLUDE}" "${HEXAGON_INCLUDE}" ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_compile_definitions(constant-partition-probe PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(constant-partition-probe PRIVATE
  MLIRParser MLIRLLVMDialect MLIRPass MLIRTransforms)
