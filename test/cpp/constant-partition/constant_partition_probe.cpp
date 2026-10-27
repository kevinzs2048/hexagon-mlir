// SPDX-License-Identifier: BSD-3-Clause
// Metadata-only model of constant packaging. Splat attributes do not allocate
// the stated large tensor buffers. This probe executes no compiled model.
#include "hexagon/Conversion/LinalgToLLVM/LowerConstantsSeparately.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <sstream>

int testResources() {
  mlir::MLIRContext context;
  context.disableMultithreading();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module attributes {test.partition_fixture = 7 : i64} {
      llvm.mlir.global internal constant @first(dense_resource<first> : tensor<2xi8>) : !llvm.array<2 x i8>
      llvm.mlir.global internal constant @large(dense<1> : tensor<500000000xi8>) : !llvm.array<500000000 x i8>
      llvm.mlir.global internal constant @last(dense_resource<last> : tensor<2xi8>) : !llvm.array<2 x i8>
    }
    {-# dialect_resources: {builtin: {
      first: "0x010000000102",
      last: "0x010000000304"
    }} #-}
  )mlir", &context);
  if (!module) return 10;
  auto pass = std::make_unique<LowerConstantsSeparatelyPass>();
  auto *inspect = pass.get();
  mlir::PassManager manager(&context);
  manager.addPass(std::move(pass));
  if (mlir::failed(manager.run(*module))) return 11;
  auto parts = inspect->getProducedModules();
  if (parts.size() != 3) return 12;
  unsigned resources = 0;
  for (auto part : parts) {
    if (mlir::failed(mlir::verify(part)) || !part->getAttr("test.partition_fixture")) return 13;
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    part.print(stream);
    stream.flush();
    mlir::MLIRContext fresh;
    fresh.disableMultithreading();
    fresh.loadDialect<mlir::LLVM::LLVMDialect>();
    auto restored = mlir::parseSourceString<mlir::ModuleOp>(printed, &fresh);
    if (!restored || mlir::failed(mlir::verify(*restored))) return 14;
    for (auto global : restored->getOps<mlir::LLVM::GlobalOp>()) {
      auto attr = mlir::dyn_cast_or_null<mlir::DenseResourceElementsAttr>(global.getValueAttr());
      if (!attr) continue;
      const auto *blob = attr.getRawHandle().getBlob();
      if (!blob || blob->getData().size() != 2) return 15;
      auto data = blob->getData();
      if (global.getSymName() == "first") {
        if (data[0] != 1 || data[1] != 2) return 16;
      } else if (global.getSymName() == "last") {
        if (data[0] != 3 || data[1] != 4) return 17;
      } else return 18;
      ++resources;
    }
  }
  if (resources != 2) return 19;
  for (auto part : parts) part.erase();
  llvm::outs() << "{\"resource_round_trip\":true,\"resource_count\":2,\"partition_count\":3}\n";
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--resource") return testResources();
  mlir::MLIRContext context;
  context.disableMultithreading();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  std::ostringstream text;
  text << "module attributes {test.partition_fixture = 7 : i64} {\n";
  for (int i = 1; i < argc; ++i) {
    uint64_t bytes = std::stoull(argv[i]);
    if (!bytes || bytes > (uint64_t{1} << 40)) return 2;
    text << "llvm.mlir.global internal constant @weight" << i
         << "(dense<1> : tensor<" << bytes << "xi8>) : !llvm.array<"
         << bytes << " x i8>\n";
  }
  text << "}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text.str(), &context);
  if (!module) return 3;
  auto pass = std::make_unique<LowerConstantsSeparatelyPass>();
  auto *inspect = pass.get();
  mlir::PassManager manager(&context);
  manager.addPass(std::move(pass));
  if (mlir::failed(manager.run(*module))) return 4;
  auto parts = inspect->getProducedModules();
  llvm::outs() << "{\"modules\":[";
  bool next = false;
  for (auto part : parts) {
    if (next) llvm::outs() << ",";
    next = true;
    llvm::outs() << "{\"sizes\":[";
    bool comma = false;
    for (auto global : part.getOps<mlir::LLVM::GlobalOp>()) {
      if (comma) llvm::outs() << ",";
      comma = true;
      llvm::outs() << mlir::cast<mlir::LLVM::LLVMArrayType>(global.getType()).getNumElements();
      auto value = mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(global.getValueAttr());
      if (!value || !value.isSplat() || value.getSplatValue<llvm::APInt>() != 1) return 5;
    }
    llvm::outs() << "],\"module_attribute_preserved\":"
                 << (part->getAttr("test.partition_fixture") ? "true" : "false") << "}";
  }
  llvm::outs() << "]}\n";
  for (auto part : parts) part.erase();
  return 0;
}
