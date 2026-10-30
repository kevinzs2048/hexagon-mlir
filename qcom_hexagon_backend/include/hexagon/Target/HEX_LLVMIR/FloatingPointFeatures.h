//===- FloatingPointFeatures.h - Explicit v73 HVX FP defaults -----*- C++ -*-===//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef HEXAGON_TARGET_HEX_LLVMIR_FLOATINGPOINTFEATURES_H
#define HEXAGON_TARGET_HEX_LLVMIR_FLOATINGPOINTFEATURES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include <string>

namespace mlir::hexagon {

// LLVM's legacy v68+ default enables QFloat, whose arithmetic/rounding is not
// IEEE f32 arithmetic. In particular, (negative_integer + 0.0) can cease to be
// integral, and a following fptosi changes an INT8 activation code by one.
// On the validated v73 target, choose IEEE HVX instructions unless the caller
// explicitly selects an FP feature policy. This is independent of fast-math:
// clearing LLVM fast flags alone does not disable QFloat instruction selection.
// Other architectures retain their existing policy; v79+ needs a different
// treatment and must not inherit a v73 feature toggle blindly.
inline std::string resolveHexagonFloatingPointFeatures(llvm::StringRef features) {
  llvm::SmallVector<llvm::StringRef> tokens;
  features.split(tokens, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  unsigned hvxVersion = 0;
  bool hvxEnabled = false;
  for (llvm::StringRef token : tokens) {
    token = token.trim();
    if (token == "+hvx-qfloat" || token == "-hvx-qfloat" ||
        token == "+hvx-ieee-fp" || token == "-hvx-ieee-fp")
      return features.str();
    if (token == "-hvx")
      hvxEnabled = false;
    else if (token == "+hvx")
      hvxEnabled = true;
    else if (token.consume_front("+hvxv")) {
      unsigned version = 0;
      if (!token.getAsInteger(10, version)) {
        hvxVersion = version;
        hvxEnabled = true;
      }
    } else if (token.consume_front("-hvxv")) {
      unsigned version = 0;
      if (!token.getAsInteger(10, version) && version == hvxVersion)
        hvxEnabled = false;
    }
  }
  if (hvxEnabled && hvxVersion == 73)
    return features.str() + ",-hvx-qfloat,+hvx-ieee-fp";
  return features.str();
}

// Limit the production use to the verified v73 IEEE instruction policy.
// Explicit QFloat, disabled HVX and other architectures retain the old path.
inline bool hasIEEEV73VectorPolicy(const llvm::Function &function) {
  if (!function.hasFnAttribute("target-features"))
    return false;
  llvm::SmallVector<llvm::StringRef> features;
  function.getFnAttribute("target-features").getValueAsString().split(features, ',');
  bool v73 = false, ieee = false, noQFloat = false;
  for (llvm::StringRef feature : features) {
    feature = feature.trim();
    if (feature == "+hvx-qfloat" || feature == "-hvx-ieee-fp" || feature == "-hvx")
      return false;
    if (feature.starts_with("+hvxv") && feature != "+hvxv73")
      return false;
    if (feature.starts_with("-hvxv"))
      return false;
    v73 |= feature == "+hvxv73";
    ieee |= feature == "+hvx-ieee-fp";
    noQFloat |= feature == "-hvx-qfloat";
  }
  return v73 && ieee && noQFloat;
}

} // namespace mlir::hexagon
#endif
