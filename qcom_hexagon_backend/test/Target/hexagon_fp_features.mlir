// RUN: env HEXAGON_ARCH_VERSION=73 linalg-hexagon-translate %s -no-hexagon-runtime | FileCheck %s --check-prefix=IEEE
// RUN: env HEXAGON_ARCH_VERSION=73 linalg-hexagon-translate %s -no-hexagon-runtime -features=+hvxv73,+hvx-length128b,+hvx-qfloat | FileCheck %s --check-prefix=QFLOAT
// RUN: env HEXAGON_ARCH_VERSION=73 linalg-hexagon-translate %s -no-hexagon-runtime -features=+hvxv79,+hvx-length128b | FileCheck %s --check-prefix=OTHER
// RUN: env HEXAGON_ARCH_VERSION=73 linalg-hexagon-translate %s -no-hexagon-runtime -features=+hvxv73,+hvx-length128b,-hvx-qfloat,+hvx-ieee-fp | FileCheck %s --check-prefix=IEEE
// RUN: env HEXAGON_ARCH_VERSION=73 linalg-hexagon-translate %s -no-hexagon-runtime -features=+hvxv73,-hvx | FileCheck %s --check-prefix=DISABLED
// RUN: env HEXAGON_ARCH_VERSION=73 linalg-hexagon-translate %s -no-hexagon-runtime -features=+hvxv73,+hvx-length128b,-hvx-ieee-fp | FileCheck %s --check-prefix=EXPLICIT
// RUN: env HEXAGON_ARCH_VERSION=73 linalg-hexagon-translate %s -no-hexagon-runtime -features=+hvxv73,+hvxv79,+hvx-length128b | FileCheck %s --check-prefix=LAST
// IEEE: "target-features"="+hvxv73,+hvx-length128b,-hvx-qfloat,+hvx-ieee-fp"
// QFLOAT: "target-features"="+hvxv73,+hvx-length128b,+hvx-qfloat"
// OTHER: "target-features"="+hvxv79,+hvx-length128b"
// DISABLED: "target-features"="+hvxv73,-hvx"
// EXPLICIT: "target-features"="+hvxv73,+hvx-length128b,-hvx-ieee-fp"
// LAST: "target-features"="+hvxv73,+hvxv79,+hvx-length128b"

llvm.func @probe() {
  llvm.return
}
