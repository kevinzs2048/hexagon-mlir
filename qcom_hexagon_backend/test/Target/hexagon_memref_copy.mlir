// RUN: linalg-hexagon-translate %s -emit=llvmir | FileCheck %s --check-prefix=RUNTIME
// RUN: linalg-hexagon-translate %s -emit=llvmir -no-hexagon-runtime | FileCheck %s --check-prefix=NO-RUNTIME
//
// The copy helper is a compiler runtime dependency, not an SDK libc symbol.
// RUNTIME: define {{.*}} @memrefCopy
// NO-RUNTIME-NOT: define {{.*}} @memrefCopy
llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
llvm.func @copy(%bytes: i64, %source: !llvm.ptr, %destination: !llvm.ptr) {
  llvm.call @memrefCopy(%bytes, %source, %destination) : (i64, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}
