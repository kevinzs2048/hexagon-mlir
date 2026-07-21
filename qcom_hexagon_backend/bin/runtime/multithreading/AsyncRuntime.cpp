//===- AsyncRuntime.cpp - Async runtime reference implementation ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements basic Async runtime API for supporting Async dialect
// to LLVM dialect lowering.
//
//===----------------------------------------------------------------------===//

#include "AsyncRuntime.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <new>
#include <vector>

#include "llvm/ADT/StringMap.h"

using namespace mlir::runtime;

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace runtime {
namespace {

// Forward declare class defined below.
class RefCounted;

// -------------------------------------------------------------------------- //
// AsyncRuntime orchestrates all async operations and Async runtime API is built
// on top of the default runtime instance.
// -------------------------------------------------------------------------- //

class AsyncRuntime {
public:
  AsyncRuntime() : numRefCountedObjects(0) {}

  ~AsyncRuntime() {
    assert(getNumRefCountedObjects() == 0 &&
           "all ref counted objects must be destroyed");
  }

  int64_t getNumRefCountedObjects() {
    return numRefCountedObjects.load(std::memory_order_relaxed);
  }

private:
  friend class RefCounted;

  // Count the total number of reference counted objects in this instance
  // of an AsyncRuntime. For debugging purposes only.
  void addNumRefCountedObjects() {
    numRefCountedObjects.fetch_add(1, std::memory_order_relaxed);
  }
  void dropNumRefCountedObjects() {
    numRefCountedObjects.fetch_sub(1, std::memory_order_relaxed);
  }

  std::atomic<int64_t> numRefCountedObjects;
};

// -------------------------------------------------------------------------- //
// A state of the async runtime value (token, value or group).
// -------------------------------------------------------------------------- //

class State {
public:
  enum StateEnum : int8_t {
    // The underlying value is not yet available for consumption.
    kUnavailable = 0,
    // The underlying value is available for consumption. This state can not
    // transition to any other state.
    kAvailable = 1,
    // This underlying value is available and contains an error. This state can
    // not transition to any other state.
    kError = 2,
  };

  /* implicit */ State(StateEnum s) : state(s) {}
  /* implicit */ operator StateEnum() { return state; }

  bool isUnavailable() const { return state == kUnavailable; }
  bool isAvailable() const { return state == kAvailable; }
  bool isError() const { return state == kError; }
  bool isAvailableOrError() const { return isAvailable() || isError(); }

  const char *debug() const {
    switch (state) {
    case kUnavailable:
      return "unavailable";
    case kAvailable:
      return "available";
    case kError:
      return "error";
    }
  }

private:
  StateEnum state;
};

// -------------------------------------------------------------------------- //
// A base class for all reference counted objects created by the async runtime.
// -------------------------------------------------------------------------- //

class RefCounted {
public:
  RefCounted(AsyncRuntime *runtime, int64_t refCount = 1)
      : runtime(runtime), refCount(refCount) {
    runtime->addNumRefCountedObjects();
  }

  virtual ~RefCounted() {
    assert(refCount.load() == 0 && "reference count must be zero");
    runtime->dropNumRefCountedObjects();
  }

  RefCounted(const RefCounted &) = delete;
  RefCounted &operator=(const RefCounted &) = delete;

  void addRef(int64_t count = 1) { refCount.fetch_add(count); }

  void dropRef(int64_t count = 1) {
    int64_t previous = refCount.fetch_sub(count);
    assert(previous >= count && "reference count should not go below zero");
    if (previous == count)
      destroy();
  }

protected:
  virtual void destroy() { delete this; }

private:
  AsyncRuntime *runtime;
  std::atomic<int64_t> refCount;
};

} // namespace

// Returns the default per-process instance of an async runtime.
static AsyncRuntime *getDefaultAsyncRuntime() {
  alignas(AsyncRuntime) static unsigned char storage[sizeof(AsyncRuntime)];
  static bool initialized = false;
  if (!initialized) {
    new (storage) AsyncRuntime();
    initialized = true;
  }
  return reinterpret_cast<AsyncRuntime *>(storage);
}

// Async token provides a mechanism to signal asynchronous operation completion.
struct AsyncToken : public RefCounted {
  // AsyncToken created with a reference count of 2 because it will be returned
  // to the `async.execute` caller and also will be later on emplaced by the
  // asynchronously executed task. If the caller immediately will drop its
  // reference we must ensure that the token will be alive until the
  // asynchronous operation is completed.
  AsyncToken(AsyncRuntime *runtime)
      : RefCounted(runtime, /*refCount=*/2), state(State::kUnavailable) {}

  std::atomic<State::StateEnum> state;

};

// Async value provides a mechanism to access the result of asynchronous
// operations. It owns the storage that is used to store/load the value of the
// underlying type, and a flag to signal if the value is ready or not.
struct AsyncValue : public RefCounted {
  // AsyncValue similar to an AsyncToken created with a reference count of 2.
  AsyncValue(AsyncRuntime *runtime, int64_t size)
      : RefCounted(runtime, /*refCount=*/2), state(State::kUnavailable),
        storage(size) {}

  std::atomic<State::StateEnum> state;

  // Use vector of bytes to store async value payload.
  std::vector<std::byte> storage;

};

// Async group provides a mechanism to group together multiple async tokens or
// values to await on all of them together (wait for the completion of all
// tokens or values added to the group).
struct AsyncGroup : public RefCounted {
  AsyncGroup(AsyncRuntime *runtime, int64_t size)
      : RefCounted(runtime), pendingTokens(size), numErrors(0), rank(0) {}

  std::atomic<int> pendingTokens;
  std::atomic<int> numErrors;
  std::atomic<int> rank;

};

// Adds references to reference counted runtime object.
extern "C" void mlirAsyncRuntimeAddRef(RefCountedObjPtr ptr, int64_t count) {
  RefCounted *refCounted = static_cast<RefCounted *>(ptr);
  refCounted->addRef(count);
}

// Drops references from reference counted runtime object.
extern "C" void mlirAsyncRuntimeDropRef(RefCountedObjPtr ptr, int64_t count) {
  RefCounted *refCounted = static_cast<RefCounted *>(ptr);
  refCounted->dropRef(count);
}

// Creates a new `async.token` in not-ready state.
extern "C" AsyncToken *mlirAsyncRuntimeCreateToken() {
  AsyncToken *token = new AsyncToken(getDefaultAsyncRuntime());
  return token;
}

// Creates a new `async.value` in not-ready state.
extern "C" AsyncValue *mlirAsyncRuntimeCreateValue(int64_t size) {
  AsyncValue *value = new AsyncValue(getDefaultAsyncRuntime(), size);
  return value;
}

// Create a new `async.group` in empty state.
extern "C" AsyncGroup *mlirAsyncRuntimeCreateGroup(int64_t size) {
  AsyncGroup *group = new AsyncGroup(getDefaultAsyncRuntime(), size);
  return group;
}

extern "C" int64_t mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token,
                                                   AsyncGroup *group) {
  // Get the rank of the token inside the group before we drop the reference.
  int rank = group->rank.fetch_add(1);

  mlirAsyncRuntimeAwaitToken(token);
  if (State(token->state).isError())
    group->numErrors.fetch_add(1);

  assert(group->pendingTokens > 0 && "wrong group size");
  group->pendingTokens.fetch_sub(1);

  return rank;
}

// Switches `async.token` to available or error state (terminatl state) and runs
// all awaiters.
static void setTokenState(AsyncToken *token, State state) {
  assert(state.isAvailableOrError() && "must be terminal state");
  assert(State(token->state).isUnavailable() && "token must be unavailable");

  token->state = state;

  // Async tokens created with a ref count `2` to keep token alive until the
  // async task completes. Drop this reference explicitly when token emplaced.
  token->dropRef();
}

static void setValueState(AsyncValue *value, State state) {
  assert(state.isAvailableOrError() && "must be terminal state");
  assert(State(value->state).isUnavailable() && "value must be unavailable");

  value->state = state;

  // Async values created with a ref count `2` to keep value alive until the
  // async task completes. Drop this reference explicitly when value emplaced.
  value->dropRef();
}

extern "C" void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  setTokenState(token, State::kAvailable);
}

extern "C" void mlirAsyncRuntimeEmplaceValue(AsyncValue *value) {
  setValueState(value, State::kAvailable);
}

extern "C" void mlirAsyncRuntimeSetTokenError(AsyncToken *token) {
  setTokenState(token, State::kError);
}

extern "C" void mlirAsyncRuntimeSetValueError(AsyncValue *value) {
  setValueState(value, State::kError);
}

extern "C" bool mlirAsyncRuntimeIsTokenError(AsyncToken *token) {
  return State(token->state).isError();
}

extern "C" bool mlirAsyncRuntimeIsValueError(AsyncValue *value) {
  return State(value->state).isError();
}

extern "C" bool mlirAsyncRuntimeIsGroupError(AsyncGroup *group) {
  return group->numErrors.load() > 0;
}

extern "C" void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  while (!State(token->state).isAvailableOrError()) {
  }
}

extern "C" void mlirAsyncRuntimeAwaitValue(AsyncValue *value) {
  while (!State(value->state).isAvailableOrError()) {
  }
}

extern "C" void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  while (group->pendingTokens != 0) {
  }
}

// Returns a pointer to the storage owned by the async value.
extern "C" ValueStorage mlirAsyncRuntimeGetValueStorage(AsyncValue *value) {
  assert(!State(value->state).isError() && "unexpected error state");
  return value->storage.data();
}

extern "C" void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
  (*resume)(handle);
}

extern "C" void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token,
                                                     CoroHandle handle,
                                                     CoroResume resume) {
  mlirAsyncRuntimeAwaitToken(token);
  (*resume)(handle);
}

extern "C" void mlirAsyncRuntimeAwaitValueAndExecute(AsyncValue *value,
                                                     CoroHandle handle,
                                                     CoroResume resume) {
  mlirAsyncRuntimeAwaitValue(value);
  (*resume)(handle);
}

extern "C" void mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group,
                                                          CoroHandle handle,
                                                          CoroResume resume) {
  mlirAsyncRuntimeAwaitAllInGroup(group);
  (*resume)(handle);
}

extern "C" int64_t mlirAsyncRuntimGetNumWorkerThreads() {
  return 1;
}

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" void mlirAsyncRuntimePrintCurrentThreadId() {
}

//===----------------------------------------------------------------------===//
// MLIR ExecutionEngine dynamic library integration.
//===----------------------------------------------------------------------===//

// Visual Studio had a bug that fails to compile nested generic lambdas
// inside an `extern "C"` function.
//   https://developercommunity.visualstudio.com/content/problem/475494/clexe-error-with-lambda-inside-function-templates.html
// The bug is fixed in VS2019 16.1. Separating the declaration and definition is
// a work around for older versions of Visual Studio.
// NOLINTNEXTLINE(*-identifier-naming): externally called.
extern "C" MLIR_ASYNC_RUNTIME_EXPORT void
__mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols);

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void __mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols) {
  auto exportSymbol = [&](llvm::StringRef name, auto ptr) {
    assert(exportSymbols.count(name) == 0 && "symbol already exists");
    exportSymbols[name] = reinterpret_cast<void *>(ptr);
  };

  exportSymbol("mlirAsyncRuntimeAddRef",
               &mlir::runtime::mlirAsyncRuntimeAddRef);
  exportSymbol("mlirAsyncRuntimeDropRef",
               &mlir::runtime::mlirAsyncRuntimeDropRef);
  exportSymbol("mlirAsyncRuntimeExecute",
               &mlir::runtime::mlirAsyncRuntimeExecute);
  exportSymbol("mlirAsyncRuntimeGetValueStorage",
               &mlir::runtime::mlirAsyncRuntimeGetValueStorage);
  exportSymbol("mlirAsyncRuntimeCreateToken",
               &mlir::runtime::mlirAsyncRuntimeCreateToken);
  exportSymbol("mlirAsyncRuntimeCreateValue",
               &mlir::runtime::mlirAsyncRuntimeCreateValue);
  exportSymbol("mlirAsyncRuntimeEmplaceToken",
               &mlir::runtime::mlirAsyncRuntimeEmplaceToken);
  exportSymbol("mlirAsyncRuntimeEmplaceValue",
               &mlir::runtime::mlirAsyncRuntimeEmplaceValue);
  exportSymbol("mlirAsyncRuntimeSetTokenError",
               &mlir::runtime::mlirAsyncRuntimeSetTokenError);
  exportSymbol("mlirAsyncRuntimeSetValueError",
               &mlir::runtime::mlirAsyncRuntimeSetValueError);
  exportSymbol("mlirAsyncRuntimeIsTokenError",
               &mlir::runtime::mlirAsyncRuntimeIsTokenError);
  exportSymbol("mlirAsyncRuntimeIsValueError",
               &mlir::runtime::mlirAsyncRuntimeIsValueError);
  exportSymbol("mlirAsyncRuntimeIsGroupError",
               &mlir::runtime::mlirAsyncRuntimeIsGroupError);
  exportSymbol("mlirAsyncRuntimeAwaitToken",
               &mlir::runtime::mlirAsyncRuntimeAwaitToken);
  exportSymbol("mlirAsyncRuntimeAwaitValue",
               &mlir::runtime::mlirAsyncRuntimeAwaitValue);
  exportSymbol("mlirAsyncRuntimeAwaitTokenAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitTokenAndExecute);
  exportSymbol("mlirAsyncRuntimeAwaitValueAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitValueAndExecute);
  exportSymbol("mlirAsyncRuntimeCreateGroup",
               &mlir::runtime::mlirAsyncRuntimeCreateGroup);
  exportSymbol("mlirAsyncRuntimeAddTokenToGroup",
               &mlir::runtime::mlirAsyncRuntimeAddTokenToGroup);
  exportSymbol("mlirAsyncRuntimeAwaitAllInGroup",
               &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroup);
  exportSymbol("mlirAsyncRuntimeAwaitAllInGroupAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroupAndExecute);
  exportSymbol("mlirAsyncRuntimGetNumWorkerThreads",
               &mlir::runtime::mlirAsyncRuntimGetNumWorkerThreads);
  exportSymbol("mlirAsyncRuntimePrintCurrentThreadId",
               &mlir::runtime::mlirAsyncRuntimePrintCurrentThreadId);
}

// NOLINTNEXTLINE(*-identifier-naming): externally called.
extern "C" MLIR_ASYNC_RUNTIME_EXPORT void __mlir_execution_engine_destroy() {
}

} // namespace runtime
} // namespace mlir
