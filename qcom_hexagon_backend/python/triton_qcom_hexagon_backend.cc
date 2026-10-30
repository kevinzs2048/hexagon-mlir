//===-- triton_qcom_hexagon_backend.cc ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "triton_qcom_hexagon_backend_api.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "triton-qcom-hexagon-backend"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace py = pybind11;

namespace {
// Python bindings retain the GIL. This policy belongs to a fresh compilation
// worker: LLVM's target switch is process-wide, not a per-function setting.
std::optional<bool> processGenInsert;
bool translationStarted = false;

void configureProcessGenInsert(bool enabled) {
  if (translationStarted)
    throw py::value_error("Configure process codegen before translation starts");
  if (processGenInsert) {
    if (*processGenInsert != enabled)
      throw py::value_error("Use a fresh process to change the codegen policy");
    return;
  }
  auto &options = llvm::cl::getRegisteredOptions();
  auto found = options.find("hexagon-insert");
  if (found == options.end())
    throw py::value_error("LLVM does not provide the hexagon-insert option");
  if (found->second->getNumOccurrences())
    throw py::value_error("LLVM hexagon-insert was already configured");
  const char *argv[] = {"hexagon-aot-worker", enabled ? "-hexagon-insert=true"
                                                   : "-hexagon-insert=false"};
  std::string error;
  llvm::raw_string_ostream stream(error);
  if (!llvm::cl::ParseCommandLineOptions(2, argv, "", &stream))
    throw py::value_error("Could not configure LLVM codegen: " + stream.str());
  processGenInsert = enabled;
}

void startTranslation(
    const std::unordered_map<std::string, std::string> &options) {
  auto found = options.find("enableHexagonGenInsert");
  if (processGenInsert && found != options.end() && found->second == "False")
    throw py::value_error(
        "Process codegen policy cannot be combined with a function opt-out");
  translationStarted = true;
}
} // namespace

namespace pybind11 {
namespace detail {
template <> struct type_caster<std::vector<char>> {
public:
  PYBIND11_TYPE_CASTER(std::vector<char>, _("bytes"));

  // Conversion from Python to C++
  bool load(handle src, bool) {
    if (!PyBytes_Check(src.ptr()))
      return false;
    PyObject *source = src.ptr();
    value = std::vector<char>(PyBytes_AsString(source),
                              PyBytes_AsString(source) + PyBytes_Size(source));
    return !PyErr_Occurred();
  }

  // Conversion from C++ to Python
  static handle cast(const std::vector<char> &src,
                     return_value_policy /* policy */, handle /* parent */) {
    return PyBytes_FromStringAndSize(src.data(), src.size());
  }
};
} // namespace detail
} // namespace pybind11

void fill_options_map(
    const py::dict &arch_kwargs,
    std::unordered_map<std::string, std::string> &options_map) {
  for (auto it : arch_kwargs)
    options_map[py::cast<std::string>(it.first)] =
        py::cast<std::string>(it.second);
}

void init_triton_hexagon_translation(py::module &m) {
  using ret = py::return_value_policy;

  m.def("configure_process_geninsert", &configureProcessGenInsert,
        py::arg("enabled").noconvert(),
        "Set the official LLVM GenInsert option once in a fresh AOT worker. "
        "The setting is process-wide and cannot change after translation.");

  m.def("get_return_list", [](mlir::ModuleOp &module_op, std::string fName) {
    return hexagon_backend::getReturnList(module_op, fName);
  });

  m.def("extract_func_name_from_mlir_module", [](mlir::ModuleOp &module_op) {
    return hexagon_backend::extractSingleFuncName(module_op);
  });

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    hexagon_backend::loadDialects(context);
  });

  m.def(
      "parse_mlir_module_from_file",
      [](std::string &module_str, mlir::MLIRContext &context) {
        return hexagon_backend::parseMlirFromFile(module_str, context);
      },
      ret::take_ownership);

  m.def(
      "parse_mlir_module_from_str",
      [](std::string &module_str, mlir::MLIRContext &context) {
        return hexagon_backend::parseMlirFromString(module_str, context);
      },
      ret::take_ownership);

  m.def(
      "translate_linalg_to_llvmir",
      [](mlir::ModuleOp &linalg_module, py::dict arch_kwargs) {
        std::unordered_map<std::string, std::string> options_map;
        fill_options_map(arch_kwargs, options_map);
        startTranslation(options_map);
        return hexagon_backend::translateLinalgToLLVMIR(linalg_module,
                                                        options_map);
      },
      ret::take_ownership);

  m.def("reoder_func_args_and_calls_tensor_first",
        [](mlir::ModuleOp &module_op, std::string fName) {
          return hexagon_backend::reorderFuncArgsAndCallsTensorFirst(module_op,
                                                                     fName);
        });

  m.def(
      "translate_linalg_to_obj",
      [](mlir::ModuleOp &linalg_module, py::dict arch_kwargs) {
        std::unordered_map<std::string, std::string> options_map;
        // Goes from a py::dict (arch_kwargs) mapping python strings to python
        // strings to a C++ mapping of strings to strings
        fill_options_map(arch_kwargs, options_map);
        startTranslation(options_map);
        return hexagon_backend::translateLinalgToObj(linalg_module,
                                                     options_map);
      },
      ret::take_ownership);
}

void init_triton_qcom_hexagon_backend(py::module &&m) {
  m.doc() = "Python bindings to the Qualcomm Hexagon Triton backend";
  init_triton_hexagon_translation(m);
}
