#ifndef ONNX_MLIR_EXTERNAL_UTIL_H
#define ONNX_MLIR_EXTERNAL_UTIL_H

#include <map>
#include <string>

namespace onnx_mlir {
static const std::string kExecPath = // fallback if not found by getExecPath
    "/usr/local/bin/onnx-mlir";
static const std::string kInstPath = "/usr/local";
static const std::string kOptPath = "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/llvm-project/build-x86/bin/opt";
static const std::string kLlcPath = "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/llvm-project/build-x86/bin/llc";
static const std::string kOnnxmlirPath = "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/Release/bin/onnx-mlir";
static const std::string kCPath = "/usr/bin/cc";
static const std::string kCxxPath = "/usr/bin/c++";
static const std::string kLinkerPath = "/usr/bin/ld";
static const std::string kObjCopyPath = "/usr/bin/objcopy";
static const std::string kArPath = "/usr/bin/ar";
static const std::string kJarPath = "/usr/bin/jar";
static const std::string kDefaultTriple = "x86_64-unknown-linux-gnu";
static const std::string kLrodataScript = R"()";

static const std::map<std::string, std::string> toolPathMap = {
    {"instPath", kInstPath}, {"opt", kOptPath}, {"llc", kLlcPath},
    {"onnx-mlir", kOnnxmlirPath}, {"c", kCPath}, {"cxx", kCxxPath},
    {"linker", kLinkerPath}, {"objcopy", kObjCopyPath}, {"ar", kArPath},
    {"jar", kJarPath}, {"defaultTriple", kDefaultTriple},
    {"lrodataScript", kLrodataScript}};
} // namespace onnx_mlir
#endif
