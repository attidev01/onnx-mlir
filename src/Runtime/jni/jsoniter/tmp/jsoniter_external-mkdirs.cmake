# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter/src/jsoniter_external"
  "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter/src/jsoniter_external-build"
  "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter"
  "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter/tmp"
  "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter/src/jsoniter_external-stamp"
  "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter/maven"
  "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter/src/jsoniter_external-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter/src/jsoniter_external-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/lahcen/toolchains/onnx-mlir-xcompile/onnx-mlir/src/Runtime/jni/jsoniter/src/jsoniter_external-stamp${cfgdir}") # cfgdir has leading slash
endif()
