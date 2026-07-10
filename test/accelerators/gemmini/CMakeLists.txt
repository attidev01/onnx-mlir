# SPDX-License-Identifier: Apache-2.0

set(GEMMINI_TEST_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools)
set(GEMMINI_REPO_ROOT ${ONNX_MLIR_SRC_ROOT})
set(GEMMINI_RESNET18_MODEL ${ONNX_MLIR_SRC_ROOT}/models/resnet18-v1-7.onnx)
set(GEMMINI_MOBILENETV2_MODEL ${ONNX_MLIR_SRC_ROOT}/models/mobilenetv2.onnx)
set(GEMMINI_BERT_TINY_MODEL ${ONNX_MLIR_SRC_ROOT}/models/bert-tiny.onnx)

find_program(GEMMINI_FILECHECK_EXECUTABLE FileCheck
  PATHS
  ${LLVM_TOOLS_BINARY_DIR}
  ${LLVM_TOOLS_DIR}
  ${ONNX_MLIR_SRC_ROOT}/llvm-project/build-x86/bin
  NO_DEFAULT_PATH)
if (GEMMINI_FILECHECK_EXECUTABLE)
  get_filename_component(GEMMINI_LLVM_TOOLS_DIR
    ${GEMMINI_FILECHECK_EXECUTABLE} DIRECTORY)
else()
  set(GEMMINI_LLVM_TOOLS_DIR ${LLVM_TOOLS_DIR})
endif()

if (NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
  set_llvm_build_mode()
  string(REPLACE "$<CONFIG>" "${LLVM_BUILD_MODE}" ONNX_MLIR_TOOLS_DIR
    "${ONNX_MLIR_RUNTIME_PATH}")
else()
  string(REPLACE "$<CONFIG>" "${CMAKE_BUILD_TYPE}" ONNX_MLIR_TOOLS_DIR
    "${ONNX_MLIR_RUNTIME_PATH}")
endif()

set(GEMMINI_EXAMPLES_TESTS_DIR ${ONNX_MLIR_SRC_ROOT}/gemmini/tests)
if(NOT DEFINED GEMMINI_RISCV_INSTALL OR GEMMINI_RISCV_INSTALL STREQUAL "")
  set(GEMMINI_RISCV_INSTALL "$ENV{HOME}/riscv-gemmini")
endif()
get_filename_component(GEMMINI_LLVM_INSTALL "${GEMMINI_LLVM_TOOLS_DIR}" DIRECTORY)

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

# Integration test suite — lit discovers all *.py files under integration/.
# Current tests (9 total; REQUIRES-guarded where model files or simulator needed):
#   bert_tiny_compile_run.py           – compile BERT-tiny with --maccel=Gemmini
#   mobilenetv2_compile_run.py         – compile MobileNetV2 with --maccel=Gemmini
#   resnet18_compile_run.py            – compile ResNet-18 with --maccel=Gemmini
#   quantization_roundtrip_run.py      – Gemmini quantization round-trip
#   simulator_matmul_i8_run.py         – quantized MatMulInteger smoke test on Spike
#   simulator_resnet18_run.py          – ResNet-18 on Spike; ONNX-RT correctness 1e-5
#   simulator_mobilenetv2_run.py       – MobileNetV2 on Spike; correctness 1e-5
#   simulator_bert_tiny_run.py         – BERT-tiny on Spike; correctness 1e-5
#   simulator_numerical_correctness.py – aggregate CI runner for all 3 models on Spike
#
# Model-path CMake variables are propagated to lit via lit.site.cfg.py.in:
#   GEMMINI_RESNET18_MODEL    -> %resnet18_model
#   GEMMINI_MOBILENETV2_MODEL -> %mobilenetv2_model
#   GEMMINI_BERT_TINY_MODEL   -> %bert_tiny_model
add_lit_testsuite(check-gemmini-integration
                  "Running Gemmini integration tests"
                  ${CMAKE_CURRENT_SOURCE_DIR}/integration
                  DEPENDS
                  onnx-mlir
                  onnx-mlir-opt
                  FileCheck)
set_target_properties(check-gemmini-integration PROPERTIES FOLDER "Tests")

# check-gemmini-examples: runs gemmini/tests/run_all_tests.sh as a standalone
# CMake target; wired into check-gemmini-integration as a dependency so that
# the full example suite executes automatically during integration testing.
if(EXISTS "${GEMMINI_EXAMPLES_TESTS_DIR}/run_all_tests.sh")
  add_custom_target(check-gemmini-examples
    COMMAND
      ${CMAKE_COMMAND} -E env
        "ONNX_MLIR_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "RISCV_INSTALL=${GEMMINI_RISCV_INSTALL}"
        "LLVM_INSTALL=${GEMMINI_LLVM_INSTALL}"
      bash "${GEMMINI_EXAMPLES_TESTS_DIR}/run_all_tests.sh"
    WORKING_DIRECTORY "${GEMMINI_EXAMPLES_TESTS_DIR}"
    COMMENT "Running Gemmini example test suite (gemmini/tests/run_all_tests.sh)"
    USES_TERMINAL
    DEPENDS onnx-mlir
  )
  set_target_properties(check-gemmini-examples PROPERTIES FOLDER "Tests")
  add_dependencies(check-gemmini-integration check-gemmini-examples)
else()
  message(STATUS
    "Gemmini example tests not found at ${GEMMINI_EXAMPLES_TESTS_DIR} — "
    "check-gemmini-examples target omitted")
endif()

# check-gemmini-zoo: runs run_model_zoo.py against local models.
# Uses --download only when GEMMINI_ZOO_DOWNLOAD=ON; spike only when
# GEMMINI_ZOO_SPIKE=ON.  Not wired into check-gemmini-integration so it
# never runs automatically in CI — trigger manually.
set(GEMMINI_ZOO_MANIFEST "${GEMMINI_TEST_TOOLS_DIR}/model_zoo_manifest.json")
set(GEMMINI_ZOO_REPORT   "${CMAKE_BINARY_DIR}/zoo_report.json")
set(GEMMINI_ZOO_WORKDIR  "${CMAKE_BINARY_DIR}/gemmini_zoo_work")

set(GEMMINI_ZOO_EXTRA_ARGS "")
if(GEMMINI_ZOO_DOWNLOAD)
  list(APPEND GEMMINI_ZOO_EXTRA_ARGS "--download")
endif()
if(GEMMINI_ZOO_SPIKE)
  list(APPEND GEMMINI_ZOO_EXTRA_ARGS "--spike")
endif()

add_custom_target(check-gemmini-zoo
  COMMAND
    ${CMAKE_COMMAND} -E env
      "ONNX_MLIR_BUILD_DIR=${CMAKE_BINARY_DIR}"
      "RISCV_INSTALL=${GEMMINI_RISCV_INSTALL}"
    ${Python3_EXECUTABLE}
      "${GEMMINI_TEST_TOOLS_DIR}/run_model_zoo.py"
      --manifest  "${GEMMINI_ZOO_MANIFEST}"
      --repo-root "${GEMMINI_REPO_ROOT}"
      --onnx-mlir "$<TARGET_FILE:onnx-mlir>"
      --workdir   "${GEMMINI_ZOO_WORKDIR}"
      --report    "${GEMMINI_ZOO_REPORT}"
      ${GEMMINI_ZOO_EXTRA_ARGS}
  COMMENT "Running Gemmini model zoo validation (report: ${GEMMINI_ZOO_REPORT})"
  USES_TERMINAL
  DEPENDS onnx-mlir
)
set_target_properties(check-gemmini-zoo PROPERTIES FOLDER "Tests")
