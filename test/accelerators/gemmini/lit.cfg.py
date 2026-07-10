# SPDX-License-Identifier: Apache-2.0

import os

import lit.formats
import lit.llvm
import lit.util

config.name = "Gemmini Integration"

use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)
config.suffixes = [".py"]
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in", "tools", "simulator_run.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(
    getattr(config, "onnx_mlir_obj_root", os.getcwd()),
    "test",
    "accelerators",
    "gemmini",
)

repo_root = os.path.abspath(os.path.join(config.test_source_root, "..", "..", ".."))
default_llvm_tools_dir = os.path.join(repo_root, "llvm-project", "build-x86", "bin")
installed_llvm_tools_dir = os.path.abspath(
    os.path.join(repo_root, "..", "llvm-install", "bin"))
if os.path.isfile(os.path.join(installed_llvm_tools_dir, "FileCheck")):
    default_llvm_tools_dir = installed_llvm_tools_dir

config.onnx_mlir_tools_dir = getattr(
    config, "onnx_mlir_tools_dir", os.path.join(repo_root, "gemmini_toolchain_build", "Release", "bin")
)
config.llvm_tools_dir = getattr(config, "llvm_tools_dir", default_llvm_tools_dir)
config.gemmini_test_tools_dir = getattr(
    config, "gemmini_test_tools_dir", os.path.join(config.test_source_root, "tools")
)
config.gemmini_repo_root = getattr(config, "gemmini_repo_root", repo_root)
config.gemmini_resnet18_model = getattr(
    config, "gemmini_resnet18_model", os.path.join(repo_root, "models", "resnet18-v1-7.onnx")
)
config.gemmini_mobilenetv2_model = getattr(
    config, "gemmini_mobilenetv2_model", os.path.join(repo_root, "models", "mobilenetv2.onnx")
)
config.gemmini_bert_tiny_model = getattr(
    config, "gemmini_bert_tiny_model", os.path.join(repo_root, "models", "bert-tiny.onnx")
)

if lit.llvm.llvm_config is None:
    lit.llvm.initialize(lit_config, config)
llvm_config = lit.llvm.llvm_config
llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_system_environment(["PYTHONPATH"])

tool_dirs = [config.onnx_mlir_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["onnx-mlir", "onnx-mlir-opt", "FileCheck"], tool_dirs)

config.substitutions.append(("%python", getattr(config, "python_executable", "python3")))
config.substitutions.append(("%gemmini_tools", config.gemmini_test_tools_dir))
config.substitutions.append(("%gemmini_repo_root", config.gemmini_repo_root))
config.substitutions.append(("%resnet18_model", config.gemmini_resnet18_model))
config.substitutions.append(("%mobilenetv2_model", config.gemmini_mobilenetv2_model))
config.substitutions.append(("%bert_tiny_model", config.gemmini_bert_tiny_model))

def add_model_feature(path, feature):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        config.available_features.add(feature)

add_model_feature(config.gemmini_resnet18_model, "gemmini-resnet18")
add_model_feature(config.gemmini_mobilenetv2_model, "gemmini-mobilenetv2")
add_model_feature(config.gemmini_bert_tiny_model, "gemmini-bert-tiny")

riscv_install = os.environ.get("RISCV_INSTALL", os.path.expanduser("~/riscv-gemmini"))
if (
    os.path.exists(os.path.join(riscv_install, "bin", "spike"))
    and (
        os.path.exists(os.path.join(riscv_install, "bin", "pk"))
        or os.path.exists(os.path.join(riscv_install, "riscv64-linux-gnu", "bin", "pk"))
    )
    and os.path.exists(os.path.join(riscv_install, "lib", "libgemmini.so"))
):
    config.available_features.add("gemmini-simulator")

# Feature: gemmini-examples — set when gemmini/tests/run_all_tests.sh exists.
# Lit tests using REQUIRES: gemmini-examples are skipped gracefully when absent.
_examples_dir = getattr(
    config, "gemmini_examples_dir",
    os.path.join(repo_root, "gemmini", "tests")
)
if os.path.isfile(os.path.join(_examples_dir, "run_all_tests.sh")):
    config.available_features.add("gemmini-examples")
config.substitutions.append(("%gemmini_examples_dir", _examples_dir))
config.substitutions.append(("%gemmini_riscv_install",
    getattr(config, "gemmini_riscv_install", riscv_install)))
