import os

import lit.formats
import lit.util
import lit.llvm

config.name = "Open Neural Network Frontend"

use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)
config.suffixes = [".mlir", ".json", ".onnxtext"]
config.test_source_root = os.path.dirname(__file__)

# Some generated test targets invoke llvm-lit directly on the source test tree,
# which bypasses lit.site.cfg.py. Derive workable defaults from the repo/build
# layout so the suite still runs in that mode.
repo_root = os.path.abspath(os.path.join(config.test_source_root, "..", ".."))
default_llvm_tools_dir = os.path.join(repo_root, "llvm-project", "build-x86", "bin")
installed_llvm_tools_dir = os.path.abspath(
    os.path.join(repo_root, "..", "llvm-install", "bin"))
if os.path.isfile(os.path.join(installed_llvm_tools_dir, "FileCheck")):
    default_llvm_tools_dir = installed_llvm_tools_dir

candidate_obj_roots = [
    os.path.join(repo_root, "gemmini_toolchain_build"),
    os.path.join(repo_root, "build"),
    repo_root,
]
default_obj_root = next(
    (path for path in candidate_obj_roots if os.path.isdir(os.path.join(path, "Release", "bin"))),
    repo_root)
config.onnx_mlir_obj_root = getattr(config, "onnx_mlir_obj_root", default_obj_root)
config.onnx_mlir_tools_dir = getattr(
    config, "onnx_mlir_tools_dir", os.path.join(config.onnx_mlir_obj_root, "Release", "bin"))
config.llvm_tools_dir = getattr(config, "llvm_tools_dir", default_llvm_tools_dir)
config.mlir_tools_dir = getattr(config, "mlir_tools_dir", config.llvm_tools_dir)
config.targets_to_build = getattr(config, "targets_to_build", "")
config.enable_stablehlo = getattr(config, "enable_stablehlo", 1)
config.enable_nnpa = getattr(config, "enable_nnpa", 0)
config.enable_gemmini = getattr(config, "enable_gemmini", 1)
config.test_exec_root = os.path.join(config.onnx_mlir_obj_root, "test", "mlir")

if lit.llvm.llvm_config is None:
    lit.llvm.initialize(lit_config, config)
llvm_config = lit.llvm.llvm_config

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.onnx_mlir_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir]
tools = [
    "onnx-mlir",
    "onnx-mlir-opt",
    "mlir-opt",
    "mlir-translate",
    "binary-decoder",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower())
