# REQUIRES: gemmini-mobilenetv2
# RUN: %python %gemmini_tools/model_compile_run.py --model %mobilenetv2_model --label mobilenetv2 --onnx-mlir onnx-mlir --workdir %t.dir --verify-tools %gemmini_tools --min-gemmini-ops 1

