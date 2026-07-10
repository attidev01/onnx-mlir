# REQUIRES: gemmini-bert-tiny
# RUN: %python %gemmini_tools/model_compile_run.py --model %bert_tiny_model --label bert_tiny --onnx-mlir onnx-mlir --workdir %t.dir --verify-tools %gemmini_tools --min-gemmini-ops 1

