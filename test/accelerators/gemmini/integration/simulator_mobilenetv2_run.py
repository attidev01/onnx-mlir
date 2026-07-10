# REQUIRES: gemmini-simulator, gemmini-mobilenetv2
# RUN: %python %gemmini_repo_root/test/accelerators/gemmini/simulator_run.py --kind mobilenetv2 --model %mobilenetv2_model --repo-root %gemmini_repo_root --workdir %t.dir --tolerance 1e-5

