# REQUIRES: gemmini-simulator, gemmini-resnet18
# RUN: %python %gemmini_repo_root/test/accelerators/gemmini/simulator_run.py --kind resnet18 --model %resnet18_model --repo-root %gemmini_repo_root --workdir %t.dir --tolerance 1e-5

