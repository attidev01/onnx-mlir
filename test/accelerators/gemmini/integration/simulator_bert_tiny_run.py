# REQUIRES: gemmini-simulator, gemmini-bert-tiny
# RUN: %python %gemmini_repo_root/test/accelerators/gemmini/simulator_run.py --kind bert_tiny --model %bert_tiny_model --repo-root %gemmini_repo_root --workdir %t.dir --tolerance 1e-5 --seq-len 8

