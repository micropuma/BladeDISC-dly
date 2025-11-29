# !/bin/bash

# =================================== tf dialect -> mhlo dialect ===================================
./tf-opt -tf-standard-pipeline tutorial.mlir -o tutorial_tf_dialect.mlir

# 强制静态rank以简化后续传递中的shape推理
./disc-opt -disc-tf-revise-args-for-static-rank \
            -disc-lower-tf \
            tutorial_tf_dialect.mlir \
            -o tutorial_tf2hlo_snapshot_0.mlir

# 做实际的type shape inference
./tf-opt  -tf-shape-inference \
           -xla-legalize-tf-types \
           "-xla-legalize-tf=allow-partial-conversion=true" \
           -canonicalize \
           -tf-shape-inference \
           "-xla-legalize-tf=allow-partial-conversion=false" \
           tutorial_tf2hlo_snapshot_0.mlir \
           -o tutorial_mhlo.mlir

TF_CPP_VMODULE=disc_compiler=1 ./disc_compiler_main tutorial_mhlo.mlir result 2>pass_pipeline.log