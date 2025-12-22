../disc-opt ./parallel-loop-tiling-inbound-check.mlir \
    -pass-pipeline='builtin.module(func.func(disc-parallel-loop-tiling{parallel-loop-tile-sizes=1,4 with-inbound-check=true}))' \
    -split-input-file -o parallel-loop-tiling.mlir

