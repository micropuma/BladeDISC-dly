======== BEGIN After TF2HLO =========
// 到这一阶段，重点是用 shape dialect 显式建模“动态 shape + broadcast 规则”
module {
  func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x10xf32> attributes {tf.entry_function = {input_placements = "gpu", inputs = "input.1_", output_placements = "gpu", outputs = "8"}} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = shape.const_shape [10] : tensor<1xindex>
    %2 = mhlo.constant dense_resource<__elided__> : tensor<10x10xf32>
    %3 = mhlo.constant dense_resource<__elided__> : tensor<10x10xf32>
    %4 = mhlo.constant dense_resource<__elided__> : tensor<10xf32>
    %5 = mhlo.constant dense_resource<__elided__> : tensor<10xf32>
    %6 = "mhlo.dot"(%arg0, %3) : (tensor<?x?xf32>, tensor<10x10xf32>) -> tensor<?x10xf32>
    %7 = shape.shape_of %6 : tensor<?x10xf32> -> tensor<2xindex>   // 计算广播后shape
    %8 = shape.broadcast %7, %1 : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
    %9 = "mhlo.dynamic_broadcast_in_dim"(%6, %8) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
    %10 = "mhlo.dynamic_broadcast_in_dim"(%4, %8) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
    
    %11 = mhlo.add %9, %10 : tensor<?x10xf32>
    
    %12 = shape.shape_of %11 : tensor<?x10xf32> -> tensor<2xindex>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%0, %12) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
    %14 = mhlo.maximum %11, %13 : tensor<?x10xf32>
    %15 = "mhlo.dot"(%14, %2) : (tensor<?x10xf32>, tensor<10x10xf32>) -> tensor<?x10xf32>
    
    %16 = shape.shape_of %15 : tensor<?x10xf32> -> tensor<2xindex>
    %17 = shape.broadcast %16, %1 : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
    %18 = "mhlo.dynamic_broadcast_in_dim"(%15, %17) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
    %19 = "mhlo.dynamic_broadcast_in_dim"(%5, %17) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
    %20 = mhlo.add %18, %19 : tensor<?x10xf32>
    
    %21 = shape.shape_of %20 : tensor<?x10xf32> -> tensor<2xindex>
    %22 = "mhlo.dynamic_broadcast_in_dim"(%0, %21) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
    %23 = mhlo.maximum %20, %22 : tensor<?x10xf32>
    return %23 : tensor<?x10xf32>
  }
}

// -----// IR Dump After ConvertShapeToStandardPass (disc-convert-shape-to-std) //----- //
// 这一阶段，重点是将 shape dialect 转换为标准 dialect
%7 = shape.shape_of %6 : tensor<?x10xf32> -> tensor<2xindex>   // 计算广播后shape
下降到如下代码：
%c0 = arith.constant 0 : index
%dim = tensor.dim %5, %c0 : tensor<?x10xf32>
%c10_0 = arith.constant 10 : index
%from_elements_1 = tensor.from_elements %dim, %c10_0 : tensor<2xindex>
%c0_2 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c0_3 = arith.constant 0 : index
%extracted = tensor.extract %from_elements_1[%c0_3] : tensor<2xindex>
%6 = arith.cmpi eq, %extracted, %c1 : index
%7 = arith.select %6, %c1, %extracted : index
%c1_4 = arith.constant 1 : index
%c1_5 = arith.constant 1 : index
%extracted_6 = tensor.extract %from_elements_1[%c1_5] : tensor<2xindex>
%8 = arith.cmpi eq, %extracted_6, %c1_4 : index
%9 = arith.select %8, %c1_4, %extracted_6 : index
%c0_7 = arith.constant 0 : index
%extracted_8 = tensor.extract %cast[%c0_7] : tensor<1xindex>
%10 = arith.cmpi eq, %extracted_8, %c1_4 : index
%11 = arith.select %10, %9, %extracted_8 : index
%from_elements_9 = tensor.from_elements %7, %11 : tensor<2xindex>

// 做CSE，这里引入tie_shape来显示将shape信息绑定到tensor上
%29 = mhlo.add %26, %28 : tensor<?x10xf32>
%30 = "disc_shape.tie_shape"(%29, %dim, %c10) : (tensor<?x10xf32>, index, index) -> tensor<?x10xf32>
即显示绑定%30的shape是[%dim, %c10]

// -----// IR Dump After DiscShapeOptimizationPass (disc-shape-optimization) //----- //
// TODO(leon):重点dump这个pass  
module {
  func.func @main(%arg0: tensor<?x10xf32, [@S0, @C10]>) -> tensor<?x10xf32, [@S0, @C10]> attributes {tf.entry_function = {input_placements = "gpu", inputs = "input.1_", output_placements = "gpu", outputs = "8"}} {
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %0 = mhlo.constant dense<[-0.102631167, 0.0970462784, 0.215114668, 0.297264218, -0.00310139358, 0.0898892953, 0.275285691, -0.295786381, -0.0738286674, 0.297302932]> : tensor<10xf32>
    %1 = mhlo.constant dense<[0.0913017764, 0.0936649814, 0.0520955399, -0.11318139, 0.211301848, -0.235358447, -0.0122448327, -0.309040517, 0.193537757, 0.0116051855]> : tensor<10xf32>
    %2 = mhlo.constant dense_resource<__elided__> : tensor<10x10xf32>
    %3 = mhlo.constant dense_resource<__elided__> : tensor<10x10xf32>
    %4 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %dim = tensor.dim %arg0, %c0 : tensor<?x10xf32, [@S0, @C10]>
    %5 = "mhlo.dot"(%arg0, %2) : (tensor<?x10xf32, [@S0, @C10]>, tensor<10x10xf32>) -> tensor<?x10xf32, [@S0, @C10]>
    %from_elements = tensor.from_elements %dim, %c10 : tensor<2xindex>
    %6 = "mhlo.dynamic_broadcast_in_dim"(%5, %from_elements) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x10xf32, [@S0, @C10]>, tensor<2xindex>) -> tensor<?x10xf32, [@S0, @C10]>
    %7 = "mhlo.dynamic_broadcast_in_dim"(%1, %from_elements) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32, [@S0, @C10]>
    %8 = mhlo.add %6, %7 : tensor<?x10xf32, [@S0, @C10]>
    %9 = "mhlo.dynamic_broadcast_in_dim"(%4, %from_elements) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32, [@S0, @C10]>
    %10 = mhlo.maximum %8, %9 : tensor<?x10xf32, [@S0, @C10]>
    %11 = "mhlo.dot"(%10, %3) : (tensor<?x10xf32, [@S0, @C10]>, tensor<10x10xf32>) -> tensor<?x10xf32, [@S0, @C10]>
    %12 = "mhlo.dynamic_broadcast_in_dim"(%11, %from_elements) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x10xf32, [@S0, @C10]>, tensor<2xindex>) -> tensor<?x10xf32, [@S0, @C10]>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%0, %from_elements) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32, [@S0, @C10]>
    %14 = mhlo.add %12, %13 : tensor<?x10xf32, [@S0, @C10]>
    %15 = mhlo.maximum %14, %9 : tensor<?x10xf32, [@S0, @C10]>
    return %15 : tensor<?x10xf32, [@S0, @C10]>
  }
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = true, knownNonSizeZero = true, sym_name = "C10", value = 10 : i64} : () -> ()
  func.func @shape_constraint_graph() {
    return
  }
}

// -----// IR Dump After PlaceOpsPass (mhlo-place-ops) //----- //
module {
  func.func @main(%arg0: tensor<?x10xf32, [@S0, @C10]>) -> tensor<?x10xf32, [@S0, @C10]> attributes {tf.entry_function = {input_placements = "gpu", inputs = "input.1_", output_placements = "gpu", outputs = "8"}} {
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %0 = mhlo.constant {disc.device = "gpu"} dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant {disc.device = "gpu"} dense_resource<__elided__> : tensor<10x10xf32>
    %2 = mhlo.constant {disc.device = "gpu"} dense_resource<__elided__> : tensor<10x10xf32>
    %3 = mhlo.constant {disc.device = "gpu"} dense<[0.0913017764, 0.0936649814, 0.0520955399, -0.11318139, 0.211301848, -0.235358447, -0.0122448327, -0.309040517, 0.193537757, 0.0116051855]> : tensor<10xf32>
    %4 = mhlo.constant {disc.device = "gpu"} dense<[-0.102631167, 0.0970462784, 0.215114668, 0.297264218, -0.00310139358, 0.0898892953, 0.275285691, -0.295786381, -0.0738286674, 0.297302932]> : tensor<10xf32>
    %dim = tensor.dim %arg0, %c0 : tensor<?x10xf32, [@S0, @C10]>
    %5 = "mhlo.dot_general"(%arg0, %2) {disc.device = "gpu", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x10xf32, [@S0, @C10]>, tensor<10x10xf32>) -> tensor<?x10xf32, [@S0, @C10]>
    %from_elements = tensor.from_elements %dim, %c10 {disc.shape_op = true} : tensor<2xindex>
    %6 = "mhlo.dynamic_broadcast_in_dim"(%3, %from_elements) {broadcast_dimensions = dense<1> : tensor<1xi64>, disc.device = "gpu"} : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32, [@S0, @C10]>
    %7 = mhlo.add %5, %6 {disc.device = "gpu"} : tensor<?x10xf32, [@S0, @C10]>
    %8 = "mhlo.dynamic_broadcast_in_dim"(%0, %from_elements) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "gpu"} : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32, [@S0, @C10]>
    %9 = mhlo.maximum %7, %8 {disc.device = "gpu"} : tensor<?x10xf32, [@S0, @C10]>
    %10 = "mhlo.dot_general"(%9, %1) {disc.device = "gpu", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x10xf32, [@S0, @C10]>, tensor<10x10xf32>) -> tensor<?x10xf32, [@S0, @C10]>
    %11 = "mhlo.dynamic_broadcast_in_dim"(%4, %from_elements) {broadcast_dimensions = dense<1> : tensor<1xi64>, disc.device = "gpu"} : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32, [@S0, @C10]>
    %12 = mhlo.add %10, %11 {disc.device = "gpu"} : tensor<?x10xf32, [@S0, @C10]>
    %13 = mhlo.maximum %12, %8 {disc.device = "gpu"} : tensor<?x10xf32, [@S0, @C10]>
    return %13 : tensor<?x10xf32, [@S0, @C10]>
  }
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = true, knownNonSizeZero = true, sym_name = "C10", value = 10 : i64} : () -> ()
  func.func @shape_constraint_graph() {
    return
  }
}

// -----// IR Dump After HLO2LHLO (canonicalize) //----- //
// 正式进入memref world
func.func @main(%arg0: memref<?x10xf32>) -> memref<?x10xf32> attributes {tf.entry_function = {input_placements = "gpu", inputs = "input.1_", output_placements = "gpu", outputs = "8"}} {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<f32>
  "lmhlo.constant"(%alloc) {disc.device = "gpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
  %alloc_0 = memref.alloc() : memref<10x10xf32>
  "lmhlo.constant"(%alloc_0) {disc.device = "gpu", value = dense_resource<__elided__> : tensor<10x10xf32>} : (memref<10x10xf32>) -> ()
  %alloc_1 = memref.alloc() : memref<10x10xf32>
  "lmhlo.constant"(%alloc_1) {disc.device = "gpu", value = dense_resource<__elided__> : tensor<10x10xf32>} : (memref<10x10xf32>) -> ()
  %alloc_2 = memref.alloc() : memref<10xf32>
  "lmhlo.constant"(%alloc_2) {disc.device = "gpu", value = dense<[0.0913017764, 0.0936649814, 0.0520955399, -0.11318139, 0.211301848, -0.235358447, -0.0122448327, -0.309040517, 0.193537757, 0.0116051855]> : tensor<10xf32>} : (memref<10xf32>) -> ()
  %alloc_3 = memref.alloc() : memref<10xf32>
  "lmhlo.constant"(%alloc_3) {disc.device = "gpu", value = dense<[-0.102631167, 0.0970462784, 0.215114668, 0.297264218, -0.00310139358, 0.0898892953, 0.275285691, -0.295786381, -0.0738286674, 0.297302932]> : tensor<10xf32>} : (memref<10xf32>) -> ()
  %dim = memref.dim %arg0, %c0 : memref<?x10xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %alloc_4 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.dot_general"(%reinterpret_cast, %alloc_1, %alloc_4) {disc.device = "gpu", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x10xf32>, memref<10x10xf32>, memref<?x10xf32>) -> ()
  %reinterpret_cast_5 = memref.reinterpret_cast %alloc_4 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %from_elements = tensor.from_elements %dim, %c10 {disc.shape_op = true} : tensor<2xindex>
  %0 = bufferization.to_memref %from_elements : memref<2xindex>
  %alloc_6 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.dynamic_broadcast_in_dim"(%alloc_2, %0, %alloc_6) {broadcast_dimensions = dense<1> : tensor<1xi64>, disc.device = "gpu"} : (memref<10xf32>, memref<2xindex>, memref<?x10xf32>) -> ()
  %reinterpret_cast_7 = memref.reinterpret_cast %alloc_6 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %alloc_8 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.add"(%reinterpret_cast_5, %reinterpret_cast_7, %alloc_8) {disc.device = "gpu"} : (memref<?x10xf32>, memref<?x10xf32>, memref<?x10xf32>) -> ()
  %reinterpret_cast_9 = memref.reinterpret_cast %alloc_8 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %alloc_10 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.dynamic_broadcast_in_dim"(%alloc, %0, %alloc_10) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "gpu"} : (memref<f32>, memref<2xindex>, memref<?x10xf32>) -> ()
  %reinterpret_cast_11 = memref.reinterpret_cast %alloc_10 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %alloc_12 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.maximum"(%reinterpret_cast_9, %reinterpret_cast_11, %alloc_12) {disc.device = "gpu"} : (memref<?x10xf32>, memref<?x10xf32>, memref<?x10xf32>) -> ()
  %reinterpret_cast_13 = memref.reinterpret_cast %alloc_12 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %alloc_14 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.dot_general"(%reinterpret_cast_13, %alloc_0, %alloc_14) {disc.device = "gpu", dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x10xf32>, memref<10x10xf32>, memref<?x10xf32>) -> ()
  %reinterpret_cast_15 = memref.reinterpret_cast %alloc_14 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %alloc_16 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.dynamic_broadcast_in_dim"(%alloc_3, %0, %alloc_16) {broadcast_dimensions = dense<1> : tensor<1xi64>, disc.device = "gpu"} : (memref<10xf32>, memref<2xindex>, memref<?x10xf32>) -> ()
  %reinterpret_cast_17 = memref.reinterpret_cast %alloc_16 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %alloc_18 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.add"(%reinterpret_cast_15, %reinterpret_cast_17, %alloc_18) {disc.device = "gpu"} : (memref<?x10xf32>, memref<?x10xf32>, memref<?x10xf32>) -> ()
  %reinterpret_cast_19 = memref.reinterpret_cast %alloc_18 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  %alloc_20 = memref.alloc(%dim) : memref<?x10xf32>
  "lmhlo.maximum"(%reinterpret_cast_19, %reinterpret_cast_11, %alloc_20) {disc.device = "gpu"} : (memref<?x10xf32>, memref<?x10xf32>, memref<?x10xf32>) -> ()
  %reinterpret_cast_21 = memref.reinterpret_cast %alloc_20 to offset: [0], sizes: [%dim, 10], strides: [10, 1] {kDiscSymbolicDimAttr = [@S0, @C10]} : memref<?x10xf32> to memref<?x10xf32>
  return %reinterpret_cast_21 : memref<?x10xf32>
}

// -----// IR Dump After DiscDuplicateComputationForFusionPass (disc-duplicate-computation-for-fusion) //----- //
// XLA对于multi output的op会拆分成多个单输出op，为了后续fusion方便，这里将这些op的计算再度合并

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
// 把小的、生命周期局部的 CPU memref.alloc，提升为栈分配 memref.alloca
// 主要是CPU shape计算  

// -----// IR Dump After DiscFusionPass (disc-fusion) //----- //
// 重点关注XLA 两个fusion + stitch fusion

// -----// IR Dump After DiscSpecializeFusionWithSpeculationPass (disc-specialize-fusion-with-speculation) //----- //
// 编译阶段不知道Runtime Shape信息，编译阶段生成不同策略的代码，运行时选择  

// -----// IR Dump After RalInjectExecutionContextPass (disc-ral-inject-execution-context) //----- //

// -----// IR Dump After DiscLowerToLibraryCallPass (disc-lower-to-library-call) //----- //

// TODO(leon) dump this codes
// -----// IR Dump After DiscLhloLegalizeRootsToParallelLoopsPass (disc-lhlo-legalize-roots-to-parallel-loops) //----- //
// -----// IR Dump After InputInlineFusionPass (disc-input-inline-fusion) //----- //

// -----// IR Dump After SCFParallelLoopTiling (disc-parallel-loop-tiling) //----- //
// -----// IR Dump After GpuMapParallelLoopsPass (gpu-map-parallel-loops) //----- //
// -----// IR Dump After ConvertParallelLoopToGpu (convert-parallel-loops-to-gpu) //----- //

// -----// IR Dump After GpuLaunchSinkIndexComputations (gpu-launch-sink-index-computations) //----- //
// -----// IR Dump After GpuKernelOutlining (gpu-kernel-outlining) //----- //
// -----// IR Dump After DiscLowerGpuOpsToNVVMOpsPass (disc-convert-gpu-to-nvvm) //----- //