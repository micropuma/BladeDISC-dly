module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}}  {
  func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<1xi32>, %arg3: tensor<2x2xi64>) -> tensor<?x?xf32> attributes {tf.entry_function = {disc.input_shape_3 = dense<0> : tensor<2x2xi64>, input_placements = "gpu,gpu,cpu,cpu", inputs = "input0,input1,input2,input3", output_placements = "gpu", outputs = "output0"}} {
    %0 = mhlo.constant dense<0> : tensor<2xi64>
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %3 = shape.const_shape [8] : tensor<1xindex>
    %4 = mhlo.constant dense<8> : tensor<1xi32>
    %5 = mhlo.constant dense<[1.200000e+00, 1.300000e+00, 1.400000e+00, 1.500000e+00, 1.600000e+00, 1.700000e+00, 1.800000e+00, 1.900000e+00]> : tensor<8xf32>
    %6 = "mhlo.concatenate"(%arg2, %4) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = shape.shape_of %arg1 : tensor<?x?x?xf32> -> tensor<3xindex>
    %8 = shape.num_elements %7 : tensor<3xindex> -> index
    %9 = mhlo.cstr_reshapable %8, %6 : index, tensor<2xi32>
    %10 = shape.assuming %9 -> (tensor<?x?xf32>) {
      %37 = mhlo.compute_reshape_shape %8, %6 : index, tensor<2xi32> -> tensor<2xi32>
      %38 = "mhlo.dynamic_reshape"(%arg1, %37) : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
      shape.assuming_yield %38 : tensor<?x?xf32>
    }
    %11 = "mhlo.dot"(%arg0, %10) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %12 = shape.shape_of %11 : tensor<?x?xf32> -> tensor<2xindex>
    %13 = shape.cstr_broadcastable %12, %3 : tensor<2xindex>, tensor<1xindex>
    %14 = shape.assuming %13 -> (tensor<?x8xf32>) {
      %37 = shape.shape_of %11 : tensor<?x?xf32> -> tensor<2xindex>
      %38 = shape.broadcast %37, %3 : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
      %39 = "mhlo.dynamic_broadcast_in_dim"(%11, %38) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x8xf32>
      %40 = "mhlo.dynamic_broadcast_in_dim"(%5, %38) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<8xf32>, tensor<2xindex>) -> tensor<?x8xf32>
      %41 = mhlo.add %39, %40 : tensor<?x8xf32>
      shape.assuming_yield %41 : tensor<?x8xf32>
    }
    %15 = "mhlo.reduce"(%14, %2) ( {
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):  // no predecessors
      %37 = mhlo.maximum %arg4, %arg5 : tensor<f32>
      "mhlo.return"(%37) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x8xf32>, tensor<f32>) -> tensor<?xf32>
    %16 = shape.shape_of %15 : tensor<?xf32> -> tensor<1xindex>
    %17 = tensor.extract %16[%c0] : tensor<1xindex>
    %18 = tensor.from_elements %17, %c1 : tensor<2xindex>
    %19 = "mhlo.dynamic_reshape"(%15, %18) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x1xf32>
    %20 = shape.shape_of %14 : tensor<?x8xf32> -> tensor<2xindex>
    %21 = shape.cstr_broadcastable %20, %18 : tensor<2xindex>, tensor<2xindex>
    %22 = shape.assuming %21 -> (tensor<?x8xf32>) {
      %37 = shape.shape_of %14 : tensor<?x8xf32> -> tensor<2xindex>
      %38 = shape.broadcast %37, %18 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %39 = "mhlo.dynamic_broadcast_in_dim"(%14, %38) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x8xf32>, tensor<2xindex>) -> tensor<?x8xf32>
      %40 = "mhlo.dynamic_broadcast_in_dim"(%19, %38) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x8xf32>
      %41 = mhlo.subtract %39, %40 : tensor<?x8xf32>
      shape.assuming_yield %41 : tensor<?x8xf32>
    }
    %23 = "mhlo.exponential"(%22) : (tensor<?x8xf32>) -> tensor<?x8xf32>
    %24 = "mhlo.reduce"(%23, %1) ( {
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):  // no predecessors
      %37 = mhlo.add %arg4, %arg5 : tensor<f32>
      "mhlo.return"(%37) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x8xf32>, tensor<f32>) -> tensor<?xf32>
    %25 = shape.shape_of %24 : tensor<?xf32> -> tensor<1xindex>
    %26 = tensor.extract %25[%c0] : tensor<1xindex>
    %27 = tensor.from_elements %26, %c1 : tensor<2xindex>
    %28 = "mhlo.dynamic_reshape"(%24, %27) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x1xf32>
    %29 = shape.shape_of %23 : tensor<?x8xf32> -> tensor<2xindex>
    %30 = shape.cstr_broadcastable %29, %27 : tensor<2xindex>, tensor<2xindex>
    %31 = shape.assuming %30 -> (tensor<?x8xf32>) {
      %37 = shape.shape_of %23 : tensor<?x8xf32> -> tensor<2xindex>
      %38 = shape.broadcast %37, %27 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %39 = "mhlo.dynamic_broadcast_in_dim"(%23, %38) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x8xf32>, tensor<2xindex>) -> tensor<?x8xf32>
      %40 = "mhlo.dynamic_broadcast_in_dim"(%28, %38) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x8xf32>
      %41 = mhlo.divide %39, %40 : tensor<?x8xf32>
      shape.assuming_yield %41 : tensor<?x8xf32>
    }
    %32 = "mhlo.transpose"(%arg3) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x2xi64>) -> tensor<2x2xi64>
    %33 = "mhlo.reshape"(%32) : (tensor<2x2xi64>) -> tensor<4xi64>
    %34 = "mhlo.slice"(%33) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> tensor<2xi64>
    %35 = "mhlo.slice"(%33) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> tensor<2xi64>
    %36 = "mhlo.dynamic_pad"(%31, %1, %34, %35, %0) : (tensor<?x8xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
    return %36 : tensor<?x?xf32>
  }
}

