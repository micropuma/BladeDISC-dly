module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}}  {
  func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<1xi32>, %arg3: tensor<?x?xi64>) -> tensor<?x?xf32> attributes {tf.entry_function = {disc.input_shape_3 = dense<0> : tensor<2x2xi64>, input_placements = "gpu,gpu,cpu,cpu", inputs = "input0,input1,input2,input3", output_placements = "gpu", outputs = "output0"}} {
    %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_0 = "tf.Const"() {value = dense<[1.200000e+00, 1.300000e+00, 1.400000e+00, 1.500000e+00, 1.600000e+00, 1.700000e+00, 1.800000e+00, 1.900000e+00]> : tensor<8xf32>} : () -> tensor<8xf32>
    %cst_1 = "tf.Const"() {value = dense<8> : tensor<1xi32>} : () -> tensor<1xi32>
    %cst_2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.ConcatV2"(%arg2, %cst_1, %cst_2) : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %1 = "tf.Reshape"(%arg1, %0) {T = f32, Tshape = i32} : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %2 = "tf.MatMul"(%arg0, %1) {transpose_a = false, transpose_b = false} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = "tf.AddV2"(%2, %cst_0) : (tensor<?x?xf32>, tensor<8xf32>) -> tensor<?x8xf32>
    %4 = "tf.Softmax"(%3) : (tensor<?x8xf32>) -> tensor<?x8xf32>
    %5 = "tf.PadV2"(%4, %arg3, %cst) : (tensor<?x8xf32>, tensor<?x?xi64>, tensor<f32>) -> tensor<?x?xf32>
    return %5 : tensor<?x?xf32>
  }
}

