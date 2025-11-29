module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<1xi32>, %arg3: tensor<?x?xi64>) -> (tensor<*xf32>) attributes {tf.entry_function = {inputs = "input0,input1,input2,input3", outputs = "output0", input_placements="gpu,gpu,cpu,cpu", output_placements="gpu", disc.input_shape_3 = dense<0> : tensor<2x2xi64>}} {
    %graph = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> tensor<1xi32>
      %2:2 = tf_executor.island wraps "tf.ConcatV2"(%arg2, %1, %0) : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
      %3:2 = tf_executor.island wraps "tf.Reshape"(%arg1, %2) {T = f32, Tshape = i32} : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
      %4:2 = tf_executor.island wraps "tf.MatMul"(%arg0, %3) {transpose_a = false, transpose_b = false} : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>)
      %5:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]> : tensor<8xf32>} : () -> (tensor<8xf32>)
      %6:2 = tf_executor.island wraps "tf.AddV2"(%4, %5) : (tensor<?x?xf32>, tensor<8xf32>) -> (tensor<?x?xf32>)
      %7:2 = tf_executor.island wraps "tf.Softmax"(%6) : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %8:2 = tf_executor.island wraps "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
      %9:2 = tf_executor.island wraps "tf.PadV2"(%7, %arg3, %8) : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>) -> tensor<*xf32>
      tf_executor.fetch %9 : tensor<*xf32>
    }
    return %graph : tensor<*xf32>
  }
}
