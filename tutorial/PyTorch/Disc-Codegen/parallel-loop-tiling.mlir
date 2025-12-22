module {
  func.func @parallel_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: memref<?x?xf32>, %arg9: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = arith.muli %arg4, %c1 : index
    %1 = arith.muli %arg5, %c4 : index
    scf.parallel (%arg10, %arg11) = (%arg0, %arg1) to (%arg2, %arg3) step (%0, %1) {
      scf.parallel (%arg12, %arg13) = (%c0, %c0) to (%0, %1) step (%arg4, %arg5) {
        %2 = arith.addi %arg12, %arg10 : index
        %3 = arith.addi %arg13, %arg11 : index
        %true = arith.constant true
        %4 = arith.muli %arg12, %arg4 : index
        %5 = arith.addi %4, %arg10 : index
        %6 = arith.cmpi ult, %5, %arg2 : index
        %7 = arith.andi %true, %6 : i1
        %8 = arith.muli %arg13, %arg5 : index
        %9 = arith.addi %8, %arg11 : index
        %10 = arith.cmpi ult, %9, %arg3 : index
        %11 = arith.andi %7, %10 : i1
        scf.if %11 {
          %12 = memref.load %arg7[%2, %3] : memref<?x?xf32>
          %13 = memref.load %arg8[%2, %3] : memref<?x?xf32>
          %14 = arith.addf %12, %13 : f32
          memref.store %14, %arg9[%2, %3] : memref<?x?xf32>
        }
        scf.yield
      }
      scf.yield
    }
    return
  }
}


// -----
module {
  func.func @static_loop_with_step() {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c22 = arith.constant 22 : index
    %c24 = arith.constant 24 : index
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = arith.muli %c3, %c1 : index
    %1 = arith.muli %c3, %c4 : index
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c22, %c24) step (%0, %1) {
      scf.parallel (%arg2, %arg3) = (%c0_0, %c0_0) to (%0, %1) step (%c3, %c3) {
        %2 = arith.addi %arg2, %arg0 : index
        %3 = arith.addi %arg3, %arg1 : index
        scf.yield
      }
      scf.yield
    }
    return
  }
}


// -----
module {
  func.func @tile_nested_innermost() {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      %c0_2 = arith.constant 0 : index
      %c1_3 = arith.constant 1 : index
      %c4_4 = arith.constant 4 : index
      %2 = arith.muli %c1, %c1_3 : index
      %3 = arith.muli %c1, %c4_4 : index
      scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c2, %c2) step (%2, %3) {
        scf.parallel (%arg4, %arg5) = (%c0_2, %c0_2) to (%2, %3) step (%c1, %c1) {
          %4 = arith.addi %arg4, %arg2 : index
          %5 = arith.addi %arg5, %arg3 : index
          %true = arith.constant true
          %6 = arith.muli %arg4, %c1 : index
          %7 = arith.addi %6, %arg2 : index
          %8 = arith.cmpi ult, %7, %c2 : index
          %9 = arith.andi %true, %8 : i1
          %10 = arith.muli %arg5, %c1 : index
          %11 = arith.addi %10, %arg3 : index
          %12 = arith.cmpi ult, %11, %c2 : index
          %13 = arith.andi %9, %12 : i1
          scf.if %13 {
          }
          scf.yield
        }
        scf.yield
      }
      scf.yield
    }
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = arith.muli %c1, %c1_1 : index
    %1 = arith.muli %c1, %c4 : index
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%0, %1) {
      scf.parallel (%arg2, %arg3) = (%c0_0, %c0_0) to (%0, %1) step (%c1, %c1) {
        %2 = arith.addi %arg2, %arg0 : index
        %3 = arith.addi %arg3, %arg1 : index
        %true = arith.constant true
        %4 = arith.muli %arg2, %c1 : index
        %5 = arith.addi %4, %arg0 : index
        %6 = arith.cmpi ult, %5, %c2 : index
        %7 = arith.andi %true, %6 : i1
        %8 = arith.muli %arg3, %c1 : index
        %9 = arith.addi %8, %arg1 : index
        %10 = arith.cmpi ult, %9, %c2 : index
        %11 = arith.andi %7, %10 : i1
        scf.if %11 {
        }
        scf.yield
      }
      scf.yield
    }
    return
  }
}


// -----
module {
  func.func @tile_nested_in_non_ploop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %0 = arith.muli %c1, %c1_1 : index
        %1 = arith.muli %c1, %c4 : index
        scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c2, %c2) step (%0, %1) {
          scf.parallel (%arg4, %arg5) = (%c0_0, %c0_0) to (%0, %1) step (%c1, %c1) {
            %2 = arith.addi %arg4, %arg2 : index
            %3 = arith.addi %arg5, %arg3 : index
            %true = arith.constant true
            %4 = arith.muli %arg4, %c1 : index
            %5 = arith.addi %4, %arg2 : index
            %6 = arith.cmpi ult, %5, %c2 : index
            %7 = arith.andi %true, %6 : i1
            %8 = arith.muli %arg5, %c1 : index
            %9 = arith.addi %8, %arg3 : index
            %10 = arith.cmpi ult, %9, %c2 : index
            %11 = arith.andi %7, %10 : i1
            scf.if %11 {
            }
            scf.yield
          }
          scf.yield
        }
      }
    }
    return
  }
  func.func @parallel_loop_with_hint(%arg0: i1, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: memref<?x?xf32>, %arg8: memref<?x?xf32>, %arg9: memref<?x?xf32>, %arg10: memref<?x?xf32>) {
    scf.if %arg0 {
      "lmhlo.fusion"() ({
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c1 = arith.constant 1 : index
        %0 = arith.muli %arg5, %c256 : index
        %1 = arith.muli %arg6, %c1 : index
        scf.parallel (%arg11, %arg12) = (%arg1, %arg2) to (%arg3, %arg4) step (%0, %1) {
          scf.parallel (%arg13, %arg14) = (%c0, %c0) to (%0, %1) step (%arg5, %arg6) {
            %2 = arith.addi %arg13, %arg11 : index
            %3 = arith.addi %arg14, %arg12 : index
            %true = arith.constant true
            %4 = arith.muli %arg13, %arg5 : index
            %5 = arith.addi %4, %arg11 : index
            %6 = arith.cmpi ult, %5, %arg3 : index
            %7 = arith.andi %true, %6 : i1
            %8 = arith.muli %arg14, %arg6 : index
            %9 = arith.addi %8, %arg12 : index
            %10 = arith.cmpi ult, %9, %arg4 : index
            %11 = arith.andi %7, %10 : i1
            scf.if %11 {
              %12 = memref.load %arg8[%2, %3] : memref<?x?xf32>
              %13 = memref.load %arg9[%2, %3] : memref<?x?xf32>
              %14 = arith.addf %12, %13 : f32
              memref.store %14, %arg10[%2, %3] : memref<?x?xf32>
            }
            scf.yield
          }
          scf.yield
        }
        "lmhlo.terminator"() : () -> ()
      }) {disc_cta_size_hint = 256 : i32} : () -> ()
    } else {
      "lmhlo.fusion"() ({
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %0 = arith.muli %arg5, %c64 : index
        %1 = arith.muli %arg6, %c1 : index
        scf.parallel (%arg11, %arg12) = (%arg1, %arg2) to (%arg3, %arg4) step (%0, %1) {
          scf.parallel (%arg13, %arg14) = (%c0, %c0) to (%0, %1) step (%arg5, %arg6) {
            %2 = arith.addi %arg13, %arg11 : index
            %3 = arith.addi %arg14, %arg12 : index
            %true = arith.constant true
            %4 = arith.muli %arg13, %arg5 : index
            %5 = arith.addi %4, %arg11 : index
            %6 = arith.cmpi ult, %5, %arg3 : index
            %7 = arith.andi %true, %6 : i1
            %8 = arith.muli %arg14, %arg6 : index
            %9 = arith.addi %8, %arg12 : index
            %10 = arith.cmpi ult, %9, %arg4 : index
            %11 = arith.andi %7, %10 : i1
            scf.if %11 {
              %12 = memref.load %arg8[%2, %3] : memref<?x?xf32>
              %13 = memref.load %arg9[%2, %3] : memref<?x?xf32>
              %14 = arith.addf %12, %13 : f32
              memref.store %14, %arg10[%2, %3] : memref<?x?xf32>
            }
            scf.yield
          }
          scf.yield
        }
        "lmhlo.terminator"() : () -> ()
      }) {disc_cta_size_hint = 64 : i32} : () -> ()
    }
    return
  }
}

