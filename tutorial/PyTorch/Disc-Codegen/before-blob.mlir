gpu.module @main_kernel {
  llvm.func @main_kRowReduction_reduce_add__5_2_0___1w1rX_vec2(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !llvm.ptr<f32>, %arg4: !llvm.ptr<f32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: !llvm.ptr<f32>, %arg12: !llvm.ptr<f32>, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: !llvm.ptr<f32>, %arg19: !llvm.ptr<f32>, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: !llvm.ptr<f32>, %arg24: !llvm.ptr<f32>, %arg25: i32, %arg26: i32, %arg27: i32) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %2 = llvm.insertvalue %arg4, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %3 = llvm.insertvalue %arg5, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %4 = llvm.insertvalue %arg6, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %5 = llvm.insertvalue %arg8, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %6 = llvm.insertvalue %arg7, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %7 = llvm.insertvalue %arg9, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %9 = llvm.insertvalue %arg11, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %10 = llvm.insertvalue %arg12, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %11 = llvm.insertvalue %arg13, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %12 = llvm.insertvalue %arg14, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %13 = llvm.insertvalue %arg16, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %14 = llvm.insertvalue %arg15, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %15 = llvm.insertvalue %arg17, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %17 = llvm.insertvalue %arg18, %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %18 = llvm.insertvalue %arg19, %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %19 = llvm.insertvalue %arg20, %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %20 = llvm.insertvalue %arg21, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %21 = llvm.insertvalue %arg22, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %22 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %23 = llvm.insertvalue %arg23, %22[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %24 = llvm.insertvalue %arg24, %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %25 = llvm.insertvalue %arg25, %24[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %26 = llvm.insertvalue %arg26, %25[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %27 = llvm.insertvalue %arg27, %26[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %28 = llvm.mlir.constant(256 : index) : i32
    %29 = llvm.mlir.constant(16 : index) : i32
    %30 = llvm.mlir.constant(32 : index) : i32
    %31 = llvm.mlir.constant(2 : index) : i32
    %32 = llvm.mlir.constant(1 : index) : i32
    %33 = llvm.mlir.constant(8 : index) : i32
    %34 = llvm.mlir.constant(0 : index) : i32
    %35 = llvm.mlir.constant(0x7FC00000 : f32) : f32
    %36 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %37 = llvm.mlir.constant(1 : i32) : i32
    %38 = llvm.mlir.constant(32 : i32) : i32
    %39 = llvm.mlir.constant(2 : i32) : i32
    %40 = llvm.mlir.constant(4 : i32) : i32
    %41 = llvm.mlir.constant(8 : i32) : i32
    %42 = llvm.mlir.constant(16 : i32) : i32
    %43 = nvvm.read.ptx.sreg.ctaid.x : i32
    %44 = nvvm.read.ptx.sreg.tid.x : i32
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %45 = llvm.mul %43, %arg0  : i32
    %46 = llvm.add %44, %45  : i32
    %47 = llvm.add %44, %45  : i32
    %48 = llvm.icmp "ult" %47, %arg1 : i32
    llvm.cond_br %48, ^bb2, ^bb10
  ^bb2:  // pred: ^bb1
    %49 = llvm.srem %46, %28  : i32
    %50 = llvm.sdiv %46, %28  : i32
    %51 = llvm.mul %50, %29  : i32
    %52 = llvm.urem %49, %30  : i32
    %53 = llvm.udiv %49, %30  : i32
    %54 = llvm.mul %53, %31  : i32
    %55 = llvm.add %51, %54  : i32
    %56 = llvm.add %55, %32  : i32
    %57 = llvm.icmp "ult" %55, %arg2 : i32
    llvm.cond_br %57, ^bb3, ^bb9
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4(%52, %36, %36 : i32, f32, f32)
  ^bb4(%58: i32, %59: f32, %60: f32):  // 2 preds: ^bb3, ^bb5
    %61 = llvm.icmp "slt" %58, %33 : i32
    llvm.cond_br %61, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %62 = llvm.mul %55, %33  : i32
    %63 = llvm.add %62, %58  : i32
    %64 = llvm.mul %arg2, %33  : i32
    %65 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %66 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %67 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %68 = llvm.insertvalue %66, %65[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %69 = llvm.insertvalue %67, %68[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %70 = llvm.insertvalue %34, %69[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %71 = llvm.insertvalue %64, %70[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %72 = llvm.insertvalue %32, %71[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %73 = llvm.extractvalue %72[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %74 = llvm.mlir.constant(0 : index) : i64
    %75 = llvm.mlir.constant(3 : index) : i64
    %76 = llvm.ptrtoint %73 : !llvm.ptr<f32> to i64
    %77 = llvm.and %76, %75  : i64
    %78 = llvm.icmp "eq" %77, %74 : i64
    "llvm.intr.assume"(%78) : (i1) -> ()
    %79 = llvm.icmp "eq" %arg10, %33 : i32
    %80 = llvm.select %79, %58, %34 : i1, i32
    %81 = llvm.mul %55, %arg10  : i32
    %82 = llvm.add %81, %80  : i32
    %83 = llvm.mul %arg2, %arg10  : i32
    %84 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %85 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %86 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
    %87 = llvm.insertvalue %85, %84[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %88 = llvm.insertvalue %86, %87[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %89 = llvm.insertvalue %34, %88[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %90 = llvm.insertvalue %83, %89[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %91 = llvm.insertvalue %32, %90[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %92 = llvm.extractvalue %91[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %93 = llvm.getelementptr %92[%82] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    %94 = llvm.load %93 : !llvm.ptr<f32>
    %95 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %96 = llvm.extractvalue %21[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %97 = llvm.extractvalue %21[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %98 = llvm.insertvalue %96, %95[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %99 = llvm.insertvalue %97, %98[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %100 = llvm.mlir.constant(0 : index) : i32
    %101 = llvm.insertvalue %100, %99[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %102 = llvm.mlir.constant(8 : index) : i32
    %103 = llvm.insertvalue %102, %101[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %104 = llvm.mlir.constant(1 : index) : i32
    %105 = llvm.insertvalue %104, %103[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %106 = llvm.extractvalue %105[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %107 = llvm.getelementptr %106[%58] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    %108 = llvm.load %107 : !llvm.ptr<f32>
    %109 = llvm.fadd %94, %108  : f32
    %110 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %111 = llvm.extractvalue %72[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %112 = llvm.extractvalue %72[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %113 = llvm.insertvalue %111, %110[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %114 = llvm.insertvalue %112, %113[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %115 = llvm.insertvalue %34, %114[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %116 = llvm.insertvalue %64, %115[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %117 = llvm.insertvalue %32, %116[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %118 = llvm.extractvalue %117[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %119 = llvm.getelementptr %118[%63] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    llvm.store %109, %119 : !llvm.ptr<f32>
    %120 = llvm.fcmp "ogt" %59, %109 : f32
    %121 = llvm.select %120, %59, %109 : i1, f32
    %122 = llvm.fcmp "uno" %59, %109 : f32
    %123 = llvm.select %122, %35, %121 : i1, f32
    %124 = llvm.mul %56, %33  : i32
    %125 = llvm.add %124, %58  : i32
    %126 = llvm.extractvalue %72[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %127 = llvm.mlir.constant(0 : index) : i64
    %128 = llvm.mlir.constant(3 : index) : i64
    %129 = llvm.ptrtoint %126 : !llvm.ptr<f32> to i64
    %130 = llvm.and %129, %128  : i64
    %131 = llvm.icmp "eq" %130, %127 : i64
    "llvm.intr.assume"(%131) : (i1) -> ()
    %132 = llvm.mul %56, %arg10  : i32
    %133 = llvm.add %132, %80  : i32
    %134 = llvm.extractvalue %91[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %135 = llvm.getelementptr %134[%133] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    %136 = llvm.load %135 : !llvm.ptr<f32>
    %137 = llvm.fadd %136, %108  : f32
    %138 = llvm.extractvalue %117[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %139 = llvm.getelementptr %138[%125] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    llvm.store %137, %139 : !llvm.ptr<f32>
    %140 = llvm.fcmp "ogt" %60, %137 : f32
    %141 = llvm.select %140, %60, %137 : i1, f32
    %142 = llvm.fcmp "uno" %60, %137 : f32
    %143 = llvm.select %142, %35, %141 : i1, f32
    %144 = llvm.add %58, %30  : i32
    llvm.br ^bb4(%144, %123, %143 : i32, f32, f32)
  ^bb6:  // pred: ^bb4
    %145 = llvm.mlir.constant(1 : i32) : i32
    %146 = llvm.shl %145, %38  : i32
    %147 = llvm.sub %146, %145  : i32
    %148 = llvm.sub %38, %145  : i32
    %149 = nvvm.shfl.sync.bfly %147, %59, %37, %148 : !llvm.struct<(f32, i1)>
    %150 = llvm.extractvalue %149[0 : index] : !llvm.struct<(f32, i1)>
    %151 = llvm.extractvalue %149[1 : index] : !llvm.struct<(f32, i1)>
    %152 = llvm.fcmp "ogt" %59, %150 : f32
    %153 = llvm.select %152, %59, %150 : i1, f32
    %154 = llvm.fcmp "uno" %59, %150 : f32
    %155 = llvm.select %154, %35, %153 : i1, f32
    %156 = llvm.mlir.constant(1 : i32) : i32
    %157 = llvm.shl %156, %38  : i32
    %158 = llvm.sub %157, %156  : i32
    %159 = llvm.sub %38, %156  : i32
    %160 = nvvm.shfl.sync.bfly %158, %60, %37, %159 : !llvm.struct<(f32, i1)>
    %161 = llvm.extractvalue %160[0 : index] : !llvm.struct<(f32, i1)>
    %162 = llvm.extractvalue %160[1 : index] : !llvm.struct<(f32, i1)>
    %163 = llvm.fcmp "ogt" %60, %161 : f32
    %164 = llvm.select %163, %60, %161 : i1, f32
    %165 = llvm.fcmp "uno" %60, %161 : f32
    %166 = llvm.select %165, %35, %164 : i1, f32
    %167 = llvm.mlir.constant(1 : i32) : i32
    %168 = llvm.shl %167, %38  : i32
    %169 = llvm.sub %168, %167  : i32
    %170 = llvm.sub %38, %167  : i32
    %171 = nvvm.shfl.sync.bfly %169, %155, %39, %170 : !llvm.struct<(f32, i1)>
    %172 = llvm.extractvalue %171[0 : index] : !llvm.struct<(f32, i1)>
    %173 = llvm.extractvalue %171[1 : index] : !llvm.struct<(f32, i1)>
    %174 = llvm.fcmp "ogt" %155, %172 : f32
    %175 = llvm.select %174, %155, %172 : i1, f32
    %176 = llvm.fcmp "uno" %155, %172 : f32
    %177 = llvm.select %176, %35, %175 : i1, f32
    %178 = llvm.mlir.constant(1 : i32) : i32
    %179 = llvm.shl %178, %38  : i32
    %180 = llvm.sub %179, %178  : i32
    %181 = llvm.sub %38, %178  : i32
    %182 = nvvm.shfl.sync.bfly %180, %166, %39, %181 : !llvm.struct<(f32, i1)>
    %183 = llvm.extractvalue %182[0 : index] : !llvm.struct<(f32, i1)>
    %184 = llvm.extractvalue %182[1 : index] : !llvm.struct<(f32, i1)>
    %185 = llvm.fcmp "ogt" %166, %183 : f32
    %186 = llvm.select %185, %166, %183 : i1, f32
    %187 = llvm.fcmp "uno" %166, %183 : f32
    %188 = llvm.select %187, %35, %186 : i1, f32
    %189 = llvm.mlir.constant(1 : i32) : i32
    %190 = llvm.shl %189, %38  : i32
    %191 = llvm.sub %190, %189  : i32
    %192 = llvm.sub %38, %189  : i32
    %193 = nvvm.shfl.sync.bfly %191, %177, %40, %192 : !llvm.struct<(f32, i1)>
    %194 = llvm.extractvalue %193[0 : index] : !llvm.struct<(f32, i1)>
    %195 = llvm.extractvalue %193[1 : index] : !llvm.struct<(f32, i1)>
    %196 = llvm.fcmp "ogt" %177, %194 : f32
    %197 = llvm.select %196, %177, %194 : i1, f32
    %198 = llvm.fcmp "uno" %177, %194 : f32
    %199 = llvm.select %198, %35, %197 : i1, f32
    %200 = llvm.mlir.constant(1 : i32) : i32
    %201 = llvm.shl %200, %38  : i32
    %202 = llvm.sub %201, %200  : i32
    %203 = llvm.sub %38, %200  : i32
    %204 = nvvm.shfl.sync.bfly %202, %188, %40, %203 : !llvm.struct<(f32, i1)>
    %205 = llvm.extractvalue %204[0 : index] : !llvm.struct<(f32, i1)>
    %206 = llvm.extractvalue %204[1 : index] : !llvm.struct<(f32, i1)>
    %207 = llvm.fcmp "ogt" %188, %205 : f32
    %208 = llvm.select %207, %188, %205 : i1, f32
    %209 = llvm.fcmp "uno" %188, %205 : f32
    %210 = llvm.select %209, %35, %208 : i1, f32
    %211 = llvm.mlir.constant(1 : i32) : i32
    %212 = llvm.shl %211, %38  : i32
    %213 = llvm.sub %212, %211  : i32
    %214 = llvm.sub %38, %211  : i32
    %215 = nvvm.shfl.sync.bfly %213, %199, %41, %214 : !llvm.struct<(f32, i1)>
    %216 = llvm.extractvalue %215[0 : index] : !llvm.struct<(f32, i1)>
    %217 = llvm.extractvalue %215[1 : index] : !llvm.struct<(f32, i1)>
    %218 = llvm.fcmp "ogt" %199, %216 : f32
    %219 = llvm.select %218, %199, %216 : i1, f32
    %220 = llvm.fcmp "uno" %199, %216 : f32
    %221 = llvm.select %220, %35, %219 : i1, f32
    %222 = llvm.mlir.constant(1 : i32) : i32
    %223 = llvm.shl %222, %38  : i32
    %224 = llvm.sub %223, %222  : i32
    %225 = llvm.sub %38, %222  : i32
    %226 = nvvm.shfl.sync.bfly %224, %210, %41, %225 : !llvm.struct<(f32, i1)>
    %227 = llvm.extractvalue %226[0 : index] : !llvm.struct<(f32, i1)>
    %228 = llvm.extractvalue %226[1 : index] : !llvm.struct<(f32, i1)>
    %229 = llvm.fcmp "ogt" %210, %227 : f32
    %230 = llvm.select %229, %210, %227 : i1, f32
    %231 = llvm.fcmp "uno" %210, %227 : f32
    %232 = llvm.select %231, %35, %230 : i1, f32
    %233 = llvm.mlir.constant(1 : i32) : i32
    %234 = llvm.shl %233, %38  : i32
    %235 = llvm.sub %234, %233  : i32
    %236 = llvm.sub %38, %233  : i32
    %237 = nvvm.shfl.sync.bfly %235, %221, %42, %236 : !llvm.struct<(f32, i1)>
    %238 = llvm.extractvalue %237[0 : index] : !llvm.struct<(f32, i1)>
    %239 = llvm.extractvalue %237[1 : index] : !llvm.struct<(f32, i1)>
    %240 = llvm.fcmp "ogt" %221, %238 : f32
    %241 = llvm.select %240, %221, %238 : i1, f32
    %242 = llvm.fcmp "uno" %221, %238 : f32
    %243 = llvm.select %242, %35, %241 : i1, f32
    %244 = llvm.mlir.constant(1 : i32) : i32
    %245 = llvm.shl %244, %38  : i32
    %246 = llvm.sub %245, %244  : i32
    %247 = llvm.sub %38, %244  : i32
    %248 = nvvm.shfl.sync.bfly %246, %232, %42, %247 : !llvm.struct<(f32, i1)>
    %249 = llvm.extractvalue %248[0 : index] : !llvm.struct<(f32, i1)>
    %250 = llvm.extractvalue %248[1 : index] : !llvm.struct<(f32, i1)>
    %251 = llvm.fcmp "ogt" %232, %249 : f32
    %252 = llvm.select %251, %232, %249 : i1, f32
    %253 = llvm.fcmp "uno" %232, %249 : f32
    %254 = llvm.select %253, %35, %252 : i1, f32
    %255 = llvm.icmp "eq" %52, %34 : i32
    llvm.cond_br %255, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %256 = llvm.extractvalue %27[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %257 = llvm.mlir.constant(0 : index) : i64
    %258 = llvm.mlir.constant(7 : index) : i64
    %259 = llvm.ptrtoint %256 : !llvm.ptr<f32> to i64
    %260 = llvm.and %259, %258  : i64
    %261 = llvm.icmp "eq" %260, %257 : i64
    "llvm.intr.assume"(%261) : (i1) -> ()
    %262 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %263 = llvm.extractvalue %27[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %264 = llvm.extractvalue %27[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %265 = llvm.insertvalue %263, %262[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %266 = llvm.insertvalue %264, %265[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %267 = llvm.insertvalue %34, %266[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %268 = llvm.insertvalue %arg2, %267[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %269 = llvm.insertvalue %32, %268[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %270 = llvm.extractvalue %269[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %271 = llvm.getelementptr %270[%55] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    llvm.store %243, %271 : !llvm.ptr<f32>
    %272 = llvm.extractvalue %269[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<1 x i32>, array<1 x i32>)>
    %273 = llvm.getelementptr %272[%56] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
    llvm.store %254, %273 : !llvm.ptr<f32>
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    llvm.br ^bb9
  ^bb9:  // 2 preds: ^bb2, ^bb8
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb1, ^bb9
    llvm.return
  }
}
