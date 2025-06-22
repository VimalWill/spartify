module {
  memref.global "private" constant @__constant_5x10xf32 : memref<5x10xf32> = dense_resource<torch_tensor_5_10_torch.float32> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x5xf32 : memref<1x5xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func private @"_insert_dense_structured[2_4]_10_5_f32_32_32"(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: !llvm.struct<(array<2 x i64>, array<2 x i64>)>, %arg3: index, %arg4: index, %arg5: f32) -> (memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.extractvalue %arg2[1, 0] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %1 = arith.index_cast %0 : i64 to index
    %2 = arith.index_cast %arg4 : index to i32
    %dim = memref.dim %arg0, %c0 : memref<?xi32>
    %3 = arith.addi %1, %c1 : index
    %4 = arith.cmpi ugt, %3, %dim : index
    %5 = scf.if %4 -> (memref<?xi32>) {
      %15 = arith.muli %dim, %c2 : index
      %16 = memref.realloc %arg0(%15) : memref<?xi32> to memref<?xi32>
      scf.yield %16 : memref<?xi32>
    } else {
      scf.yield %arg0 : memref<?xi32>
    }
    memref.store %2, %5[%1] : memref<?xi32>
    %6 = arith.index_cast %3 : index to i64
    %7 = llvm.insertvalue %6, %arg2[1, 0] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.extractvalue %arg2[1, 1] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %9 = arith.index_cast %8 : i64 to index
    %dim_0 = memref.dim %arg1, %c0 : memref<?xf32>
    %10 = arith.addi %9, %c1 : index
    %11 = arith.cmpi ugt, %10, %dim_0 : index
    %12 = scf.if %11 -> (memref<?xf32>) {
      %15 = arith.muli %dim_0, %c2 : index
      %16 = memref.realloc %arg1(%15) : memref<?xf32> to memref<?xf32>
      scf.yield %16 : memref<?xf32>
    } else {
      scf.yield %arg1 : memref<?xf32>
    }
    memref.store %arg5, %12[%9] : memref<?xf32>
    %13 = arith.index_cast %10 : index to i64
    %14 = llvm.insertvalue %13, %7[1, 1] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    return %5, %12, %14 : memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>
  }
  func.func @spartify_net(%arg0: memref<1x10xf32>) -> memref<1x5xf32> {
    %c5_i64 = arith.constant 5 : i64
    %c10_i64 = arith.constant 10 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.mlir.poison : !llvm.struct<(array<2 x i64>, array<2 x i64>)>
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c5 = arith.constant 5 : index
    %1 = memref.get_global @__constant_1x5xf32 : memref<1x5xf32>
    %2 = memref.get_global @__constant_5x10xf32 : memref<5x10xf32>
    %alloc = memref.alloc() : memref<16xi32>
    %cast = memref.cast %alloc : memref<16xi32> to memref<?xi32>
    %alloc_0 = memref.alloc() : memref<16xf32>
    %cast_1 = memref.cast %alloc_0 : memref<16xf32> to memref<?xf32>
    %3 = llvm.insertvalue %c0_i64, %0[1, 0] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %c0_i64, %3[1, 1] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %c10_i64, %4[0, 0] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %c5_i64, %5[0, 1] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %7:3 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cast, %arg3 = %cast_1, %arg4 = %6) -> (memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>) {
      %12:3 = scf.for %arg5 = %c0 to %c5 step %c1 iter_args(%arg6 = %arg2, %arg7 = %arg3, %arg8 = %arg4) -> (memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>) {
        %13 = memref.load %2[%arg5, %arg1] : memref<5x10xf32>
        %14 = arith.cmpf une, %13, %cst : f32
        %15:3 = scf.if %14 -> (memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>) {
          %16:3 = func.call @"_insert_dense_structured[2_4]_10_5_f32_32_32"(%arg6, %arg7, %arg8, %arg1, %arg5, %13) : (memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>, index, index, f32) -> (memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>)
          scf.yield %16#0, %16#1, %16#2 : memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>
        } else {
          scf.yield %arg6, %arg7, %arg8 : memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>
        }
        scf.yield %15#0, %15#1, %15#2 : memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>
      } {"Emitted from" = "linalg.generic"}
      scf.yield %12#0, %12#1, %12#2 : memref<?xi32>, memref<?xf32>, !llvm.struct<(array<2 x i64>, array<2 x i64>)>
    } {"Emitted from" = "linalg.generic"}
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x5xf32>
    memref.copy %1, %alloc_2 : memref<1x5xf32> to memref<1x5xf32>
    %8 = llvm.extractvalue %7#2[1, 1] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %9 = arith.index_cast %8 : i64 to index
    %subview = memref.subview %7#1[0] [%9] [1] : memref<?xf32> to memref<?xf32>
    %10 = llvm.extractvalue %7#2[1, 0] : !llvm.struct<(array<2 x i64>, array<2 x i64>)> 
    %11 = arith.index_cast %10 : i64 to index
    %subview_3 = memref.subview %7#0[0] [%11] [1] : memref<?xi32> to memref<?xi32>
    scf.for %arg1 = %c0 to %c10 step %c1 {
      %12 = memref.load %arg0[%c0, %arg1] : memref<1x10xf32>
      %13 = arith.muli %arg1, %c2 : index
      %14 = arith.addi %13, %c2 : index
      scf.for %arg2 = %13 to %14 step %c1 {
        %15 = memref.load %subview_3[%arg2] : memref<?xi32>
        %16 = arith.extui %15 : i32 to i64
        %17 = arith.index_cast %16 : i64 to index
        %18 = memref.load %alloc_2[%c0, %17] : memref<1x5xf32>
        %19 = memref.load %subview[%arg2] : memref<?xf32>
        %20 = arith.mulf %12, %19 : f32
        %21 = arith.addf %18, %20 : f32
        memref.store %21, %alloc_2[%c0, %17] : memref<1x5xf32>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    return %alloc_2 : memref<1x5xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_5_10_torch.float32: "0x04000000ADED993DB622A8BD4CC39DBE1FB5F2BD7CFA30BEF7285FBE762549BD19C69E3C2C1C9EBEC37C933EDC951D3EA422B33D751E1E3E47E225BEC32453BE024C7ABEFB65643E2768E73DA09E07BEC2446EBE89A7873EAD869E3DE4F1673D3FD35F3ECEBBF23DEF5614BE9E73803D021A6BBE31A926BE5FB3B63D71BF1D3EAF5D65BD9C046CBD1B9154BC2029EABD447A0B3E3B9C62BEAB264BBEF0DD6F3D92D287BECF6A0BBD365DD23D9B991C3E139B98BE9A5FBE3DE9520CBE1AE204BEB8A386BCAE518DBDD129793E"
    }
  }
#-}