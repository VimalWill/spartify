module {
  func.func @spartify_net(%arg0: tensor<1x10xf32>) -> tensor<1x5xf32> {
    %cst = arith.constant dense_resource<torch_tensor_5_10_torch.float32> : tensor<5x10xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<10x5xf32>
    %transposed = linalg.transpose ins(%cst : tensor<5x10xf32>) outs(%0 : tensor<10x5xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x5xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x5xf32>) -> tensor<1x5xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x10xf32>, tensor<10x5xf32>) outs(%2 : tensor<1x5xf32>) -> tensor<1x5xf32>
    return %3 : tensor<1x5xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_5_10_torch.float32: "0x04000000ADED993DB622A8BD4CC39DBE1FB5F2BD7CFA30BEF7285FBE762549BD19C69E3C2C1C9EBEC37C933EDC951D3EA422B33D751E1E3E47E225BEC32453BE024C7ABEFB65643E2768E73DA09E07BEC2446EBE89A7873EAD869E3DE4F1673D3FD35F3ECEBBF23DEF5614BE9E73803D021A6BBE31A926BE5FB3B63D71BF1D3EAF5D65BD9C046CBD1B9154BC2029EABD447A0B3E3B9C62BEAB264BBEF0DD6F3D92D287BECF6A0BBD365DD23D9B991C3E139B98BE9A5FBE3DE9520CBE1AE204BEB8A386BCAE518DBDD129793E"
    }
  }
#-}
