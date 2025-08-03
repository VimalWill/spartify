#ifndef SPARTIFY_TRANSFORMS_PASSES_TD
#define SPARTIFY_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SpartifyConvertToSparse : Pass<"spartify-convert-to-sparse", ""> {
    let summary = "converts dense tensors to sparse tensors";
    let constructor = "mlir::spartify_compiler::createSpartifyConvertToSparsePass()"; 
}

def SpartifyGenericConversion : Pass<"spartify-generic-conversion", "" > {
    let summary = "converts named ops to generic ops"; 
    let constructor = "mlir::spartify_compiler::createSpartifyGenericConversionPass()"; 
}

def SpartifyTileAndFuse : Pass<"spartify-tile-and-fuse", "func::FuncOp"> {
    let summary = "tile and fuse operations"; 
    let constructor = "mlir::spartify_compiler::createSpartifyTileAndFusePass()"; 
}

def SpartifyConvertToTransposedMatmulOp : Pass<"spartify-convert-transposed-matmul-op", "func::FuncOp"> {
    let summary = "convert transpose & matmul op to transposedMatmulOp";
    let constructor = "mlir::spartify_compiler::createSpartifyConvertToTransposedMatmulOpPass()"; 
}

#endif // SPARTIFY_TRANSFORMS_PASSES_TD