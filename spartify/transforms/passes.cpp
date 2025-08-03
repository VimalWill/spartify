#include "spartify/transforms/passes.h"

namespace mlir::spartify_compiler {

void spartifyCodegenPassPipeline(mlir::OpPassManager &pm,
                                 bool isSparse) {
  if (isSparse) {
    pm.addNestedPass<func::FuncOp>(createSpartifyConvertToSparsePass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCCPPass());

    pm.addPass(mlir::createSparsificationAndBufferizationPass());
    // pm.addPass(mlir::createLowerSparseOpsToForeachPass(false, true));
    // pm.addPass(mlir::createLowerForeachToSCFPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addPass(mlir::createStorageSpecifierToLLVMPass());
    // pm.addPass(mlir::createSparseTensorConversionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  } else {
    pm.addNestedPass<func::FuncOp>(createSpartifyGenericConversionPass());
    pm.addNestedPass<func::FuncOp>(createSpartifyTileAndFusePass()); 

    pm.addNestedPass<func::FuncOp>(createSpartifyConvertToTransposedMatmulOpPass()); 
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addNestedPass<func::FuncOp>(createSpartifyGenericConversionPass()); 
  }
}

void registerSpartifyPasses() {
  mlir::spartify_compiler::registerPasses(); // generated
}

} // namespace mlir::spartify_compiler