#include "spartify/transforms/passes.h"

namespace mlir::spartify_compiler {

void spartifyCodegenPassPipeline(mlir::OpPassManager& pm) {
  pm.addNestedPass<func::FuncOp>(createSpartifyConvertToSparsePass()); 
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCCPPass());
  

  pm.addPass(mlir::createLowerSparseOpsToForeachPass(false, true)); 
  pm.addPass(mlir::createLowerForeachToSCFPass()); 
}

void registerSpartifyPasses() {
  mlir::spartify_compiler::registerPasses(); // generated
}

} // namespace mlir::spartify_compiler