#include "spartify/transforms/passes.h"

namespace mlir::spartify_compiler {

void spartifyCodegenPassPipeline(mlir::OpPassManager& pm) {
  pm.addNestedPass<func::FuncOp>(createSpartifyConvertToSparsePass()); 
}

void registerSpartifyPasses() {
  mlir::spartify_compiler::registerPasses(); // generated
}

} // namespace mlir::spartify_compiler