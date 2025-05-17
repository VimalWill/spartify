#include "spartify/tools/spartify-opt.h"

int mlir::spartify_compiler::runSpartifyMain(int agrc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::sparse_tensor::SparseTensorDialect>();

  mlir::MLIRContext context(registry);
  return 0;
}