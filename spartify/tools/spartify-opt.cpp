#include "spartify/tools/spartify-opt.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "sparity-compiler", registry));
}