#include "spartify/tools/spartify-opt.h"

int main(int argc, char **argv) {

  mlir::spartify_compiler::registerSpartifyPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::sparse_tensor::SparseTensorDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::PassPipelineRegistration<> pipeline(
      "spartify-pass-pipeline", "lowers to sparse data-struct",
      mlir::spartify_compiler::spartifyCodegenPassPipeline);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "sparity-compiler", registry));
}