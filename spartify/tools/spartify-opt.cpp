#include "spartify/tools/spartify-opt.h"

int main(int argc, char **argv) {

  mlir::spartify_compiler::registerSpartifyPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::sparse_tensor::SparseTensorDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect, 
                  mlir::bufferization::BufferizationDialect>();

  mlir::MLIRContext context(registry);
  mlir::registerAllDialects(registry);

  // ref: https://llvm.org/devmtg/2024-04/slides/TechnicalTalks/Amini-DeepDiveOnMLIRInternals.pdf
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry); 
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry); 
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry); 
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry); 
  mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry); 

  context.loadAllAvailableDialects();

  mlir::PassPipelineRegistration<> pipeline(
      "spartify-pass-pipeline", "lowers to sparse data-struct",
      mlir::spartify_compiler::spartifyCodegenPassPipeline);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "sparity-compiler", registry));
}