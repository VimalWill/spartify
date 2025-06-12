#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
namespace mlir::spartify_compiler {
std::unique_ptr<Pass> createSpartifyConvertToSparsePass();
void spartifyCodegenPassPipeline(mlir::OpPassManager& pm); 

namespace {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "passes.h.inc"
} // namespace

void registerSpartifyPasses();
} // namespace mlir::spartify_compiler