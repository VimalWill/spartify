#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"

#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
namespace mlir::spartify_compiler {
std::unique_ptr<Pass> createSpartifyConvertToSparsePass();
std::unique_ptr<Pass> createSpartifyGenericConversionPass(); 
std::unique_ptr<Pass> createSpartifyTileAndFusePass(); 
void spartifyCodegenPassPipeline(mlir::OpPassManager& pm, bool isSparse = true); 

namespace {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "passes.h.inc"
} // namespace

void registerSpartifyPasses();
} // namespace mlir::spartify_compiler