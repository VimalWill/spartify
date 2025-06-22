#include "spartify/transforms/passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Passes.h"

namespace mlir::spartify_compiler {
#define GEN_PASS_DEF_SPARTIFYGENERICCONVERSION
#include "passes.h.inc"

namespace {
class SpartifyGenericConversion
    : public impl::SpartifyGenericConversionBase<SpartifyGenericConversion> {
public:
  void runOnOperation() override;
};
} // namespace

void SpartifyGenericConversion::runOnOperation() {
  auto funcOp = cast<func::FuncOp>(getOperation());
  if (!funcOp) return; 

  MLIRContext *ctx = &getContext(); 
  RewritePatternSet patterns(ctx); 

  mlir::linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns); 
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    funcOp.emitError("Failed to apply SpartifyGenericConversion patterns"); 
    return signalPassFailure(); 
  }
}

std::unique_ptr<Pass> createSpartifyGenericConversionPass() {
    return std::make_unique<SpartifyGenericConversion>(); 
}
} // namespace mlir::spartify_compiler