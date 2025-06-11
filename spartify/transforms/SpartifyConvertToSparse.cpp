#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "spartify/transforms/passes.h"

#include "mlir/Dialect/Linalg/Passes.h"

namespace mlir::spartify_compiler {
#define GEN_PASS_DEF_SPARTIFYCONVERTTOSPARSE
#include "passes.h.inc"

namespace {
class SpartifyConvertToSparse
    : public impl::SpartifyConvertToSparseBase<SpartifyConvertToSparse> {
public:
  void runOnOperation() override;
};
} // namespace

class ConvertToSparsePattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(op))
      return failure();

    auto iteratorType = op.getIteratorTypesArray();
    auto numIteratorType = iteratorType.size();

    if (numIteratorType < 3)
      return failure();

    // Check that innermost 3 iterators are 'parallel, parallel, reduction'.
    for (auto i : {2, 3}) {
      if (!linalg::isParallelIterator(iteratorType[numIteratorType - i]))
        return failure();
    }
    if (!linalg::isReductionIterator(iteratorType[iteratorType.size() - 1]))
      return failure();
  }
};

void SpartifyConvertToSparse::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = cast<func::FuncOp>(getOperation());
  if (!funcOp)
    return;

  RewritePatternSet patterns(context);

  mlir::linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
  patterns.add<ConvertToSparsePattern>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    funcOp.emitError("Failed to apply SpartifyConvertToSparse patterns");
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createSpartifyConvertToSparsePass() {
  return std::make_unique<SpartifyConvertToSparse>();
}
} // namespace mlir::spartify_compiler