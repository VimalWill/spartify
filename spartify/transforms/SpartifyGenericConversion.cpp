#include "spartify/transforms/passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
  if (!funcOp)
    return;

  SmallVector<linalg::LinalgOp> initalGenericCandidate;
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (!linalg::isaContractionOpInterface(linalgOp) &&
        !isa<linalg::MatmulOp>(linalgOp)) {
      initalGenericCandidate.push_back(linalgOp); 
    }
  });

  IRRewriter rewriter(&getContext());
  for (auto candidate : initalGenericCandidate) {
    rewriter.setInsertionPoint(candidate);
    FailureOr<linalg::GenericOp> generizedOp =
        linalg::generalizeNamedOp(rewriter, candidate);
    if (failed(generizedOp)) {
      candidate->emitError("unable to lower to generic op");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> createSpartifyGenericConversionPass() {
  return std::make_unique<SpartifyGenericConversion>();
}
} // namespace mlir::spartify_compiler