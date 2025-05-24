#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "spartify/transforms/passes.h"

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

void SpartifyConvertToSparse::runOnOperation() {
  auto funcOp = getOperation();
  if (!funcOp)
    return;
}

std::unique_ptr<Pass> createSpartifyConvertToSparsePass() {
  return std::make_unique<SpartifyConvertToSparse>();
}
} // namespace mlir::spartify_compiler