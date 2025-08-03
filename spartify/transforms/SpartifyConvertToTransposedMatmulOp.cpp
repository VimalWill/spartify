#include "mlir/IR/PatternMatch.h"
#include "spartify/transforms/passes.h"

namespace mlir::spartify_compiler {
#define GEN_PASS_DEF_SPARTIFYCONVERTTOTRANSPOSEDMATMULOP
#include "passes.h.inc"

namespace {
class SpartifyConvertToTransposedMatmulOp
    : public impl::SpartifyConvertToTransposedMatmulOpBase<
          SpartifyConvertToTransposedMatmulOp> {
public:
  void runOnOperation() override;
};
} // namespace

static SmallVector<NamedAttribute> getPrunedAttributeList(linalg::LinalgOp op) {
  const StringLiteral memoAttr =
      linalg::LinalgDialect::kMemoizedIndexingMapsAttrName;
  SmallVector<NamedAttribute> prunedAttributeList;
  for (auto attr : op->getDiscardableAttrs()) {
    if (attr.getName() != memoAttr) {
      prunedAttributeList.push_back(attr);
    }
  }
  return prunedAttributeList;
}

LogicalResult
insertTransposedMatmulOp(IRRewriter &rewriter,
                         SmallVector<Operation *> &fuseableCandidate) {

  const int size = fuseableCandidate.size();
  if (fuseableCandidate.empty() || (size < 1))
    return failure();

  int count = 0;
  for (Operation *op : fuseableCandidate) {

    if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
      rewriter.setInsertionPointAfter(matmulOp);

      int idx = (count == size) ? count : count - 1;
      auto transposeOp = dyn_cast<linalg::GenericOp>(fuseableCandidate[idx]);
      if (!transposeOp)
        return failure();

      SmallVector<NamedAttribute> attrs = getPrunedAttributeList(matmulOp);
      SmallVector<Value> newInputs = matmulOp.getInputs();

      newInputs[1] = transposeOp.getInputs()[0];
      auto newOp = rewriter.replaceOpWithNewOp<linalg::MatmulTransposeBOp>(
          matmulOp, matmulOp.getResultTypes(), newInputs,
          matmulOp.getDpsInits(), attrs);

      if (transposeOp->use_empty())
        rewriter.eraseOp(transposeOp);

      return success();
    }
    count++;
  }

  return failure();
}

// pattern for combining linalg.matmul and linalg.transpose to
// linalg.linalg.matmul_transpose_b
static bool isaTransposeOpInterface(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return false;

  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return false;
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (mapRange.size() != 2 || !mapRange.front().isPermutation() ||
      !mapRange.back().isPermutation() || mapRange.front() == mapRange.back()) {
    return false;
  }
  return llvm::hasSingleElement(linalgOp.getBlock()->getOperations());
}

void SpartifyConvertToTransposedMatmulOp::runOnOperation() {
  auto funcOp = getOperation();
  if (!funcOp)
    return;

  SmallVector<Operation *> fuseableCandidate;
  funcOp->walk([&](linalg::LinalgOp op) {
    if (isaTransposeOpInterface(op) || isa<linalg::MatmulOp>(op)) {
      fuseableCandidate.push_back(op.getOperation());
    }
  });

  IRRewriter rewriter(&getContext());
  if (failed(insertTransposedMatmulOp(rewriter, fuseableCandidate))) {
    funcOp->emitError("failed to fuse transpose and matmul");
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createSpartifyConvertToTransposedMatmulOpPass() {
  return std::make_unique<SpartifyConvertToTransposedMatmulOp>();
}
} // namespace mlir::spartify_compiler