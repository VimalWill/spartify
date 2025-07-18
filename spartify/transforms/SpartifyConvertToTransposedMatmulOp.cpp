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

// ref:
// https://github.com/iree-org/iree/blob/56ae9ce5a1f4278e8a07a55b709e1d90aaf9bb51/compiler/src/iree/compiler/GlobalOptimization/PropagateLinalgTranspose.cpp#L96-L108
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

  funcOp->walk([&](linalg::GenericOp op) {
    if (isaTransposeOpInterface(op)) {
      llvm::outs() << op->getName() << "\n";
    }
  });
}

std::unique_ptr<Pass> createSpartifyConvertToTransposedMatmulOpPass() {
    return std::make_unique<SpartifyConvertToTransposedMatmulOp>(); 
}
} // namespace mlir::spartify_compiler