#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "spartify/transforms/passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

namespace mlir::spartify_compiler {
#define GEN_PASS_DEF_SPARTIFYCONVERTTOSPARSE
#include "passes.h.inc"

namespace {
class SpartifyConvertToSparse
    : public impl::SpartifyConvertToSparseBase<SpartifyConvertToSparse> {
public:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::sparse_tensor::SparseTensorDialect,
                    mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();
  }
};
} // namespace

LogicalResult insertSparseTensor(linalg::GenericOp op,
                                 PatternRewriter &rewriter) {

  MLIRContext *ctx = rewriter.getContext();
  auto inputs = op.getInputs();
  if (inputs.size() < 2)
    return failure();

  auto weight = inputs[1];
  if (auto weightType = dyn_cast<RankedTensorType>(weight.getType())) {
    if (weightType.getEncoding()) {
      return failure();
    }
  }
  auto weightAttrOp = weight.getDefiningOp();
  if (!weightAttrOp)
    return failure();

  RankedTensorType srcType = dyn_cast<RankedTensorType>(weight.getType());
  if (!srcType || srcType.getEncoding())
    return failure();

  auto nmType = sparse_tensor::buildLevelType(
      sparse_tensor::LevelFormat::NOutOfM, true, true, 2, 4);
  auto denseType = sparse_tensor::buildLevelType(
      sparse_tensor::LevelFormat::Dense, true, true);

  SmallVector<sparse_tensor::LevelType, 2> levelType;
  levelType.push_back(denseType.value());
  levelType.push_back(nmType.value());

  unsigned rank = srcType.getRank();
  auto idMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
  auto sparseEnc = sparse_tensor::SparseTensorEncodingAttr::get(
      ctx, levelType, idMap, idMap, 32, 32, nullptr, nullptr, {});

  auto sparseType = RankedTensorType::get(srcType.getShape(),
                                          srcType.getElementType(), sparseEnc);
  auto convertOp = rewriter.create<sparse_tensor::ConvertOp>(
      weightAttrOp->getLoc(), sparseType, weight);

  SmallVector<Value> newInputs(inputs.begin(), inputs.end());
  newInputs[1] = convertOp;

  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  auto genericOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(), op.getResultTypes(), newInputs, op.getOutputs(),
      indexingMaps, iterators);
  rewriter.inlineRegionBefore(op->getRegion(0), genericOp.getRegion(),
                              genericOp.getRegion().begin());
  rewriter.replaceOp(op, genericOp->getResults());
  return success();
}

class ConvertToSparsePattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<linalg::CopyOp, linalg::FillOp>(op))
      return failure();

    if (!linalg::isaContractionOpInterface(op))
      return failure();

    auto iteratorType = op.getIteratorTypesArray();
    auto numIteratorType = iteratorType.size();

    if (numIteratorType < 3)
      return failure();

    auto inputs = op.getInputs();
    if (inputs.size() < 2)
      return failure();

    // Check that innermost 3 iterators are 'parallel, parallel, reduction'.
    for (auto i : {2, 3}) {
      if (!linalg::isParallelIterator(iteratorType[numIteratorType - i]))
        return failure();
    }
    if (!linalg::isReductionIterator(iteratorType[iteratorType.size() - 1]))
      return failure();

    return insertSparseTensor(op, rewriter);
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
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    funcOp.emitError("Failed to apply SpartifyConvertToSparse patterns");
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createSpartifyConvertToSparsePass() {
  return std::make_unique<SpartifyConvertToSparse>();
}
} // namespace mlir::spartify_compiler