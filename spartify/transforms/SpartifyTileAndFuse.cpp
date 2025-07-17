#include "mlir/Interfaces/TilingInterface.h"
#include "spartify/transforms/passes.h"

namespace mlir::spartify_compiler {
#define GEN_PASS_DEF_SPARTIFYTILEANDFUSE
#include "passes.h.inc"
namespace {
class SpartifyTileAndFuse
    : public impl::SpartifyTileAndFuseBase<SpartifyTileAndFuse> {
public:
  void runOnOperation() override;
};
} // namespace

auto anyTilableOp(func::FuncOp funcOp) {
  bool tilable = false;
  funcOp->walk([&](Operation *op) {
    if (auto tileInteraface = dyn_cast<TilingInterface>(op)) {
      tilable = true;
      llvm::outs() << op->getName() << "\n";
    }
  });

  return tilable;
}
void SpartifyTileAndFuse::runOnOperation() {
  auto funcOp = getOperation();
  if (!funcOp)
    return;

  anyTilableOp(funcOp); 
}

std::unique_ptr<Pass> createSpartifyTileAndFusePass() {
  return std::make_unique<SpartifyTileAndFuse>();
}
} // namespace mlir::spartify_compiler