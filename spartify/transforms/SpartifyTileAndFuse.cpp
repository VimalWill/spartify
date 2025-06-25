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

void SpartifyTileAndFuse::runOnOperation() {
  auto moduleOp = getOperation();
  if (!moduleOp)
    return;
}

std::unique_ptr<Pass> createSpartifyTileAndFusePass() {
  return std::make_unique<SpartifyTileAndFuse>();
}
} // namespace mlir::spartify_compiler