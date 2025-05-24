#include "spartify/transforms/passes.h"

namespace mlir::spartify_compiler {

void registerSpartifyPasses() {
  mlir::spartify_compiler::registerPasses(); // generated
}

} // namespace mlir::spartify_compiler