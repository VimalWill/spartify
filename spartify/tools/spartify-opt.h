#ifndef SPARTIFY_TOOLS_SPARTIFY_OPT_H
#define SPARTIFY_TOOLS_SPARTIFY_OPT_H

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::spartify_compiler {
int runSpartifyMain(int argc, char **argv);
}

#endif // SPARTIFY_TOOLS_SPARTIFY_OPT_H
