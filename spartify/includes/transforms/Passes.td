#ifndef SPARTIFY_INCLUDES_TRANSFORMS_PASSES_TD
#define SPARTIFY_INCLUDES_TRANSFORMS_PASSES_TD

SpartifyConvertToSparseLayout : Pass<"spartify-convert-to-sparse-layout"> {
    let summary = "converts the dense tensor layout to a sparse tensor layout";
    let constructor = "mlir::spartify_compiler::createSpartifyConvertToSparseLayoutPass()";   
}