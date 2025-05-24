set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc --gen-pass-decls)

add_public_tablegen_target(MLIRSpartifyTransformsIncGen)

add_mlir_library(SpartifyPasses
  passes.cpp
  SpartifyConvertToSparse.cpp

  DEPENDS
  MLIRSpartifyTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

target_include_directories(SpartifyPasses
  PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)