set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc --gen-pass-decls)

add_public_tablegen_target(MLIRSpartifyTransformsIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(SpartifyPasses
  passes.cpp 
  SpartifyConvertToSparse.cpp
  SpartifyGenericConversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}
  
  DEPENDS
  MLIRSpartifyTransformsIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFToSPIRV
  MLIRSCFTransforms
  MLIRSPIRVConversion  # <-- Add this
  MLIRSPIRVTransforms  # <-- And possibly this
  MLIRTransformUtils
  ${dialect_libs} 
)

target_include_directories(SpartifyPasses
  PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)