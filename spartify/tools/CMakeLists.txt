# Get all MLIR dialect and conversion libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(spartify-opt
  spartify-opt.cpp
)

llvm_update_compile_flags(spartify-opt)
target_link_libraries(spartify-opt
  PRIVATE
  ${dialect_libs} 
  ${conversion_libs} 
  MLIROptLib
  SpartifyPasses
  MLIRSCFToSPIRV
  
  
)
