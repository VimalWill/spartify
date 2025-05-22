add_llvm_executable(spartify-opt
  spartify-opt.cpp
)

target_include_directories(spartify-opt 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}
)

llvm_update_compile_flags(spartify-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(spartify-opt 
    PRIVATE 
    ${dialect_libs} 
    ${conversion_libs}
    MLIROptLib
)
