set(WEEK5_EXAMPLES
    01_shared_basics
    02_reduce_shared
    03_transpose_shared
    04_stencil_constant
    05_naive_gemm
    06_tiled_gemm
)
foreach(name IN LISTS WEEK5_EXAMPLES)
    add_musa_executable(${name} ${name}.mu)
endforeach()
