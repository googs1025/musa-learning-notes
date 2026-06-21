# Week 3 示例:执行模型 / Reduce / 多维 grid
set(WEEK3_EXAMPLES
    01_warp_divergence
    02_reduce_naive
    03_reduce_unrolling
    04_reduce_shfl
    05_nested_hello
    06_sum_matrix_2d
)
foreach(name IN LISTS WEEK3_EXAMPLES)
    add_musa_executable(${name} ${name}.mu)
endforeach()
