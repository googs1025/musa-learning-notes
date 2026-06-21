set(WEEK4_EXAMPLES
    01_saxpy_bandwidth
    02_offset_access
    03_offset_unrolling
    04_aos_vs_soa
    05_transpose_naive
    06_transpose_padded
)
foreach(name IN LISTS WEEK4_EXAMPLES)
    add_musa_executable(${name} ${name}.mu)
endforeach()
