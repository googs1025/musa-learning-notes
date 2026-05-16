# ── LeetGPU Easy 解答 ──
#  对照 https://leetgpu.com/challenges 的 Easy 难度题目
#  每个 .mu 自带 main(),编译后可独立运行,做最小 smoke test

set(LEETGPU_EASY_EXAMPLES
    01_vector_add
    02_matrix_multiplication
    03_matrix_transpose
    07_color_inversion
    08_matrix_addition
    09_1d_convolution
    19_reverse_array
    21_relu
    23_leaky_relu
    24_rainbow_table
    31_matrix_copy
    52_silu
    54_swiglu
    62_value_clipping
    63_interleave
    65_geglu
    66_rgb_to_grayscale
    68_sigmoid
)

foreach(name IN LISTS LEETGPU_EASY_EXAMPLES)
    add_musa_executable(${name} ${name}.mu)
endforeach()
