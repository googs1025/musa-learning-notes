set(ATTENTION_CASES
    01_row_softmax
    02_online_softmax
    03_naive_attention
    04_fused_attention_small_d
    05_flash_attention_mini
)

foreach(name IN LISTS ATTENTION_CASES)
    add_musa_executable(${name} ${name}.mu)
endforeach()
