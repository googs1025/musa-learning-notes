# ── Week 2 示例 ──
#  对应官方指南 Ch5,主题:vectorAdd / Stream / Event / Graph / Callback
#  顶层已注入 add_musa_executable() 帮助函数,这里只声明示例

set(WEEK2_EXAMPLES
    01_vector_add_runtime
    02_vector_add_pinned
    03_vector_add_timer
    04_vector_add_unified
    05_multi_stream
    06_stream_event_dep
    07_musa_graph
    08_stream_callback
)

foreach(name IN LISTS WEEK2_EXAMPLES)
    add_musa_executable(${name} ${name}.mu)
endforeach()
