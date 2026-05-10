# ── Week 1 示例 ──
#  对应官方指南 Ch1–4,主题:Hello World / 线程索引 / 设备 / 显存 / 错误 / 异步
#  顶层已注入 add_musa_executable() 帮助函数,这里只声明示例

set(WEEK1_EXAMPLES
    01_hello_world
    02_thread_index
    03_device_info
    04_memory_basics
    05_error_check
    06_async_kernel
)

foreach(name IN LISTS WEEK1_EXAMPLES)
    add_musa_executable(${name} ${name}.mu)
endforeach()
