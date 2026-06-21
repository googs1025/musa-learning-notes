set(WEEK6_EXAMPLES
    02_musa_gdb_demo
    03_error_dump
)
foreach(name IN LISTS WEEK6_EXAMPLES)
    add_musa_executable(${name} ${name}.mu)
endforeach()
