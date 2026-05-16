# ============================================================================
#  顶层 CMakeLists  ──  musa-learning-notes
#
#  目标:用一个 CMake 把所有 week 子目录串起来,提供"一键构建全部示例"。
#        Makefile 仍然保留,IDE 党(VS Code/CLion)走 CMake 体验更好。
#
#  用法:
#     cd code
#     cmake -B build -DMUSA_PATH=/usr/local/musa
#     cmake --build build -j
#     ./build/week1/01_hello_world
#
#  也可以只编单个 target:
#     cmake --build build --target 01_hello_world
# ============================================================================

cmake_minimum_required(VERSION 3.20)

# ── 1. 必须在 project() 之前指定 mcc 为 CXX 编译器 ──
#    .mu 本质是带 __global__ 扩展的 C++,直接走 mcc 就行
if(NOT CMAKE_CXX_COMPILER)
    if(DEFINED ENV{MUSA_PATH})
        set(CMAKE_CXX_COMPILER "$ENV{MUSA_PATH}/bin/mcc"
            CACHE FILEPATH "MUSA C++ compiler" FORCE)
    else()
        set(CMAKE_CXX_COMPILER "/usr/local/musa/bin/mcc"
            CACHE FILEPATH "MUSA C++ compiler" FORCE)
    endif()
endif()

project(musa_learning_notes LANGUAGES CXX)

# ── 2. 通用 C++ 设定 ──
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── 3. .mu 扩展名当 C++ 处理 ──
#    让 add_executable(foo foo.mu) 自动按 CXX 走 mcc
list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS mu)

# ── 4. MUSA SDK 路径 + include + lib ──
if(NOT DEFINED MUSA_PATH)
    if(DEFINED ENV{MUSA_PATH})
        set(MUSA_PATH $ENV{MUSA_PATH} CACHE PATH "MUSA SDK install path")
    else()
        set(MUSA_PATH "/usr/local/musa" CACHE PATH "MUSA SDK install path")
    endif()
endif()

message(STATUS "MUSA_PATH        = ${MUSA_PATH}")
message(STATUS "CXX (mcc)        = ${CMAKE_CXX_COMPILER}")

include_directories(
    ${MUSA_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
link_directories(${MUSA_PATH}/lib)

# 公共链接库
set(MUSA_LIBS musa musart)

# ── 5. 帮助函数:add_musa_executable(name source...) ──
#    保证每个示例都自动链上 MUSA runtime + 公共头
#
#    .mu 语言识别策略(诊断验证后的最终版):
#      1) set_source_files_properties LANGUAGE CXX
#         让 CMake 知道 .mu 走 CXX 编译流程(否则 "Cannot determine link language")
#         代价:CMake 会自动加 -x c++,这会关掉 mcc 的 MUSA 前端
#
#      2) 仅在 mcc 上额外加 -x musa
#         clang/mcc 里多个 -x 后者覆盖前者,-x musa 重新打开 MUSA 前端
#         (经 mcc -x musa 测试验证,等价于 nvcc 的 -x cu)
function(add_musa_executable name)
    add_executable(${name} ${ARGN})
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CXX)
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
    if(CMAKE_CXX_COMPILER MATCHES "mcc")
        target_compile_options(${name} PRIVATE -x musa)
    endif()
    target_link_libraries(${name} PRIVATE ${MUSA_LIBS})
    target_include_directories(${name}
        PRIVATE ${CMAKE_SOURCE_DIR}/include)
endfunction()

# ── 6. 各周子目录 ──
add_subdirectory(week1)
add_subdirectory(week2)
# add_subdirectory(week3)   # 待添加

# ── 7. 练习题:LeetGPU 移植 ──
add_subdirectory(leetgpu/easy)
