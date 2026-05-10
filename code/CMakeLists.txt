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
function(add_musa_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE ${MUSA_LIBS})
    target_include_directories(${name}
        PRIVATE ${CMAKE_SOURCE_DIR}/include)
endfunction()

# ── 6. 各周子目录 ──
add_subdirectory(week1)
# add_subdirectory(week2)   # 待添加
# add_subdirectory(week3)   # ...
