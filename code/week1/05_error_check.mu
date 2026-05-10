// ============================================================================
//  文件：05_error_check.mu
//  标题：Week 1 · 示例 5 · 错误检查的两个时机
//  目标：理解"同步错误"和"异步错误"的差别，写出生产可用的 CHECK 范式
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 1.1 ──────────────────────────────────────────────────────────────────
// │  错误从哪里来：两类完全不同
// └──────────────────────────────────────────────────────────────────────────
//
//  (A) 同步错误（synchronous）
//      API 调用立刻就能判断对错，比如：
//        • musaMalloc 分配失败（OOM）
//        • musaMemcpy 方向参数错
//        • kernel 启动配置非法（block 超过 1024、grid 维度越界等）
//
//      这类错误通过两种方式拿到：
//        • 直接函数返回值：musaError_t e = musaXxx(...)
//        • kernel<<<>>> 没有返回值，要靠 musaGetLastError()
//
//  (B) 异步错误（asynchronous）
//      kernel 启动后异步执行，"运行时崩了"要等同步点才能发现：
//        • illegal address（指针越界、null 指针）
//        • stack overflow
//        • 浮点异常（取决于配置）
//
//      只能通过同步点检测：
//        • musaDeviceSynchronize()
//        • musaStreamSynchronize(stream)
//        • 下一次 musaMemcpy（隐式同步）

// ┌─ § 1.2 ──────────────────────────────────────────────────────────────────
// │  CHECK 宏：所有 musa API 都包一层
// └──────────────────────────────────────────────────────────────────────────
//
//  目标：把 if (err != musaSuccess) ... 那一坨从业务代码里清出去，
//        统一打印 文件名 + 行号 + 错误码，方便定位。
//
//      #define CHECK(call) do {                                       \
//          musaError_t _e = (call);                                   \
//          if (_e != musaSuccess) {                                   \
//              fprintf(stderr, "MUSA error %d at %s:%d\n",            \
//                      (int)_e, __FILE__, __LINE__);                  \
//              return 1;                                              \
//          }                                                          \
//      } while (0)
//
//  生产代码可以再加：
//      • musaGetErrorString(_e) 拿到人类可读的名字
//      • exit(EXIT_FAILURE) / std::abort() 立刻挂掉，避免错上加错

// ┌─ § 1.3 ──────────────────────────────────────────────────────────────────
// │  Kernel 启动后的"标准两步检查"
// └──────────────────────────────────────────────────────────────────────────
//
//      kernel<<<g, b>>>(...);
//      CHECK(musaGetLastError());        // ← 同步错误：launch 配置非法
//      CHECK(musaDeviceSynchronize());   // ← 异步错误：执行期崩溃
//
//  缺哪一步都会"吞错误"：
//    • 只 sync 不 GetLastError：launch 失败 → kernel 根本没跑 → sync 不会报错
//    • 只 GetLastError 不 sync：kernel 越界 → 程序看似正常，下次 musaMemcpy 才挂

// ┌─ § 1.4 ──────────────────────────────────────────────────────────────────
// │  常见错误码（速记）
// └──────────────────────────────────────────────────────────────────────────
//
//      musaSuccess                     0   一切正常
//      musaErrorMemoryAllocation       2   显存不够
//      musaErrorInvalidValue           1   参数非法
//      musaErrorInvalidConfiguration   9   block/grid 超限
//      musaErrorIllegalAddress        700  kernel 越界（异步）
//      musaErrorLaunchOutOfResources   7   寄存器/shared 用太多
//
//  对照表见官方"附录"，跑出来记不住的拿这张表查。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
//  本例故意分别触发 (A) 同步错误 和 (B) 异步错误，
//  用同一套 CHECK 把它们都抓出来。
//
//  编译：make          运行：./05_error_check

#include <musa_runtime.h>
#include <cstdio>

// ── 不退出版 CHECK：只打印不 return，方便单文件演示多种错误 ──
#define CHECK_SOFT(call)                                                  \
    do {                                                                  \
        musaError_t _e = (call);                                          \
        if (_e != musaSuccess) {                                          \
            fprintf(stderr, "  ↳ MUSA error %d at %s:%d\n",               \
                    (int)_e, __FILE__, __LINE__);                         \
        } else {                                                          \
            fprintf(stderr, "  ↳ ok\n");                                  \
        }                                                                 \
    } while (0)

// ── Kernel A：什么也不干（用来当"正常基准"）──
__global__ void noop() {}

// ── Kernel B：故意越界写一个非法指针 → 异步 illegal address ──
__global__ void write_bad(int* p) {
    p[0] = 42;        // 如果 p 是 null 或越界，运行时报错
}

int main() {
    printf("=== 1) 正常 launch（应当 ok）===\n");
    noop<<<1, 32>>>();
    CHECK_SOFT(musaGetLastError());          // 同步：配置 OK
    CHECK_SOFT(musaDeviceSynchronize());     // 异步：执行 OK

    printf("\n=== 2) 配置非法：block 超过 maxThreadsPerBlock ===\n");
    // 故意把 block 大小设成 99999，远超硬件上限 → 同步错误
    noop<<<1, 99999>>>();
    CHECK_SOFT(musaGetLastError());          // ★ 这里能抓住
    CHECK_SOFT(musaDeviceSynchronize());     // 这一步通常也会顺带报同一个错

    printf("\n=== 3) 异步错误：kernel 写 null 指针 ===\n");
    int* d_null = nullptr;                   // 没 musaMalloc，直接传 0
    write_bad<<<1, 1>>>(d_null);
    CHECK_SOFT(musaGetLastError());          // launch 配置 OK，这里通常显示 ok
    CHECK_SOFT(musaDeviceSynchronize());     // ★ 这里才会报 illegal address

    printf("\n=== 4) 显存分配失败：申请 100 TB ===\n");
    void* huge = nullptr;
    // 故意申请远超物理显存的大小
    musaError_t e = musaMalloc(&huge, (size_t)100 * 1024 * 1024 * 1024 * 1024ULL);
    if (e != musaSuccess) {
        fprintf(stderr, "  ↳ musaMalloc failed as expected, code=%d\n", (int)e);
    } else {
        fprintf(stderr, "  ↳ unexpectedly succeeded?!\n");
        musaFree(huge);
    }

    printf("\n=== done ===\n");
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  练习题与解答（对应 exercises.md E1.8）                       ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: 为什么 case 3 中 musaGetLastError() 会显示 ok？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 答案：launch 本身（grid/block 配置）合法，参数也只是个指针值，
//          编译器/运行时无法在 launch 时就判断它是否是非法地址。
//          直到 kernel 真正在 GPU 上跑到 *p = 42 那一行才崩，
//          所以错误必须等下一个同步点才能浮出来。
//
//  ▸ 这就是"异步错误"的本质：错误发生时间和检测时间是分离的。

// ★ Q2: 错误检测被吞会怎样？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 假如 case 2 后忘了检测，继续做 musaMalloc / musaMemcpy：
//      • CUDA/MUSA 把"上一次错误"挂在 thread-local 状态里
//      • 下一个 API 调用会**继承**这个错误状态
//      • 你的报错点会出现在一个看起来无辜的 musaMemcpy 上
//      • 调试时间从 5 分钟变成 5 小时
//
//  ▸ 所以："每个 API 都 CHECK，每次 launch 后两步都查"是底线。

// ★ Q3 (扩展): 怎么把错误码翻译成可读文本？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 用 musaGetErrorString(err)：
//      printf("MUSA error: %s (code=%d)\n", musaGetErrorString(_e), (int)_e);
//
//  ▸ 还有 musaGetErrorName(err) 拿宏名（如 "musaErrorIllegalAddress"），
//          适合写日志或上报。
// ============================================================================
