// ============================================================================
//  文件：01_mublas_sgemm.mu
//  标题：Week 5 · 示例 1 · muBLAS SGEMM 调用骨架
//  目标：建立"自写 GEMM vs 官方库"的对比实验入口
// ============================================================================
//
//  编译（可选）：make mublas
//
//  本文件故意保守：不同 MUSA SDK 的 muBLAS 头文件和函数签名可能变化。
//  先用本地官方 API 参考补齐 handle/create/sgemm/destroy，再记录真实性能。

#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>

#if __has_include(<mublas.h>)
#include <mublas.h>
#define HAVE_MUBLAS_HEADER 1
#else
#define HAVE_MUBLAS_HEADER 0
#endif

#define MUSA_CHECK(call) do {                                       \
    musaError_t _e = (call);                                   \
    if (_e != musaSuccess) {                                   \
        fprintf(stderr, "MUSA error %d at %s:%d\n",            \
                (int)_e, __FILE__, __LINE__);                  \
        std::exit(1);                                          \
    }                                                          \
} while (0)

int main() {
    const int M = 1024, N = 1024, K = 1024;
    printf("Week 5 muBLAS SGEMM comparison skeleton\n");
    printf("Shape: M=%d N=%d K=%d\n", M, N, K);

#if !HAVE_MUBLAS_HEADER
    printf("\nmuBLAS header not found.\n");
    printf("Next steps on a MUSA SDK machine:\n");
    printf("  1. Confirm the muBLAS include path and library name.\n");
    printf("  2. Fill handle/create/SGEMM/destroy calls from official API docs.\n");
    printf("  3. Compare against ../week3/01_naive_gemm and ../week3/02_tiled_gemm.\n");
    printf("  4. Record results in ../../notes/week5.md.\n");
    return 0;
#else
    printf("\nmuBLAS header found.\n");
    printf("Fill SDK-specific SGEMM invocation here before measuring.\n");
    printf("Record: naive GFLOPS, tiled GFLOPS, muBLAS GFLOPS, ratio to tiled.\n");
    return 0;
#endif
}
