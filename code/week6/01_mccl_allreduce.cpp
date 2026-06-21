// ============================================================================
//  文件：01_mccl_allreduce.cpp
//  标题：Week 6 · 示例 1 · MCCL AllReduce 骨架
//  目标：理解多卡 collective 的资源初始化和测量入口
// ============================================================================
//
//  编译（可选）：make mccl
//
//  MCCL 通常需要多进程启动、rank 分配和 out-of-band unique id 分发。
//  本文件是结构骨架，不假设当前机器有多卡或 MCCL 运行环境。

#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>

#if __has_include(<mccl.h>)
#include <mccl.h>
#define HAVE_MCCL_HEADER 1
#else
#define HAVE_MCCL_HEADER 0
#endif

#define CHECK_MUSA(call) do {                                  \
    musaError_t _e = (call);                                   \
    if (_e != musaSuccess) {                                   \
        fprintf(stderr, "MUSA error %d at %s:%d\n",            \
                (int)_e, __FILE__, __LINE__);                  \
        std::exit(1);                                          \
    }                                                          \
} while (0)

int main(int argc, char** argv) {
    int rank = argc > 1 ? std::atoi(argv[1]) : 0;
    int nranks = argc > 2 ? std::atoi(argv[2]) : 1;
    int device = argc > 3 ? std::atoi(argv[3]) : rank;

    printf("MCCL AllReduce skeleton (Week 6)\n");
    printf("rank=%d nranks=%d device=%d\n", rank, nranks, device);

#if !HAVE_MCCL_HEADER
    printf("\nMCCL header not found.\n");
    printf("Next steps on a multi-GPU MUSA machine:\n");
    printf("  1. Confirm mccl.h include path and libmccl link name.\n");
    printf("  2. Generate or receive MCCL unique id out-of-band.\n");
    printf("  3. Set each rank's MUSA device and stream.\n");
    printf("  4. Initialize communicator and call mcclAllReduce.\n");
    printf("  5. Record algbw/busbw and topology notes in ../../notes/week6.md.\n");
    return 0;
#else
    CHECK_MUSA(musaSetDevice(device));
    musaStream_t stream;
    CHECK_MUSA(musaStreamCreate(&stream));

    printf("\nMCCL header found.\n");
    printf("Fill SDK-specific mcclUniqueId exchange, mcclCommInitRank, and mcclAllReduce calls.\n");
    printf("Use MCCL_DEBUG=INFO to inspect topology selection when running real tests.\n");

    CHECK_MUSA(musaStreamDestroy(stream));
    return 0;
#endif
}
