// ============================================================================
//  文件：04_memory_basics.mu
//  标题：Week 1 · 示例 4 · 显存三件套
//  目标：搞清楚 host 和 device 是两片独立内存，必须显式拷贝
//        掌握最小四步：musaMalloc / musaMemset / musaMemcpy / musaFree
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 1.1 ──────────────────────────────────────────────────────────────────
// │  CPU 和 GPU 是两片独立的内存
// └──────────────────────────────────────────────────────────────────────────
//
//  GPU 不能直接访问 host 上 malloc/new 出来的指针，
//  反过来 host 也不能解引用 musaMalloc 返回的指针。
//
//      float *h_ptr = (float*)malloc(N * sizeof(float));   // host 内存
//      float *d_ptr; musaMalloc(&d_ptr, N * sizeof(float));// device 显存
//
//      d_ptr[0] = 1.0f;        // ✗ Segmentation fault（host 解引用 device 指针）
//      kernel<<<...>>>(h_ptr); // ✗ illegal address（kernel 用 host 指针）
//
//  规则：host 指针只在 host 用，device 指针只在 device 用。
//        要在两侧之间搬数据，必须 musaMemcpy。

// ┌─ § 1.2 ──────────────────────────────────────────────────────────────────
// │  显存四个基本动作
// └──────────────────────────────────────────────────────────────────────────
//
//      musaMalloc(&d_ptr, bytes)                分配显存
//      musaMemset(d_ptr, byteVal, bytes)        把每个字节置为某值（注意是字节！）
//      musaMemcpy(dst, src, bytes, kind)        拷贝（4 种方向）
//      musaFree(d_ptr)                          释放显存
//
//  musaMemcpy 的 kind：
//      musaMemcpyHostToDevice    H → D
//      musaMemcpyDeviceToHost    D → H
//      musaMemcpyDeviceToDevice  D → D
//      musaMemcpyHostToHost      H → H（极少用，等价于 memcpy）
//
//  → musaMemcpy 默认是同步的，等数据搬完才返回。

// ┌─ § 1.3 ──────────────────────────────────────────────────────────────────
// │  musaMemset 的"字节级"陷阱
// └──────────────────────────────────────────────────────────────────────────
//
//  和标准 memset 一样：第二个参数是 unsigned char，按字节填充。
//
//      musaMemset(d, 0, N*4);     // ✓ 把 N 个 float 全部清 0
//      musaMemset(d, 1, N*4);     // ✗ 不是把每个 float 置 1，
//                                 //   而是每个字节置 0x01 → float = 2.36e-38
//
//  → 想把 float 数组初始化为某常数，得自己写 kernel（本例就是这么干的）。

// ┌─ § 1.4 ──────────────────────────────────────────────────────────────────
// │  GPU 程序的最小骨架（"6 步法"）
// └──────────────────────────────────────────────────────────────────────────
//
//      1. 算 bytes，host 端 malloc 用来收结果
//      2. musaMalloc 分配 device 内存
//      3. （可选）musaMemcpy H→D 把输入送进显存
//      4. kernel<<<g, b>>>(d_ptr, ...) 计算
//      5. musaMemcpy D→H 把结果搬回 host
//      6. musaFree + free 释放
//
//  本例没有"输入"，只让 GPU 把 d_ptr 全部填成同一个常数后拷回，
//  把流程压到最短，先建立"双内存 + 显式 memcpy"的肌肉记忆。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
//  编译：make          运行：./04_memory_basics

#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>

// 错误检查宏（同 03_device_info）
#define CHECK(call) do {                                       \
    musaError_t _e = (call);                                   \
    if (_e != musaSuccess) {                                   \
        fprintf(stderr, "MUSA error %d at %s:%d\n",            \
                (int)_e, __FILE__, __LINE__);                  \
        return 1;                                              \
    }                                                          \
} while (0)

// ── Kernel：把数组每个元素填成同一个常数 ──
__global__ void fill_const(float* d, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程编号
    if (i < n) {                                     // 边界保护
        d[i] = v;
    }
}

int main() {
    const int   N     = 16;                          // 数组长度
    const size_t BYTES = N * sizeof(float);
    const float VAL   = 3.14f;                       // 要填的常数

    // ── 1. host 端 buffer：用来收 GPU 算出来的结果 ──
    float* h = (float*)malloc(BYTES);

    // ── 2. device 端分配显存 ──
    float* d = nullptr;
    CHECK(musaMalloc(&d, BYTES));

    // ── 3. 先把显存清 0（演示 musaMemset；按字节，不是按元素）──
    CHECK(musaMemset(d, 0, BYTES));

    // ── 4. launch kernel：填常数 ──
    int threads = 64;
    int blocks  = (N + threads - 1) / threads;       // 向上取整
    fill_const<<<blocks, threads>>>(d, N, VAL);
    CHECK(musaGetLastError());                       // 检查 launch 配置错误
    CHECK(musaDeviceSynchronize());                  // 等 kernel 跑完 + 检查执行错误

    // ── 5. D → H 拷回结果 ──
    CHECK(musaMemcpy(h, d, BYTES, musaMemcpyDeviceToHost));

    // ── 6. host 端验证 ──
    int ok = 1;
    for (int i = 0; i < N; ++i) {
        if (h[i] != VAL) { ok = 0; break; }
    }
    printf("h[0..3] = %.2f %.2f %.2f %.2f\n", h[0], h[1], h[2], h[3]);
    printf("verify: %s\n", ok ? "PASS" : "FAIL");

    // ── 7. 释放 ──
    CHECK(musaFree(d));
    free(h);
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  练习题与解答（对应 exercises.md E1.6 / E1.7）                ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: 直接 host 解引用 d 会怎样？
// ──────────────────────────────────────────────────────────────────────────
//
//  把 musaMemcpy 那一行删掉，改成：
//      printf("%f\n", d[0]);
//
//  ▸ 答案：Segmentation fault。
//      d 指向显存地址空间，对 CPU 来说就是个非法地址。
//      这是新手最常见的踩坑：Run 没报错的"指针都是地址"直觉，
//      到了异构编程立刻失效。
//
//  ▸ 调试技巧：用变量命名约定区分 ── h_ 前缀给 host 指针，d_ 前缀给 device 指针。

// ★ Q2: musaMemset(d, 1, BYTES) 之后 d[0] 是 1.0 吗？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 答案：不是。
//      musaMemset 按字节，每个 float 被填成 0x01010101，
//      解释为 float ≈ 2.369e-38（一个非常小的正数）。
//
//  ▸ 想让数组等于 1.0：
//      a) 写 kernel 用 fill_const(d, N, 1.0f)，本例就是这么做
//      b) host 端 for 循环写 1.0f 后 H→D 拷过来
//      c) thrust::fill / 算法库（更高级，第 5 周再说）

// ★ Q3 (扩展): 忘了 musaFree 会泄漏显存吗？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 答案：进程退出时操作系统/驱动会回收，所以单次跑不会出事。
//          但服务/训练循环里不释放就会持续涨显存，最终 OOM。
//
//  ▸ 习惯：
//      • 显存 RAII：用 std::unique_ptr + 自定义 deleter，或封 thrust::device_vector
//      • 长任务定期 musaMemGetInfo 监控 free / total
//      • 出错路径也要走释放（CHECK 宏目前 return 1 会泄漏，生产代码该用 RAII）
// ============================================================================
