// ============================================================================
//  文件:08_stream_callback.mu
//  标题:Week 2 · 示例 8 · Stream Callback · GPU 完成时 host 回调
//  目标:在流水线里把 GPU 完成事件"通知"给 host 端逻辑
//        理解回调的执行时机和禁止动作
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 8.1 ──────────────────────────────────────────────────────────────────
// │  为什么需要 callback
// └──────────────────────────────────────────────────────────────────────────
//
//  多流流水线(#05)里,每个 chunk 跑完 D2H 后,host 端往往还要做事:
//    • 把结果写文件 / 发到下游服务
//    • 触发下一批数据准备
//    • 更新进度条 / 日志
//
//  传统做法:host 起一个线程,musaStreamSynchronize 等流完成 → 然后干事。
//  问题:每条流要起一个线程,代码乱、资源浪费。
//
//  Stream Callback 思路:让 driver 在 stream 完成到某个位置时,**主动回调**
//  host 端的一个函数,host 不需要常驻一个等待线程。

// ┌─ § 8.2 ──────────────────────────────────────────────────────────────────
// │  API 与签名
// └──────────────────────────────────────────────────────────────────────────
//
//      musaStreamAddCallback(stream, callback, userData, 0);
//
//  callback 函数签名(头文件 musa_runtime_api.h 里的 typedef):
//
//      typedef void (MUSART_CB *musaStreamCallback_t)(
//          musaStream_t stream,
//          musaError_t  status,
//          void*        userData);
//
//  注意:
//    • flags 当前**必须填 0**(MUSA / CUDA 都这样规定)
//    • userData 是 callback 唯一能"带参数进去"的方式(指向你自己的结构体)
//    • callback 在 driver 线程上跑,不是 main 线程

// ┌─ § 8.3 ──────────────────────────────────────────────────────────────────
// │  在 callback 里能做什么 / 不能做什么
// └──────────────────────────────────────────────────────────────────────────
//
//  ✓ 能做:
//    • atomic 累加计数器、入条件变量、push 到无锁队列
//    • printf / fprintf 调试日志(stdio 本身线程安全)
//    • 设置 std::promise / 通知 std::condition_variable
//
//  ✗ 绝对不能:
//    • 调任何 musa* API(包括 musaMalloc / musaMemcpy / 又一个 launch)
//      → driver 在 callback 上下文里持着内部锁,reentry 立刻死锁
//    • 跑很重的 CPU 工作(回调会延迟 driver 的下一个调度)
//    • 抛异常穿过 C 边界(driver 接不住,undefined behavior)
//
//  实践:callback 只做"通知",真正的工作让 main 线程或独立 worker 来做。

// ┌─ § 8.4 ──────────────────────────────────────────────────────────────────
// │  本例的演示
// └──────────────────────────────────────────────────────────────────────────
//
//  起 4 个 chunk,每个 chunk 跑一个 kernel 后挂 callback。callback 里:
//    • atomic 累加全局计数器
//    • printf "chunk X done"
//
//  main 线程等所有 stream 都 sync,然后看计数器是不是 4。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include "musa_common.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>

__global__ void fill_with(float* x, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

// userData 的载荷:每个 chunk 一份,告诉 callback 自己是几号 + 全局计数器在哪
struct CbCtx {
    int                  chunk_id;
    std::atomic<int>*    done_counter;
};

// ── 这就是回调:driver 在 stream 跑完前面任务时调用它 ──
static void MUSART_CB on_chunk_done(musaStream_t /*stream*/,
                                    musaError_t  status,
                                    void*        userData) {
    auto* ctx = static_cast<CbCtx*>(userData);
    int done = ctx->done_counter->fetch_add(1) + 1;
    std::printf("[cb] chunk %d done  (status=%d)  total_done=%d\n",
                ctx->chunk_id, static_cast<int>(status), done);
    // ★ 这里禁止调用 musa* API。只做轻量通知 / 计数。
}

int main() {
    const int    CHUNKS = 4;
    const int    N      = 1 << 18;             // 每 chunk 1M 元素
    const size_t bytes  = N * sizeof(float);

    float* d[CHUNKS];
    musaStream_t s[CHUNKS];
    CbCtx ctx[CHUNKS];

    std::atomic<int> done_counter{0};

    for (int i = 0; i < CHUNKS; ++i) {
        MUSA_CHECK(musaMalloc(&d[i], bytes));
        MUSA_CHECK(musaStreamCreate(&s[i]));
        ctx[i].chunk_id     = i;
        ctx[i].done_counter = &done_counter;
    }

    const int tpb = 256;
    const int bpg = (N + tpb - 1) / tpb;

    // ── 派发:每个 chunk 一个 kernel + 一个 callback ──
    for (int i = 0; i < CHUNKS; ++i) {
        fill_with<<<bpg, tpb, 0, s[i]>>>(d[i], N, static_cast<float>(i));
        MUSA_CHECK(musaStreamAddCallback(s[i], on_chunk_done, &ctx[i], 0));
    }

    // ── host 在这里等所有 stream 完成(也会等回调跑完)──
    for (int i = 0; i < CHUNKS; ++i) {
        MUSA_CHECK(musaStreamSynchronize(s[i]));
    }

    std::printf("\nall chunks finished. counter=%d (expect %d)  %s\n",
                done_counter.load(), CHUNKS,
                done_counter.load() == CHUNKS ? "OK ✓" : "FAILED ✗");

    for (int i = 0; i < CHUNKS; ++i) {
        MUSA_CHECK(musaStreamDestroy(s[i]));
        MUSA_CHECK(musaFree(d[i]));
    }
    return done_counter.load() == CHUNKS ? 0 : 1;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  Q&A                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: 多 stream 的 callback 触发顺序保证吗?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 同一个 stream 内:严格 FIFO,挂了多个 callback,按入队顺序触发。
//
//  ▸ 不同 stream 之间:**无序**。两个流上的 kernel 哪个先跑完,callback 就先触发。
//    本例每次跑出来的 "chunk X done" 顺序可能都不一样,这是正常现象。
//
//  ▸ 如果你需要"chunk 0..3 按编号顺序写文件",不要靠 callback 顺序,
//    而是在 callback 里把 chunk_id 推进一个队列,let main 线程拉队列按号处理。
//
//  ▸ 实测样本(AutoDL MTT,CHUNKS=4,各跑在独立 stream 上):
//      [cb] chunk 0 done  (status=0)  total_done=1
//      [cb] chunk 3 done  (status=0)  total_done=2
//      [cb] chunk 2 done  (status=0)  total_done=3
//      [cb] chunk 1 done  (status=0)  total_done=4
//      all chunks finished. counter=4 (expect 4)  OK ✓
//    提交顺序是 0/1/2/3,完成顺序却是 0/3/2/1,正好印证"跨流无序"——
//    counter 是 std::atomic,所以最终值仍然正确,但顺序不可依赖。

// ★ Q2: 把 callback 里换成 musaMemcpy,会怎样?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 大概率死锁。driver 在调你的 callback 时还持着 stream 调度锁,
//    musaMemcpy 进 driver 后想再拿同一把锁 → 死循环 / hang。
//
//  ▸ 即使不死锁(某些实现做了 reentry 检测),也会 return error,
//    本例的 MUSA_CHECK 会 fprintf + exit。
//
//  ▸ 正确做法:callback 里只标记"chunk X 准备好了",主线程 / worker 线程
//    醒来后再发起 musaMemcpy 把数据搬到下游。

// ★ Q3 (扩展): musaLaunchHostFunc 是什么?跟 callback 的区别?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 新接口:
//      musaLaunchHostFunc(stream, host_fn, userData);
//
//      typedef void (MUSART_CB *musaHostFn_t)(void* userData);   // 只有一个参数
//
//  ▸ 区别:
//      • 函数签名更简洁(没有 stream / status 参数)
//      • 可以被 musaStreamBeginCapture 录进 Graph;callback 不能(它带 status,
//        graph 没有"出错路径"的概念)
//      • 现代代码推荐用 musaLaunchHostFunc;只有要拿到 status / 兼容老代码
//        才用 musaStreamAddCallback
//
//  ▸ 本例为了演示完整签名(包括 status 参数)用 musaStreamAddCallback,
//    生产代码偏好 musaLaunchHostFunc + 自己用 atomic 维护 status。
// ============================================================================
