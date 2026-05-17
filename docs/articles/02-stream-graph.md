# MUSA 学习笔记 (二) · Stream 与 Graph 把吞吐量挤到极限

> 接上一篇 [《我是怎么把第一段代码跑起来的》](01-first-musa-code.md)。
> 这一周的目标是从"能跑"升级到"跑得满"——把 H2D / Kernel / D2H 重叠起来。
> 配套代码:`code/week2/`,对应官方指南:Ch5 编程模型与 Stream。

---

## 这周要解决的问题

Week 1 把 4 件事打了地基:线程模型、显存四件套、错误检查、异步 kernel。但写出来的程序都是"串行三段":先 H2D,再算,再 D2H,GPU 一半时间在等数据。

Week 2 把这些组合起来,目标具体:**写出一个 H2D / Kernel / D2H 真的并行的 vectorAdd**。沿途要解决一连串铺路问题——怎么测 kernel 时间才不被异步骗、为什么 pinned 内存是 Async API 的前提、跨流依赖怎么不靠 sync 表达。最后再看 MUSA Graph 这个"把 launch overhead 压到极限"的特性。

我给自己排了 8 个示例,按从单点到组合的顺序:

| # | 文件 | 学什么 |
|---|---|---|
| 01 | `vector_add_runtime` | Runtime API 的 7 步骨架,第一个完整 vectorAdd |
| 02 | `vector_add_pinned` | pinned vs pageable,H2D 实测加速比 |
| 03 | `vector_add_timer` | CPU clock / CPU+sync / GpuTimer 三种计时方式怎么选 |
| 04 | `vector_add_unified` | Unified Memory + prefetch |
| 05 | `multi_stream` | 4 个 stream 把三阶段并起来 |
| 06 | `stream_event_dep` | event 让跨流形成 DAG 依赖 |
| 07 | `musa_graph` | Stream Capture → Graph,5000 步小 kernel 看 launch overhead |
| 08 | `stream_callback` | GPU 完成时回调 host |

跑的环境跟上周一样:AutoDL · MTT S4000 · MUSA 3.1.0。下面挑这周觉得"学完之后心态变了"的几个点写。

---

## 三种计时方式,只有两种是对的

写完 `01_vector_add_runtime` 之后我想测一下 kernel 跑多久。第一版很自然:

```cpp
auto t0 = chrono::now();
vector_add<<<g, b>>>(...);
double ms = ms_since(t0);     // 0.15 ms
```

跟 Week 1 学的一样,**这个数字是假的**——kernel 是异步入队,这一行只量到了 launch 开销。但具体差多少呢?我写了 `03_vector_add_timer` 把三种姿势并排跑了一遍:

```
[A] CPU clock, NO sync     :   0.1554 ms   ← 量到的其实是 launch 入队耗时
[B] CPU clock + sync       : avg  0.1195   min  0.1180   max  0.1205  (ms)
[C] GpuTimer (musaEvent)   : avg  0.1224   min  0.1168   max  0.1495  (ms)
```

几个我学到的事:

- **[B] 和 [C] 差 ~3 µs(2.4%)**,量级吻合,可以互相印证——这套数字是可信的;
- **[A] 比真值小 30%**——而且方向跟我以前的直觉相反。我一开始以为"不 sync 会量到更多",其实是"量到 launch 那一小段就结束了,根本没等 GPU 算完";
- 这种"假数字"的可怕之处在于,它在 vectorAdd 这种几百 µs 的 kernel 上看着很合理,你不并排对比根本发现不了;
- kernel 越小,[B] 的 sync overhead 占比越大。真要测 µs 级 kernel,只能信 [C]。

从这一节之后,我把所有 PART III Q&A 里跟"ms 数字"挂钩的论断都标了 TODO,等跑完实测再回填——不实测过就别下结论。这一条算是 Week 2 最早攒下来的纪律。

---

## pinned 的 ×3.81

`02_vector_add_pinned` 这个示例最直接,改一行 API 看带宽:

```
pageable H2D :   1.966 ms    7.95 GB/s
pinned   H2D :   0.516 ms   30.29 GB/s
speedup      : ×3.81
```

pinned 那一档基本打满了机器的 PCIe(理论 ~32 GB/s 量级)。pageable 只有 1/4,这中间损失的并不是 PCIe,而是 driver 内部多走的一道:**先 memcpy 到内部的 pinned staging buffer,再 DMA 出去**——多了一次 host 内存搬运。

这个数字我之前在 NVIDIA 文档上见过类似的(×2~×3),但 ×3.81 这么大的差距亲手量一次还是有点震撼。更关键的认知是:**Async API(`musaMemcpyAsync`)给 pageable 内存会自动退化成同步路径**。也就是说,如果你写多流流水线但 host buffer 用 `malloc`,你以为开了 4 个流,实际上还是串行的——这是 `05_multi_stream` 要踩的下一个坑。

---

## Unified Memory 在这台机器上"半残"

`04_vector_add_unified` 本来是冲着 `musaMemPrefetchAsync` 去的:用统一内存写 vectorAdd,host 端塞数据,kernel 跑之前提前 prefetch 回 device,看冷热启动差距。

跑出来一句 info 把我拦了:

```
[info] musaMemPrefetchAsync 不被本设备支持,跳过 prefetch
```

也就是说,**MUSA 3.1.0 上 prefetch 没实现**,我代码里调用会被 wrapper 检测到 ENOSYS 自动跳过。"prefetch 那一版"实际上跟"no prefetch"走同一条路。完整跑出来:

```
[run 1] 冷启动:
  no  prefetch         kernel 1.7515 ms
  want prefetch (n/a)  kernel 1.5135 ms
[run 2] 热启动:
  no  prefetch         kernel 1.5338 ms
  want prefetch (n/a)  kernel 1.5536 ms
```

冷启动确实多花 ~0.2 ms,page fault → host→device migration 的代价能量到。但 prefetch 这条优化路径在这台机器上完全用不了。

这事提醒我两件:

1. **MUSA 现在的成熟度还不能假设跟 CUDA 等价**。一对一 API 映射 ≠ 一对一行为,有些 API 存在但是空实现 / 返回 unsupported。生产代码要做能力探测,不能盲调。
2. **统一内存在生产里本来就慎用**。训练框架(torch_musa)默认走 `musaMalloc` + 显式拷贝,不是因为不会用 UM,而是 UM 的 page fault 是非确定性的——每个 step 的耗时方差会变大,破坏 batch / lr scheduler 的稳态。

这条 prefetch 缺失我先记下来,不急着提 issue,等后面 Week 3/4 实际用到 UM 再回头看。

---

## 4 个 stream 只快 1.14×,跟"理论"差远了

`05_multi_stream` 是这周的"主菜"。理论上把数据切 4 块,每块走自己的 stream,H2D / Kernel / D2H 三件事在 4 个流之间错峰,理论加速比 ~3。我兴致勃勃跑出来:

```
[A] single stream serial :   5.938 ms
[B] 4-stream pipeline    :   5.210 ms
```

**×1.14**。这就让人沮丧了。但回头看也很合理:加速比这么小,只能说明 H2D / Kernel / D2H 这三件事并没有真正并行起来。问题是,**到底是哪一段没并行?** 看总耗时是看不出来的。

这就是为什么我专门把"加 event 拆 timeline"写成习题 E2.9——在每个 stream 上 record 四个点(start / after-H2D / after-kernel / after-D2H),算出每段的实际起止,画出甘特图,才知道是 H2D 之间被 copy engine 数量卡死了,还是 kernel 之间被调度串行化了。这个调查留给跑习题时做,文章里先记下"理论 ×3,实测 ×1.14"这个落差,以及"用 event 才能拆 timeline 找原因"这个方法论。

教训:**性能数字不是"算出来"的,是"量出来"的**。在没拆 timeline 之前,别假设 stream 真的并行了。

---

## Graph 反而比 direct launch **慢**

最让我意外的是 `07_musa_graph`。CUDA 圈子里 Graph 的故事很性感:把 N 个小 kernel 录制成一个静态 DAG,后续 N 次 launch 合成一次 graph launch,LLM 推理常用优化。我按这套理论写完,跑出来:

```
[A] direct launch  5000 steps × 5 ops = 25000 launches : 410.232 ms  (16.41 µs/launch)
[B] graph  launch  5000 steps (each = 5 ops in graph)        : 677.012 ms  (135.40 µs/step)
```

**Graph 比 direct 慢 1.65×**。我反复确认了几遍——verify 是 OK 的,数字也复现稳定。这跟 CUDA 上的直觉完全相反。

掰开看一下数学:

- direct launch 每次 ~16 µs,5 次约 82 µs;
- graph launch 一次 ~135 µs。

也就是说,**MUSA 当前版本 `musaGraphLaunch` 自身的 driver 路径开销 ~135 µs,反而比 5 次直接 launch 加起来还贵**。可能的原因(我现在不能完全确认,只是猜):

- direct launch 是 driver 长期优化过的热路径;
- Graph 是相对新的特性,launch 路径还没优化到位;
- 我每个 graph 里只塞 5 个轻量 kernel,摊不平 graph 的固定开销。

这条结论对我的影响:**短期内不要在 MUSA 上盲套 CUDA 的 Graph 优化**。但 API 本身值得掌握(Capture / Instantiate / Launch 这套范式将来肯定会用),所以代码我还是按完整 demo 写完了,只在 Q&A 和 README 里把"现状"写清楚,不夸大也不抱怨。

习题 E2.7 留了个扫描题:把每个 graph 里的 op 数从 5 加到 20、100,看在多大颗粒度上 graph 才能追平甚至超过 direct。这个数字一拿到,以后再用 MUSA Graph 就有个工程判据了。

---

## Stream callback 的"乱序"是 feature 不是 bug

`08_stream_callback` 跑出来:

```
[cb] chunk 0 done  (status=0)  total_done=1
[cb] chunk 3 done  (status=0)  total_done=2
[cb] chunk 2 done  (status=0)  total_done=3
[cb] chunk 1 done  (status=0)  total_done=4
```

我提交顺序是 0 → 1 → 2 → 3,完成顺序却是 0 → 3 → 2 → 1。第一眼看到我以为是 bug,反应过来这正好印证了"跨 stream 之间无序"——4 个流上的 kernel 哪个先跑完,callback 就先触发,跟入队顺序无关。

这事在工程上的意思:**任何"按 chunk 编号顺序处理"的逻辑都不能依赖 callback 触发顺序**。要么用 1 个 stream(stream 内严格 FIFO),要么 callback 里只把 chunk_id 推队列,让主线程按号拉数据处理。

还有一条容易踩的(没自己验证,文档里写的):**callback 函数体里不能调 MUSA API**——driver 在调你的 callback 时还持着 stream 调度锁,你再进 driver 就死循环了。习题 E2.8 让你亲手验证一下报什么错。

---

## 这周用得最爽的工具:GpuTimer + 多次取平均

写了一个小 RAII 工具,封装 `musaEvent` 计时:

```cpp
class GpuTimer {
public:
    GpuTimer()  { musaEventCreate(&s_); musaEventCreate(&e_); }
    ~GpuTimer() { musaEventDestroy(s_); musaEventDestroy(e_); }
    void start(musaStream_t s = 0) { musaEventRecord(s_, s); }
    void stop (musaStream_t s = 0) { musaEventRecord(e_, s); }
    float elapsed_ms() {
        musaEventSynchronize(e_);
        float ms; musaEventElapsedTime(&ms, s_, e_);
        return ms;
    }
private:
    musaEvent_t s_, e_;
};
```

加上"跑 6 次、丢首次冷启动、剩 5 次取平均"这个固定 pattern,后面所有示例的计时都靠这个。一次写好,贯穿 Week 2 整周。

这个东西放在 `03_vector_add_timer.mu` 里,后续示例需要量时间的都把它复制过去。属于"个人小工具,但帮你建立测量纪律"那一类。

---

## 这周积累的几条经验

- **Async API + pageable 内存 = 没用 Async**。多流流水线第一步必须把 host buffer 换成 `musaMallocHost`,否则一切并行都是幻觉。
- **测 kernel 时间永远用 musaEvent**。CPU clock 不 sync 量到的是 launch 开销,sync 之后又包含 sync overhead——对 µs 级 kernel 都不够准。
- **跨流读写必须显式 event**。靠"碰巧对"的代码在小数据上能跑 100 次,数据规模一变就崩。GPU race 是"概率不可观察"的。
- **总耗时不够,要拆 timeline**。多流没拿到预期加速时,加 event 拆每段,才能定位是哪一段没并行。
- **MUSA ≠ CUDA**。API 一对一映射不代表行为一致。这周亲眼撞到两条:`musaMemPrefetchAsync` 在 3.1.0 上返回 unsupported;`musaGraphLaunch` 比 direct launch 还慢。生产代码要做能力探测,优化套路要重新量化,不能盲套。

---

## 下周计划

跟着官方继续走,Week 3 想啃下来:

- [ ] 性能分析:`mprof` 或等价工具怎么看 timeline,把 Week 2 留下的"4 流为啥只快 1.14×"挖出来;
- [ ] Reduce / Scan:第一次写需要 shared memory + `__syncthreads` 的 kernel;
- [ ] Occupancy 怎么算:MTT=128 这个 warp size 在 register / shared mem 约束下怎么选 block size;
- [ ] 把 LeetGPU 上的 Easy 题继续移植几道(已经在 `code/leetgpu/easy/` 起了头)。

---

## 今天的产出

- 代码:`code/week2/01..08_*.mu`,8 个示例 + Makefile + CMake target + 10 道习题
- 笔记:本文
- 仓库:[github.com/googs1025/musa-learning-notes](https://github.com/googs1025/musa-learning-notes)

下周见。
