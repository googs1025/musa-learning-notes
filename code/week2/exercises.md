# Week 2 习题

> 完成后把答案 / 截图放到 `notes/week2.md`。
> 本周题目都跟"性能数字"挂钩，凡是要填 ms 的请用 `03_vector_add_timer` 里学到的 **GpuTimer + 多次取平均** 方法，别用裸 CPU 计时。

## E2.1 vectorAdd launch config 扫描（基础）

修改 `01_vector_add_runtime.mu`，固定 `N = 1<<20`，把 `threadsPerBlock` 依次改成 `32 / 64 / 128 / 256 / 512 / 1024`。

填这张表：

| threadsPerBlock | blocksPerGrid | kernel time (ms) |
|---|---|---|
| 32 | ? | ? |
| 64 | ? | ? |
| ... | ... | ... |
| 1024 | ? | ? |

回答：
- 最快的那档是多少？跟 `03_device_info` 里 `maxThreadsPerBlock / warpSize` 有什么关系？
- 为什么 32 那档反而慢——画一下 occupancy 怎么算的。

## E2.2 pinned 收益随 N 的变化（动手）

修改 `02_vector_add_pinned.mu`，把 `N` 依次设成 `1<<16 / 1<<18 / 1<<20 / 1<<22 / 1<<24`。

| N | bytes | pageable H2D (GB/s) | pinned H2D (GB/s) | speedup |
|---|---|---|---|---|
| 1<<16 | 256 KB | ? | ? | ? |
| ... | ... | ... | ... | ... |
| 1<<24 | 64 MB | ? | ? | ? |

回答：
- pinned 的带宽随 N 怎么变？什么时候才"打满 PCIe"？
- 小 N 时 pinned 收益反而不明显，为什么？（提示：固定开销）

## E2.3 三种计时方式的"陷阱重现"（思考 + 验证）

在 `03_vector_add_timer.mu` 里：
- (a) 把 [A] 的 kernel 换成一个 **几乎不干活的 kernel**（比如只 `out[0] = in[0]`），再跑一次，[A] 的数字跟 [B][C] 差多少？
- (b) 把 [B] 的 `musaDeviceSynchronize` 改成 `musaStreamSynchronize(0)`，结果会变吗？为什么？
- (c) 故意把 GpuTimer 的 `stop` 事件放在 D2H 拷贝**之后**而不是 kernel 之后，量到的数字怎么变？

写一段话总结："什么时候 CPU clock 够用，什么时候必须用 musaEvent"。

## E2.4 Unified Memory 在这台机器的"退化"（边界）

在 AutoDL MUSA 3.1.0 上 `musaMemPrefetchAsync` 不支持，`04_vector_add_unified.mu` 跑出来是：

```
冷启动 1.75 ms → 热启动 1.53 ms （prefetch 分支退化为普通 UM 访问）
```

- (a) 把 N 加大到 `1<<24` / `1<<26`，冷热启动差距会变大还是变小？记录数字。
- (b) 把 `musaMallocManaged` 改成 `musaMalloc` + 显式 `musaMemcpy`，同样 N 下 kernel 时间是多少？UM 比显式拷贝慢多少？
- (c) 解释：为什么 prefetch **缺失** 时，UM 的"第一次 kernel"会更慢？（提示：page fault → host→device migration）

## E2.5 多流到底要开几个（动手）

修改 `05_multi_stream.mu`，把流数 `NSTREAMS` 依次设成 `1 / 2 / 4 / 8 / 16`，固定总数据量。

| NSTREAMS | 总耗时 (ms) | 相对单流加速 |
|---|---|---|
| 1 | ? | ×1 |
| 2 | ? | ? |
| ... | ... | ... |
| 16 | ? | ? |

回答：
- 加速比在哪个 NSTREAMS 后开始**不再线性**？这跟你机器上有几个 copy engine 有关系吗？
- 我们实测 4 流只快 1.14×（5.94 → 5.21 ms），猜一下为什么这么少——是 H2D 没并行，还是 kernel 没并行，还是 D2H 没并行？设计一个最小实验区分这三种可能。

## E2.6 给跨流 DAG 加一个分支（动手）

在 `06_stream_event_dep.mu` 的基础上，把依赖图从

```
streamA: H2D_A → kernel_A
                    ↓ event
streamB:           kernel_B(use A's result)
```

扩成：

```
streamA: H2D_A → kernel_A ──event1──┐
streamB: H2D_B → kernel_B ──event2──┤
                                    ↓
streamC:                       kernel_C(use A and B)
```

写出来，verify 结果正确。回答：
- streamC 上要 `musaStreamWaitEvent` 几次？
- 如果把 event1 / event2 都创建成 `musaEventDisableTiming`，对性能有影响吗？为什么？

## E2.7 Graph 为什么反而比 direct launch 慢（思考）

我们实测：

```
direct launch  25000 次       :  410 ms   (16.4 µs/launch)
graph  launch  5000 step × 5  :  677 ms   (135 µs/step)
```

- (a) 把 `07_musa_graph.mu` 里每个 step 的 op 数从 5 改成 1 / 20 / 100，重新跑，填一张 "ops/step vs graph_us_per_step / direct_us_per_op" 的表。
- (b) 在什么 ops/step 阈值上 graph 才追平 direct launch？这个阈值在 NVIDIA CUDA 上一般是多少（查文档）？
- (c) 写一段话回答："如果你今天在 MUSA 上做 LLM inference，**该不该**用 Graph？"

> 不用提 issue，只在 `notes/week2.md` 里留个观察记录就够了。

## E2.8 Stream callback 完成顺序（基础）

`08_stream_callback.mu` 输出：

```
[cb] chunk 0 done
[cb] chunk 3 done
[cb] chunk 2 done
[cb] chunk 1 done
```

- (a) 解释：为什么 chunk 提交顺序是 0/1/2/3，回调顺序却不是？
- (b) 如果想让 callback 严格按 chunk 编号顺序触发，应该怎么改？（提示：要么 1 个流，要么 callback 里加锁/计数）
- (c) 在 callback 函数里调用 `musaMemcpy` 会怎样？（提示：文档里说不行，亲自试一下看报什么错）

## E2.9 拿数字证明"三阶段重叠"真的发生了（动手）

`05_multi_stream.mu` 现在只打印总耗时，看不出 H2D / kernel / D2H 到底有没有 overlap。

- (a) 在每个 stream 上加 `musaEventRecord`：start / after-H2D / after-kernel / after-D2H 四个点。
- (b) 算每个 stream 的三段耗时，画出来（手画 timeline 或者用 Python 画甘特图都行）。
- (c) 如果发现"H2D 之间是串行的"，说明 copy engine 只有 1 个 / 或者 H2D 没用 pinned host buffer——验证一下是哪种。

> 这题做出来基本就理解了"为什么实测 4 流只快 1.14×"。

## E2.10 综合：streamed 累加（综合 / 选做）

写一个新文件 `09_streamed_reduce.mu`（不进 `CMakeLists.txt` 也行，先放本地）：

- 输入：长度 `N = 1<<26` 的 float 数组，按 chunk 分成 `K = 8` 份。
- 用 4 个流交替做：H2D chunk_i → reduce_kernel chunk_i → D2H partial_sum_i。
- 最后 host 把 8 个 partial_sum 加起来，跟 `std::accumulate` 对拍。

要求：
- (a) 跟 "单流串行 + 单次大 H2D + 单 kernel + 单 D2H" 比时间，给出加速比。
- (b) 如果用 `musaMallocManaged` 替代 pinned host + device 显式拷贝，能省代码吗？性能差多少？（结合 E2.4 的结论）
- (c) 把这个 pipeline 改成用 `musaGraph` 表达，跑一遍看 graph launch 在"重活"上是不是终于赢过 direct launch（呼应 E2.7）。

完成后给一句话总结："Stream / Event / Graph 三者各自的最佳使用场景是什么"。
