# MUSA Runtime API 速查

> 常用 Runtime API 的 **真实 signature(从 `musa_runtime_api.h` 直接抓)+ 一句话用法 + 出现的示例编号**。
> 用于编码时快速查参数顺序,**不是**教程——讲解放在对应的 `code/weekN/*.mu` 里。

## 通用约定

- 所有 API 在 `<musa_runtime.h>` 里(实际声明在 `musa_runtime_api.h`,前者会自动包含)
- 返回类型基本都是 `musaError_t`,**0 = `musaSuccess`,非 0 = 失败**
- 失败后用 `musaGetErrorString(err)` 拿可读消息
- 用 `MUSA_CHECK(call)`(见 `code/include/musa_common.h`)统一包一层,失败 fprintf + exit
- 名字以 `musa*` 开头 = Runtime API;`mu*` 开头 = Driver API(在 `<musa.h>`,本表不收)

---

## 1. 设备管理

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaGetDeviceCount` | `musaError_t musaGetDeviceCount(int *count)` | 系统有几块 GPU。Week 1 · #03 |
| `musaGetDevice` | `musaError_t musaGetDevice(int *device)` | 当前 host 线程在用哪块。Week 2 · #04 |
| `musaSetDevice` | `musaError_t musaSetDevice(int device)` | 切设备。多卡程序每个线程开头调一次 |
| `musaGetDeviceProperties` | `musaError_t musaGetDeviceProperties(musaDeviceProp *prop, int device)` | 取 SM 数 / 显存 / warp 大小等。Week 1 · #03 |
| `musaDeviceSynchronize` | `musaError_t musaDeviceSynchronize(void)` | 阻塞 host,等本设备所有 stream 跑完。Week 1 · #06 |
| `musaDeviceReset` | `musaError_t musaDeviceReset(void)` | 清空设备状态,主要用于退出前 / 测试隔离 |
| `musaMemGetInfo` | `musaError_t musaMemGetInfo(size_t *free, size_t *total)` | 查可用 / 总显存,适合在 OOM 排查时打日志 |

---

## 2. 错误处理

| API | Signature | 用途 |
|---|---|---|
| `musaGetLastError` | `musaError_t musaGetLastError(void)` | **取并清** 最近一个错误。Kernel launch 后必调,抓同步错误。Week 1 · #05 |
| `musaPeekAtLastError` | `musaError_t musaPeekAtLastError(void)` | 取但**不清**。debug 时想多看几遍同一个 error 用这个 |
| `musaGetErrorString` | `const char* musaGetErrorString(musaError_t)` | error code → 可读字符串 |
| `musaGetErrorName` | `const char* musaGetErrorName(musaError_t)` | error code → 枚举名(比如 `"musaErrorMemoryAllocation"`) |

> **kernel launch 之后的"标准两步"**:
> ```cpp
> kernel<<<g,b>>>(...);
> musaGetLastError();        // 同步错误(launch 配置非法)
> musaDeviceSynchronize();   // 异步错误(kernel 执行期崩溃)
> ```
> 仓库里直接用 `MUSA_CHECK_KERNEL()` 宏一行搞定。

---

## 3. 显存:分配 / 释放

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaMalloc` | `musaError_t musaMalloc(void **devPtr, size_t size)` | 分配 device 显存。Week 1 · #04 |
| `musaFree` | `musaError_t musaFree(void *devPtr)` | 释放 `musaMalloc` 的指针 |
| `musaMallocPitch` | `musaError_t musaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)` | 分配 2D 显存,行有 padding。Week 4 转置示例会用 |
| `musaMallocManaged` | `musaError_t musaMallocManaged(void **devPtr, size_t size, unsigned int flags = musaMemAttachGlobal)` | 统一内存,host/device 同指针。Week 2 · #04 |

> 二级指针 `void**` 是 CUDA / MUSA 风格——返回 error code 给函数返回值,真正的指针走 out-参数。

---

## 4. 显存:Memcpy / Memset

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaMemcpy` | `musaError_t musaMemcpy(void *dst, const void *src, size_t count, musaMemcpyKind kind)` | **同步**拷贝。`kind` 取 `musaMemcpyHostToDevice` 等 4 种。Week 1 · #04 |
| `musaMemcpyAsync` | `musaError_t musaMemcpyAsync(void *dst, const void *src, size_t count, musaMemcpyKind kind, musaStream_t stream = 0)` | **异步**拷贝。要求 host 端 pinned 才是真异步。Week 2 · #05 |
| `musaMemset` | `musaError_t musaMemset(void *devPtr, int value, size_t count)` | **按字节**填充。`musaMemset(p, 1, 4)` 不是把 float 设成 1.0 |
| `musaMemsetAsync` | `musaError_t musaMemsetAsync(void *devPtr, int value, size_t count, musaStream_t stream = 0)` | 同上,异步版 |

> `musaMemcpyKind`:`musaMemcpyHostToDevice` / `DeviceToHost` / `DeviceToDevice` / `HostToHost`。
> 不知道方向也可以填 `musaMemcpyDefault`(driver 根据指针推断,需要支持 UVA 的硬件)。

---

## 5. Pinned Host Memory

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaMallocHost` | `musaError_t musaMallocHost(void **ptr, size_t size)` | 经典 API,分配 page-locked host 内存。Week 2 · #02 #05 #08 |
| `musaHostAlloc` | `musaError_t musaHostAlloc(void **pHost, size_t size, unsigned int flags)` | 带 flag 的高级版,可选 `musaHostAllocWriteCombined` / `Mapped` / `Portable` |
| `musaFreeHost` | `musaError_t musaFreeHost(void *ptr)` | 释放上面两者分配的内存。**不能用 `free()`!** |
| `musaHostRegister` | `musaError_t musaHostRegister(void *ptr, size_t size, unsigned int flags)` | 把已有的 `malloc` 内存"钉"成 pinned(不重新分配) |

> `WriteCombined`:host 端只写、GPU 读的 staging,带宽再涨 10-30%,但 host 读会非常慢。

---

## 6. 统一内存(Managed Memory)

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaMallocManaged` | 见 §3 | 一份指针 host/device 共用 |
| `musaMemPrefetchAsync` | `musaError_t musaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, musaStream_t stream = 0)` | 把 managed 内存预搬到目标设备,避免 page fault。**某些硬件不支持,返回 `musaErrorNotSupported`(801),要软失败**。Week 2 · #04 |
| `musaMemAdvise` | `musaError_t musaMemAdvise(const void *devPtr, size_t count, musaMemoryAdvise advice, int device)` | 提示访问模式(`ReadMostly` / `PreferredLocation` / `AccessedBy`) |

---

## 7. Stream

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaStreamCreate` | `musaError_t musaStreamCreate(musaStream_t *pStream)` | 创建非默认流。Week 2 · #05 |
| `musaStreamCreateWithFlags` | `musaError_t musaStreamCreateWithFlags(musaStream_t *pStream, unsigned int flags)` | 带 flag(`musaStreamNonBlocking` 等) |
| `musaStreamDestroy` | `musaError_t musaStreamDestroy(musaStream_t stream)` | 销毁。**不会等流跑完**,如果还有任务在跑请先 sync |
| `musaStreamSynchronize` | `musaError_t musaStreamSynchronize(musaStream_t stream)` | 阻塞 host,等指定流跑完。Week 2 · #05 |
| `musaStreamWaitEvent` | `musaError_t musaStreamWaitEvent(musaStream_t stream, musaEvent_t event, unsigned int flags = 0)` | **device 侧依赖**:让 stream 等 event 触发(不阻塞 host)。Week 2 · #06 |
| `musaStreamQuery` | `musaError_t musaStreamQuery(musaStream_t stream)` | 非阻塞查询流是否空闲。返回 `musaSuccess` = 已空 |

> kernel 启动时把 stream 传进去:`kernel<<<grid, block, sharedBytes, stream>>>(...)`。`sharedBytes` 是动态 shared memory 大小,不用填 0。

---

## 8. Event

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaEventCreate` | `musaError_t musaEventCreate(musaEvent_t *event)` | 创建事件。Week 2 · #03 #06 |
| `musaEventDestroy` | `musaError_t musaEventDestroy(musaEvent_t event)` | 销毁 |
| `musaEventRecord` | `musaError_t musaEventRecord(musaEvent_t event, musaStream_t stream = 0)` | **在 stream 当前位置插标记**,GPU 跑到这里才触发。不是"立刻打时间戳"。Week 2 · #06 |
| `musaEventSynchronize` | `musaError_t musaEventSynchronize(musaEvent_t event)` | host 阻塞,等 event 真正触发 |
| `musaEventQuery` | `musaError_t musaEventQuery(musaEvent_t event)` | 非阻塞查 event 是否已触发 |
| `musaEventElapsedTime` | `musaError_t musaEventElapsedTime(float *ms, musaEvent_t start, musaEvent_t end)` | 两 event 之间的毫秒数,~µs 精度。`GpuTimer` 的核心 |

---

## 9. Graph(Stream Capture)

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaStreamBeginCapture` | `musaError_t musaStreamBeginCapture(musaStream_t stream, musaStreamCaptureMode mode)` | 开始录制。`mode` 常用 `musaStreamCaptureModeGlobal`。Week 2 · #07 |
| `musaStreamEndCapture` | `musaError_t musaStreamEndCapture(musaStream_t stream, musaGraph_t *pGraph)` | 结束录制,产出 graph |
| `musaGraphCreate` | `musaError_t musaGraphCreate(musaGraph_t *pGraph, unsigned int flags)` | 手动建空 graph(不用 capture 时) |
| `musaGraphInstantiate` | `musaError_t musaGraphInstantiate(musaGraphExec_t *pGraphExec, musaGraph_t graph, musaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize)` | graph 编译成可执行实例。最后 4 个参数全填 nullptr / 0 即可 |
| `musaGraphLaunch` | `musaError_t musaGraphLaunch(musaGraphExec_t graphExec, musaStream_t stream)` | 反复跑这个 |
| `musaGraphExecDestroy` | `musaError_t musaGraphExecDestroy(musaGraphExec_t graphExec)` | 销毁实例 |
| `musaGraphDestroy` | `musaError_t musaGraphDestroy(musaGraph_t graph)` | 销毁 graph 本体 |

> 录制期间 kernel **不真跑**,只是把 launch 记进 DAG。仓库 #07 PART III Q2 专门讲这个踩坑。

---

## 10. Host 回调 / Launch Host Function

| API | Signature | 用途 / 出现位置 |
|---|---|---|
| `musaStreamAddCallback` | `musaError_t musaStreamAddCallback(musaStream_t stream, musaStreamCallback_t callback, void *userData, unsigned int flags)` | 流完成到这里时回调 host 函数。`flags` 必填 0。Week 2 · #08 |
| `musaLaunchHostFunc` | `musaError_t musaLaunchHostFunc(musaStream_t stream, musaHostFn_t fn, void *userData)` | 新接口,签名更干净;可被 Stream Capture 录进 Graph。生产代码偏好这个 |

callback typedef:
```cpp
typedef void (MUSART_CB *musaStreamCallback_t)(
    musaStream_t stream,
    musaError_t  status,
    void*        userData);

typedef void (MUSART_CB *musaHostFn_t)(void *userData);
```

> **回调里禁止调任何 `musa*` API**——driver 持锁,reentry 立刻死锁。

---

## 11. Kernel Launch:`<<<>>>` vs `musaLaunchKernel`

99% 场景直接用三角括号:
```cpp
kernel<<<grid, block, sharedBytes, stream>>>(arg1, arg2, ...);
```

需要"在 host 侧动态选 kernel"(JIT、按配置切版本)才用底层 API:

```cpp
musaError_t musaLaunchKernel(
    const void *func,         // kernel 函数指针(取地址)
    dim3 gridDim,
    dim3 blockDim,
    void **args,              // 指向"参数指针数组"
    size_t sharedMem,
    musaStream_t stream);
```

只声明,日常不用。

---

## 参考

- 头文件本体:`/usr/local/musa/include/musa_runtime_api.h`(全部 signature 的真相)
- [`cuda-vs-musa.md`](cuda-vs-musa.md) · CUDA → MUSA 命名对照与迁移
- [`concepts.md`](concepts.md) · SIMT / 硬件 / 内存层级
- [`glossary.md`](glossary.md) · 术语小词典
- 官方编程指南:<https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/>
