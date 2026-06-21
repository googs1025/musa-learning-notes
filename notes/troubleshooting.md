# MUSA 故障排查手册 v0.1

## 快速流程

1. 找到第一个失败的 MUSA API。
2. 区分同步错误和异步错误。
3. 每个 kernel 后加 `MUSA_CHECK_KERNEL()`。
4. 用 printf / MUSA SDK 调试器 / Error Dump 缩小到具体 kernel 和线程。

## 常见错误

| Error | 常见根因 | 检查点 |
|---|---|---|
| InvalidConfiguration | grid/block 越限 | device prop |
| InvalidValue | 指针、size、方向错误 | API 参数 |
| OutOfMemory | 显存不足或泄漏 | allocation size |
| NoDevice | 驱动或容器设备不可见 | mthreads-gmi |
| IllegalAddress | kernel 越界读写 | index + bounds check |
| LaunchTimeout | kernel 卡死或过长 | loop bounds |
