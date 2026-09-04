# Week 6 习题

1. 运行 `02_musa_gdb_demo`，记录第一个报错 API 和错误码。

   预计输出 / 预期现象：

   ```text
   Launching intentionally broken kernel: allocated=1024 floats, launched=2048 threads
   MUSA error ... at 02_musa_gdb_demo.mu:...
   ```

   第一个 launch configuration 错误可能由 `musaGetLastError` 暴露；真正的 illegal address 往往在 `musaDeviceSynchronize` 处暴露。以实际第一条报错为准。

2. 用 MUSA SDK 调试器单步定位越界线程。

   预计输出 / 预期现象：

   ```text
   thread/block: ...
   idx >= allocated
   ```

   应能定位到写越界的线程索引。记录触发越界的 `blockIdx/threadIdx`、计算出的全局索引和分配数组长度。

3. 按官方文档开启 Error Dump，运行 `03_error_dump`。

   预计输出 / 预期现象：

   ```text
   Launching intentionally broken kernel: allocated=1024 floats, launched=2048 threads
   MUSA error ... at 03_error_dump.mu:...
   ```

   开启 Error Dump 后，应额外生成 SDK 指定的 dump / log 文件。记录文件路径、错误类型和能否从 dump 中定位到越界位置。

4. 在多卡环境补齐并运行 `01_mccl_allreduce.cpp`。

   预计输出 / 预期现象：

   ```text
   MCCL AllReduce skeleton (Week 6)
   rank=0 nranks=... device=...
   MCCL header found.
   ```

   如果没有 MCCL 头文件，会看到 `MCCL header not found`。补齐真实 allreduce 后，各 rank 的输出 buffer 应得到所有 rank 输入之和。

5. 运行 `04_torch_musa_minimal.py`，记录 torch/torch_musa 版本。

   预计输出 / 预期现象：

   ```text
   torch ...
   musa available: True
   sample ...
   ```

   如果环境没装 `torch_musa` 或没有可用 MUSA 设备，会在 import 或 availability 检查处失败；记录 `torch.__version__`、`torch_musa` 包版本和错误信息。
