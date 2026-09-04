# 01 Check Dimension MUSA

来源主题：Tony-Tan/CUDA_Freshman `1_check_dimension`

参考链接：<https://github.com/Tony-Tan/CUDA_Freshman/tree/master/1_check_dimension>

## 学习点

这个案例用最小代码打印四组内置维度变量：

- `threadIdx`: 当前线程在 block 内的位置。
- `blockIdx`: 当前 block 在 grid 内的位置。
- `blockDim`: 每个 block 的形状，由 host launch config 决定。
- `gridDim`: grid 的形状，由 host launch config 决定。

它补充 `../02_thread_index.mu`：主线示例更强调全局索引公式，这个 external case 更强调 launch config 在 host 和 kernel 两侧看到的值是一致的。

## CUDA 到 MUSA 改写点

| CUDA 写法 | MUSA 写法 | 说明 |
|---|---|---|
| `#include <cuda_runtime.h>` | `#include <musa_runtime.h>` | Runtime 头文件替换 |
| `cudaDeviceReset()` | `musaDeviceSynchronize()` | 这个例子只需要等待 printf flush |
| `cudaGetLastError()` | `musaGetLastError()` | launch 后检查配置错误 |
| `<<<grid, block>>>` | `<<<grid, block>>>` | kernel launch 语法保留 |
| `threadIdx/blockIdx/blockDim/gridDim` | 同名 | 内置变量保留 |

## 编译运行

```bash
cd code/week1/external-cases
mcc -O2 -std=c++17 01_check_dimension_musa.mu -o 01_check_dimension_musa -L/usr/local/musa-3.1.0/lib -lmusa -lmusart
./01_check_dimension_musa
```

预期现象：

```text
host grid=(2,1,1)
host block=(3,1,1)
threadIdx=(0,0,0) blockIdx=(0,0,0) blockDim=(3,1,1) gridDim=(2,1,1)
...
```

GPU printf 顺序不固定，但一共应该有 `2 * 3 = 6` 条 kernel 输出。
