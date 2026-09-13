# Week 5 CUDA reference

这组 .cu 文件来自
[kriegalex/wrox-pro-cuda-c](https://github.com/kriegalex/wrox-pro-cuda-c)，固定
来源 commit 为
63825d64683b644198dd9cb0d4d472d6914d4f72。文件保留上游版权头，只导入
本周指定的 Chapter 05 和 Chapter 07 示例；公共头统一使用
../../week1/cuda-reference/common/common.h。

## 构建

在本目录执行：

```bash
make BACKEND=cuda
make BACKEND=musa MUSA_ARCH=mp_31
make BACKEND=cuda TARGET=chapter05__checkSmemSquare
make BACKEND=musa MUSA_ARCH=mp_31 TARGET=chapter07__my-atomic-add
make BACKEND=cuda TARGET=chapter05__reduceIntegerShfl
make clean
```

BACKEND=cuda 使用 nvcc；BACKEND=musa 使用 mcc -mtgpu 和 --offload-arch。
当前环境没有 CUDA/MUSA SDK，以下状态是源码审阅和构建命令层面的记录，不代表真实编译通过。MUSA Mapping 能否覆盖某个 CUDA runtime API、intrinsic 或 device attribute，必须在目标 SDK/设备上验证。

默认 make all 排除 chapter05__reduceIntegerShfl，因为它直接使用
__shfl_xor、warpSize 和 CUDA 32-lane 假设。该目标仍可显式构建，但属于
optional；运行前应确认 MUSA shuffle intrinsic、warp 宽度和 mask 语义。其余文件也只表示“可尝试双后端”，不承诺相同数值或性能。

## 文件清单

| 文件 | Chapter / 主题 | target | CUDA 状态 | MUSA 状态与限制 |
|---|---|---|---|---|
| chapter05/checkSmemSquare.cu | Ch5 / 方形 block 的 shared 访问与 padding | chapter05__checkSmemSquare | 预计可用，未实编 | 依赖 shared、device 属性和 cudaSharedMemConfig 映射，未验证 |
| chapter05/checkSmemRectangle.cu | Ch5 / 矩形 block 的 shared 访问顺序 | chapter05__checkSmemRectangle | 预计可用，未实编 | 依赖矩形 block 与 shared bank 行为，未验证 |
| chapter05/constantReadOnly.cu | Ch5 / constant 与 read-only 指针 stencil | chapter05__constantReadOnly | 预计可用，未实编 | cudaMemcpyToSymbol、read-only cache 语义需 SDK 验证 |
| chapter05/constantStencil.cu | Ch5 / constant memory stencil | chapter05__constantStencil | 预计可用，未实编 | constant symbol 拷贝、边界和大数组行为未验证 |
| chapter05/reduceInteger.cu | Ch5 / global/shared/unroll reduction | chapter05__reduceInteger | 预计可用，未实编 | 依赖 block size、shared 和旧式 unroll 写法，未验证 |
| chapter05/reduceIntegerShfl.cu | Ch5 / shuffle reduction | chapter05__reduceIntegerShfl | optional，依赖 CUDA shuffle | optional；32-lane warpSize 假设与 MUSA warp 宽度可能不同 |
| chapter07/my-atomic-add.cu | Ch7 / CAS 实现自定义 atomic add | chapter07__my-atomic-add | 预计可用，未实编 | atomicCAS 对整数类型的支持需设备验证 |
| chapter07/atomic-ordering.cu | Ch7 / atomic 与非原子更新顺序 | chapter07__atomic-ordering | 预计可用，未实编 | 并发调度和结果分布依赖设备，不能把一次输出当作排序保证 |
| chapter07/floating-point-accuracy.cu | Ch7 / 浮点表示与误差 | chapter07__floating-point-accuracy | 预计可用，未实编 | 数值结果受编译器、精度模式和设备影响 |
| chapter07/floating-point-perf.cu | Ch7 / float 与 double 传输/计算代价 | chapter07__floating-point-perf | 预计可用，未实编 | 性能高度依赖设备双精度吞吐和内存，不能预填结论 |
| chapter07/fmad.cu | Ch7 / FMAD 融合与数值准确性 | chapter07__fmad | 预计可用，未实编 | FMA 是否融合由编译器选项、架构和表达式决定，需实测 |

上游目录中未列出的文件没有导入。路径名中的 / 会映射为 target 名中的
__，与 Week 1–4 reference 的约定一致。
