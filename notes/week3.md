# Week 3 记录

> 真实结果在 MUSA 环境运行后填写。

| 示例 | 时间 | 正确性 | 观察 |
|---|---:|---|---|
| warp divergence | | | |
| reduce naive | | | |
| reduce unrolling | | | |
| reduce shfl | | | |
| nested hello | | | |
| sum matrix 2d | | | 矩阵按行求和对比；实际数字依设备实测 |
| sum matrix 1d | | | 与 06_sum_matrix_2d 都是矩阵按行求和对比；实际数字依设备实测 |

## 统一运行记录模板

本模板只填写真实执行结果；没有 SDK、编译器或设备时填写“未运行”，不填估计性能。

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc；版本 | sm_XX / mp_XX | cuda / musa | `chapterXX__name` | N、grid/block、warp 等 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：`make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>`，随后记录 `./build/<target>` 的完整输出。
