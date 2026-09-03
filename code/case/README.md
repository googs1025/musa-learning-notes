# Case · Attention / Flash Kernel 手撸路线

这个目录把 attention 相关练习集中成一条独立路线，不再塞进 `week*/`。每个 `.mu` 都包含三块内容：

1. 注解：解释这个 case 要练的 kernel 思维。
2. MUSA kernel：尽量保持短小，先保证数据流清楚。
3. `main()` smoke test：小尺寸输入 + CPU reference + 最大误差。

## 学习顺序

| 文件 | 主题 | 重点 |
|---|---|---|
| `01_row_softmax.mu` | Row-wise softmax | max/sum 两次 block reduction、数值稳定 |
| `02_online_softmax.mu` | Online softmax | 流式维护 `m` 和 `l`，为 FlashAttention 铺垫 |
| `03_naive_attention.mu` | 三段式 attention | `QK^T -> softmax -> PV` 的完整数据流 |
| `04_fused_attention_small_d.mu` | 小维度 fused attention | 不落全量 score/probability 矩阵 |
| `05_flash_attention_mini.mu` | Mini FlashAttention | tile 扫描 K/V，在线更新输出累加器 |

## 编译运行

```bash
cd code
cmake -B build -DMUSA_PATH=/usr/local/musa
cmake --build build -j

./build/case/01_row_softmax
./build/case/02_online_softmax
./build/case/03_naive_attention
./build/case/04_fused_attention_small_d
./build/case/05_flash_attention_mini
```

也可以只在本目录用 Makefile：

```bash
cd code/case
make
./01_row_softmax
```

## 核心公式

稳定 softmax：

```text
m = max(x)
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

Online softmax 合并两个片段：

```text
m_new = max(m_old, m_tile)
l_new = l_old * exp(m_old - m_new) + l_tile * exp(m_tile - m_new)
```

FlashAttention 输出累加器更新：

```text
acc_new = acc_old * (l_old * exp(m_old - m_new) / l_new)
        + acc_tile * (exp(m_tile - m_new) / l_new)
```

这里的 `acc_tile` 是当前 K/V tile 内的 `sum_j exp(score_j - m_tile) * V_j`。

## 推荐观察点

- `01` 和 `02` 的输出应该接近，但 `02` 的 reduction 合并方式更接近 FlashAttention。
- `03` 最容易 debug，因为中间矩阵 `scores` 和 `prob` 都落 global memory。
- `04` 去掉了中间矩阵，适合理解 fusion 的收益和代价。
- `05` 才是 FlashAttention 的学习版核心：按 tile 读 K/V，不保存 `S x S` attention 矩阵。

## 和真实 FlashAttention 的差距

这个目录是教学版，不追求生产性能。真实实现还会继续处理：

- fp16/bf16 输入和 fp32 accumulate
- vectorized load/store
- 更细的 warp/block 任务划分
- causal mask / padding mask
- 多 batch、多 head、变长序列
- register pressure 和 occupancy 调参
