# MUSA 环境配置

> 官方安装文档：<https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/install_guide/>
> 本文档是上手要点 + 常见坑，详细步骤以官方为准。

## 三种环境路径（按推荐度排序）

### 路径 A · AutoDL 云实例（推荐新手）

最快路径，0 硬件成本。

1. 打开 <https://www.autodl.com/>，注册登录
2. **算力市场** → 筛选 GPU = **MTT S4000** 或 **MTT S80** 等摩尔线程卡
3. 镜像选择带 **MUSA Toolkit** 预装的版本（一般有 PyTorch + torch_musa）
4. 创建实例 → SSH 登录
5. 直接进 [验证安装](#验证安装)

### 路径 B · 本地 MTT S4000 / S80 主机

物理机或带 PCIe 直通的虚拟机。

1. **驱动**：从 <https://developer.mthreads.com/sdk/download/musa> 下载对应内核版本的 `mt-driver`，按官方 README 安装
2. **DDK** (Device Driver Kit)：与驱动版本匹配的 `musa-ddk`
3. **MUSA Toolkit**：`musa-toolkit-x.y.z` deb / rpm 包
4. **配置环境变量**（加进 `~/.bashrc` 或 `~/.zshrc`）：
   ```bash
   export MUSA_HOME=/usr/local/musa
   export PATH=$MUSA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
   ```
5. `source` 一下，进入 [验证安装](#验证安装)

### 路径 C · Docker 镜像

如果你已有摩尔线程 GPU 主机，但不想污染系统：

```bash
docker pull <官方镜像名，参考官方文档>
docker run --rm -it \
    --device=/dev/mtgpu \
    --device=/dev/mt-mem \
    -v $PWD:/workspace \
    <image> bash
```

> 设备节点名以 `ls /dev | grep mt` 实际为准。

---

## 验证安装

进入实例 / 主机后执行：

```bash
# 1. 编译器是否就位
mcc --version

# 2. 设备是否可见（具体命令以官方为准，常见的有：）
mthreads-gmi          # 类似 nvidia-smi
# 或
musaInfo              # 部分版本叫这个

# 3. 简单 kernel 是否能跑
cd ~/musa-study/code/week1
make
./01_hello_world
```

预期看到类似：

```
CPU: Hello world!
GPU: tid=0 Hello world!
GPU: tid=1 Hello world!
GPU: tid=2 Hello world!
GPU: tid=3 Hello world!
GPU: tid=4 Hello world!
```

---

## 编译命令模板

```bash
# 单文件
mcc -O2 source.mu -o binary

# 多文件
mcc -O2 file1.mu file2.cpp -o binary

# 链接 muBLAS / muDNN 等加速库
mcc -O2 source.mu -o binary -lmublas -lmudnn

# Driver API（需要预先把 kernel 编出 module）
mcc --fatbin kernel.mu -o kernel.fatbin    # 编 device 端
mcc -O2 host.cpp -o binary -lmusa          # 编 host 端，运行时加载 .fatbin
```

> `--fatbin` / 链接库名以官方 `mcc --help` 输出为准。

---

## 常见坑

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `mcc: command not found` | 没 export PATH | `source ~/.bashrc`；检查 `MUSA_HOME` |
| `cannot find -lmusa` | LD_LIBRARY_PATH 没设 | 同上 |
| `MUSA_ERROR_NO_DEVICE` | 驱动没装 / 内核版本不匹配 | `dmesg | grep mt` 看驱动加载情况 |
| `MUSA_ERROR_ILLEGAL_ADDRESS` | kernel 里越界访问 | 用 MUSA GDB 单步定位 |
| `mthreads-gmi` 看不到卡 | 驱动 / 用户组权限 | `groups` 看是否在 `video` 或 `render` 组 |
| 编译报头文件找不到 | include path | `mcc -I$MUSA_HOME/include ...` |

---

## 下一步

- 环境就绪 → 跟 [`code/week1/README.md`](../code/week1/README.md) 跑 Hello World
- 想了解 MUSA 整体心智模型 → 看 [`overview.md`](overview.md)
