# 远程开发工作流

Mac 本地编辑 + AutoDL 远程编译跑。**适配每次都是新机器的场景**——
两条命令切换到新机，之后全程 0 密码。

> **为什么这样**：Mac 上没 mcc，本地 clang 不认识 `<<<>>>`/`threadIdx`，IDE 跳转
> 只能覆盖纯 host 代码。完整索引/编译/运行必须放到远程。
> 编辑器里 Find Usages、文件内符号跳转不受影响，平时改代码够用。

## 一次性准备（仅这台 Mac 第一次用）

如果 `~/.ssh/id_ed25519` 设了 passphrase，先把它加进 macOS Keychain，
之后 ssh-agent 永久自动解锁，所有 ssh 调用全静默：

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
# 输一次 passphrase,Mac 重启后 agent 自动加载
```

验证：`ssh-add -l` 应能看到 ed25519 一行。这步不做的话，每个远程命令都会
弹 passphrase 提示。

> 没设 passphrase 的密钥可以跳过这步。

## 每次换新机器：2 条命令

```bash
# 1. 切换 + 推公钥(只这一次需要输 AutoDL 给的密码)
./scripts/musa.sh switch connect.sha1.seetacloud.com 13491

# 2. 远程环境一次性准备(检测 SDK / 装 cmake / 写环境变量)
./scripts/musa.sh bootstrap
```

跑完之后：
```bash
./scripts/musa.sh run 01_hello_world      # 全程免密、自动同步、自动编译、跑
```

`switch` 把 SSH 配置写到本地 `scripts/ssh_config`（gitignored，不污染你的
`~/.ssh/config`），并把公钥追加到远程 `~/.ssh/authorized_keys` 去重。
之后所有命令都通过这个项目专属配置走，免密。

> **ControlMaster 兜底**：即便公钥被 sshd 拒（如 `/root` 权限 755 触发
> `StrictModes`），`switch` 写的配置里启用了连接复用——输 1 次密码，30 分钟内
> 所有 sync/build/run 都搭这一条 TCP 连接，**不会再次提示**。

## 命令一览

```
▌新机器流程
  switch <host> [port]   切换到新机器(自动推公钥免密)
  bootstrap              远程一次性环境准备

▌日常(全免密)
  run <target>           同步 + 编译 + 跑(如 01_hello_world)
  build                  只同步 + 编译
  sync                   只同步
  shell                  ssh 到远程 code/ 目录
  clean                  清远程 build/
  pull                   远程 → 本地(罕用)
  status                 显示当前配置 + 连通性
```

## 工作流示意

```
┌──────────────┐  rsync   ┌──────────────┐
│  Mac CLion   │ ───────→ │   AutoDL     │
│  (编辑)      │          │   mcc 编译    │
│  Find Usages │ ←─stdout │   ./binary   │
└──────────────┘          └──────────────┘
        │                        ▲
        └─── ./scripts/musa.sh ──┘
```

## 切到新机器之后的细节

`switch` 会做这些事，你不用记：

1. 在 `scripts/ssh_config` 写新 hostname + port（覆盖旧的）
2. 清掉同 host 的旧 known_hosts 记录（防止 hostkey 警告）
3. 用密码登录一次，把 `~/.ssh/id_ed25519.pub` 追加到远程 `authorized_keys`
4. 用免密登录验证一遍

`bootstrap` 会做这些事，幂等（再跑一次不会重复污染）：

1. 检查 `/usr/local/musa/bin/mcc` 在不在
2. 没 cmake 的话 `apt-get install`
3. 写远程 `~/.musa_env`（含 `MUSA_PATH` / `PATH` / `LD_LIBRARY_PATH`）
4. 在 `~/.bashrc` 和 `~/.profile` 里加一行 `source ~/.musa_env`
5. 创建远程项目目录

## 路径不一致怎么办

AutoDL 不同镜像里 MUSA SDK 路径可能是 `/usr/local/musa-3.1.0` 而不是
`/usr/local/musa`。所有命令都接受 `MUSA_PATH=` 前缀：

```bash
MUSA_PATH=/usr/local/musa-3.1.0 ./scripts/musa.sh bootstrap
MUSA_PATH=/usr/local/musa-3.1.0 ./scripts/musa.sh run 01_hello_world
```

或者在 shell 里 `export` 一下。

## CLion External Tool（可选）

省掉切终端：`Settings → Tools → External Tools → +`

| 字段 | 值 |
|---|---|
| Name | `Run on AutoDL` |
| Program | `$ProjectFileDir$/scripts/musa.sh` |
| Arguments | `run $FileNameWithoutExtension$` |
| Working directory | `$ProjectFileDir$` |

`Settings → Keymap` 给它绑个快捷键（如 `⌃⌥R`），打开任意 `.mu` 一按就远程跑。

## 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| 反复弹 `Enter passphrase for key …` | 私钥有 passphrase 但没进 agent | `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` 一次永逸 |
| `Permission denied (publickey,password)` | hostname/port 错或 AutoDL 改密码了 | 重新 `switch <host> <port>` |
| `unix_listener: path "..." too long` | ControlPath 路径超 macOS 104 字节上限 | 脚本默认 `/tmp/musa-%C` 已规避;如手改过 `scripts/ssh_config` 把它改回去 |
| `bootstrap: 找不到 mcc` | SDK 路径不对 | `MUSA_PATH=...` 覆盖再跑 |
| `cannot open shared object file: libmusart.so` | 非交互 ssh 没 source `.musa_env` | 已自动 source；如仍报，用 `./scripts/musa.sh shell` 进去 `source ~/.musa_env` 跑 |
| `Host key verification failed` | 同 host:port 上次用过别的机器 | `switch` 会自动清，再跑一次即可 |
| `ssh_config 不存在` | 还没 switch 过 | `./scripts/musa.sh switch <host> <port>` |
| rsync 慢/同步多余文件 | 排除清单不够 | 改 `scripts/musa.sh` 里 `RSYNC_EXCLUDES` |

强制断开复用连接：
```bash
ssh -F scripts/ssh_config -O exit autodl
```

## 不用脚本的等价命令

脚本只是包了一层，本质就这两条，懂原理后随时可手敲：

```bash
# 同步
rsync -az --delete --exclude='.git' --exclude='build' --exclude='cmake-build-*' \
  ~/musa-learning-notes/  autodl:/root/musa-learning-notes/

# 远程编译 + 跑
ssh autodl 'cd /root/musa-learning-notes/code && \
  cmake -B build -DMUSA_PATH=/usr/local/musa && \
  cmake --build build -j && \
  ./build/week1/01_hello_world'
```

## 环境变量速查

| 变量 | 默认 | 何时改 |
|---|---|---|
| `MUSA_PATH` | `/usr/local/musa` | SDK 装在别处 |
| `REMOTE_USER` | `root` | 远端不是 root |
| `REMOTE_DIR` | `/root/musa-learning-notes` | 想放别的目录 |
| `SSH_KEY` | `~/.ssh/id_ed25519`，回落 `id_rsa` | 用别的密钥 |
