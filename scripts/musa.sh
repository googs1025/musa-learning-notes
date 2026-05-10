#!/usr/bin/env bash
# musa.sh — 本地编辑 + 远程跑的工作流脚本(适配每次新 AutoDL 机器)
#
# 常用命令(每换一台新机器跑前两步):
#   ./scripts/musa.sh switch <hostname> [port]   # 切换到新机器(自动推公钥免密)
#   ./scripts/musa.sh bootstrap                  # 远程一次性环境准备
#   ./scripts/musa.sh run 01_hello_world         # 同步 + 编译 + 跑
#
# 其他:
#   sync   只同步      build  同步+编译     shell  进远程
#   clean  清远程build pull   远程→本地     status 显示当前配置
#
# 环境变量(覆盖默认):
#   REMOTE_USER  默认 root
#   REMOTE_DIR   默认 /root/musa-learning-notes
#   MUSA_PATH    默认 /usr/local/musa
#   SSH_KEY      默认 ~/.ssh/id_ed25519(找不到回落 id_rsa)

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/musa-learning-notes}"
MUSA_PATH="${MUSA_PATH:-/usr/local/musa}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CONFIG="$LOCAL_DIR/scripts/ssh_config"
KNOWN_HOSTS="$LOCAL_DIR/scripts/.known_hosts"
ALIAS="autodl"

c()   { printf '\033[1;36m%s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
err() { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }

# 找到 SSH 私钥(优先 ed25519, 回落 rsa)
pick_key() {
  if [[ -f "$SSH_KEY" ]]; then echo "$SSH_KEY"; return; fi
  if [[ -f "$HOME/.ssh/id_rsa" ]]; then echo "$HOME/.ssh/id_rsa"; return; fi
  err "找不到 SSH 私钥; 用 ssh-keygen -t ed25519 生成一个"
  exit 1
}

require_config() {
  if [[ ! -f "$SSH_CONFIG" ]]; then
    err "scripts/ssh_config 不存在,先跑:"
    err "  ./scripts/musa.sh switch <hostname> [port]"
    exit 1
  fi
}

_ssh()   { ssh -F "$SSH_CONFIG" "$@"; }
_rsync() { rsync -e "ssh -F $SSH_CONFIG" "$@"; }
_scp()   { scp -F "$SSH_CONFIG" "$@"; }

RSYNC_EXCLUDES=(
  --exclude='.git/'
  --exclude='build/'
  --exclude='cmake-build-*/'
  --exclude='.idea/'
  --exclude='.vscode/'
  --exclude='.DS_Store'
  --exclude='*.o'
  --exclude='*.so'
  --exclude='__pycache__/'
  --exclude='.cache/'
  --exclude='compile_commands.json'
  --exclude='scripts/ssh_config'
  --exclude='scripts/.known_hosts'
)

# ─────────────────────────────────────────────────────────────
# 切换到新机器 — 写本地 ssh_config, 推公钥免密(只输 1 次密码)
# ─────────────────────────────────────────────────────────────
cmd_switch() {
  local host="${1:-}"
  local port="${2:-22}"
  if [[ -z "$host" ]]; then
    err "用法: $(basename "$0") switch <hostname> [port]"
    err "示例: $(basename "$0") switch connect.sha1.seetacloud.com 13491"
    exit 1
  fi

  local key; key=$(pick_key)
  local pubkey="${key}.pub"
  if [[ ! -f "$pubkey" ]]; then
    err "找不到公钥 $pubkey"; exit 1
  fi

  c "→ 写 $SSH_CONFIG"
  # ControlPath 必须放短路径(macOS unix socket 上限 104 字节)
  # /tmp/musa-%C 中 %C 是 hash, 安全且短
  cat > "$SSH_CONFIG" <<EOF
# Auto-managed by scripts/musa.sh — 不要手改
# 用 ./scripts/musa.sh switch <hostname> [port] 切换机器
Host $ALIAS
    HostName $host
    Port $port
    User $REMOTE_USER
    IdentityFile $key
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    UserKnownHostsFile $KNOWN_HOSTS
    ServerAliveInterval 60
    ConnectTimeout 10
    AddKeysToAgent yes
    # ── 连接复用:第一次连通后,30 分钟内所有命令免重复输密码/passphrase ──
    ControlMaster auto
    ControlPath /tmp/musa-%C
    ControlPersist 30m
EOF
  chmod 600 "$SSH_CONFIG"

  # 清掉同 host:port 的旧 hostkey(防止 MITM 警告)
  ssh-keygen -f "$KNOWN_HOSTS" -R "[$host]:$port" 2>/dev/null || true
  ssh-keygen -f "$KNOWN_HOSTS" -R "$host"          2>/dev/null || true

  c "→ 推送公钥(输一次密码)"
  # 不依赖 ssh-copy-id(各平台版本差异), 自己拼.
  # chmod 700 ~ 是为了通过 sshd 的 StrictModes 检查(AutoDL /root 默认有时是 755)
  if cat "$pubkey" | _ssh -o ControlMaster=no -o ControlPath=none \
       -o PubkeyAuthentication=no -o PreferredAuthentications=password "$ALIAS" \
       "mkdir -p ~/.ssh && \
        chmod 700 ~ ~/.ssh && \
        cat >> ~/.ssh/authorized_keys && \
        awk '!seen[\$0]++' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp && \
        mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys && \
        chmod 600 ~/.ssh/authorized_keys"; then
    ok "✓ 公钥已推送"
  else
    err "✗ 推送失败,可能是密码错或 host/port 不对"
    exit 1
  fi

  c "→ 验证连接"
  if _ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none "$ALIAS" \
       'echo "  ✓ 全自动(pubkey + 无 passphrase 或 agent 已加载):" "$(uname -srm)"' 2>/dev/null; then
    ok "✓ 已切换到 $host:$port"
  else
    # BatchMode 失败有两种可能,提示用户对症处理
    printf '\033[1;33m%s\033[0m\n' "  ⚠ 全自动连接失败,可能原因之一:"
    printf '\033[1;33m%s\033[0m\n' "      a) AutoDL 不接受 pubkey   → 后续每个命令输密码"
    printf '\033[1;33m%s\033[0m\n' "      b) 你的私钥有 passphrase  → 后续每个命令输 passphrase"
    printf '\033[1;33m%s\033[0m\n' "  → 强烈建议:ssh-add --apple-use-keychain $key"
    printf '\033[1;33m%s\033[0m\n' "    一次解锁后永久免输(macOS Keychain 保管)"
    printf '\033[1;33m%s\033[0m\n' "  → 或退一步:ControlMaster 已开,30 分钟内只问 1 次"
    ok "✓ 已切换到 $host:$port (连接复用 30m)"
  fi
  echo
  c "下一步: ./scripts/musa.sh bootstrap"
}

# ─────────────────────────────────────────────────────────────
# 远程一次性准备(幂等) — 探测 SDK / 装 cmake / 写 .musa_env / 创建项目目录
# ─────────────────────────────────────────────────────────────
cmd_bootstrap() {
  require_config

  c "→ [1/4] 探测 MUSA SDK"
  _ssh "$ALIAS" "set -e
    if [[ ! -x $MUSA_PATH/bin/mcc ]]; then
      echo '  ✗ 找不到 $MUSA_PATH/bin/mcc' >&2
      echo '    试试: ls /usr/local | grep musa  ' >&2
      echo '    然后: MUSA_PATH=<对的路径> ./scripts/musa.sh bootstrap' >&2
      exit 1
    fi
    echo \"  ✓ \$($MUSA_PATH/bin/mcc --version 2>&1 | head -1)\""

  c "→ [2/4] 探测 cmake / rsync"
  _ssh "$ALIAS" "
    if ! command -v cmake >/dev/null; then
      echo '  cmake 缺失,apt 安装中...'
      apt-get update -qq && apt-get install -y -qq cmake
    fi
    echo \"  ✓ cmake: \$(cmake --version | head -1)\"
    command -v rsync >/dev/null || (apt-get install -y -qq rsync)
    echo '  ✓ rsync 在'"

  c "→ [3/4] 写远程 ~/.musa_env(每次 bootstrap 覆盖)"
  local tmpfile; tmpfile=$(mktemp)
  cat > "$tmpfile" <<EOF
# Auto-managed by scripts/musa.sh
export MUSA_PATH=$MUSA_PATH
export PATH=\$MUSA_PATH/bin:\$PATH
export LD_LIBRARY_PATH=\$MUSA_PATH/lib:\${LD_LIBRARY_PATH:-}
EOF
  _scp -q "$tmpfile" "$ALIAS:~/.musa_env"
  rm -f "$tmpfile"
  _ssh "$ALIAS" "
    grep -q 'source ~/.musa_env' ~/.bashrc || echo 'source ~/.musa_env' >> ~/.bashrc
    grep -q 'source ~/.musa_env' ~/.profile 2>/dev/null || echo 'source ~/.musa_env' >> ~/.profile
    echo '  ✓ ~/.musa_env 已就绪 (.bashrc/.profile 已 source)'"

  c "→ [4/4] 创建项目目录 $REMOTE_DIR"
  _ssh "$ALIAS" "mkdir -p $REMOTE_DIR && echo '  ✓ 就绪'"

  ok "✓ Bootstrap 完成"
  echo
  c "下一步: ./scripts/musa.sh run 01_hello_world"
}

# ─────────────────────────────────────────────────────────────
# 同步
# ─────────────────────────────────────────────────────────────
cmd_sync() {
  require_config
  c "→ rsync  $LOCAL_DIR/  →  $ALIAS:$REMOTE_DIR/"
  _ssh "$ALIAS" "mkdir -p $REMOTE_DIR"
  _rsync -az --delete "${RSYNC_EXCLUDES[@]}" \
    "$LOCAL_DIR/" "$ALIAS:$REMOTE_DIR/"
}

cmd_pull() {
  require_config
  c "← rsync  $ALIAS:$REMOTE_DIR/  →  $LOCAL_DIR/"
  _rsync -az "${RSYNC_EXCLUDES[@]}" \
    "$ALIAS:$REMOTE_DIR/" "$LOCAL_DIR/"
}

# ─────────────────────────────────────────────────────────────
# 编译 / 运行
# ─────────────────────────────────────────────────────────────
cmd_build() {
  cmd_sync
  c "→ remote cmake build (MUSA_PATH=$MUSA_PATH)"
  _ssh "$ALIAS" "set -e
    [[ -f ~/.musa_env ]] && source ~/.musa_env
    cd $REMOTE_DIR/code
    cmake -B build -DMUSA_PATH=$MUSA_PATH
    cmake --build build -j"
}

cmd_run() {
  local name="${1:-}"
  if [[ -z "$name" ]]; then
    err "用法: $(basename "$0") run <target>  (如 01_hello_world)"; exit 1
  fi
  cmd_build
  c "→ remote run: $name"
  _ssh "$ALIAS" "set -e
    [[ -f ~/.musa_env ]] && source ~/.musa_env
    cd $REMOTE_DIR/code
    bin=\$(find build -maxdepth 3 -type f -perm -u+x -name '$name' | head -1)
    if [[ -z \"\$bin\" ]]; then echo 'binary 未找到: $name' >&2; exit 1; fi
    echo \"--- run \$bin ---\"
    ./\"\$bin\""
}

cmd_shell() {
  require_config
  c "→ ssh  $ALIAS  (cd $REMOTE_DIR/code)"
  _ssh -t "$ALIAS" "cd $REMOTE_DIR/code && exec \$SHELL -l"
}

cmd_clean() {
  require_config
  c "→ remote: rm -rf $REMOTE_DIR/code/build"
  _ssh "$ALIAS" "rm -rf $REMOTE_DIR/code/build" && ok "✓ 已清理"
}

cmd_status() {
  echo "LOCAL_DIR     = $LOCAL_DIR"
  echo "SSH_CONFIG    = $SSH_CONFIG"
  echo "REMOTE_USER   = $REMOTE_USER"
  echo "REMOTE_DIR    = $REMOTE_DIR"
  echo "MUSA_PATH     = $MUSA_PATH"
  echo "SSH_KEY       = $(pick_key 2>/dev/null || echo '<缺失>')"
  echo
  if [[ -f "$SSH_CONFIG" ]]; then
    echo "── 当前 ssh_config ──"
    grep -E '^\s*(HostName|Port|User)' "$SSH_CONFIG" | sed 's/^/  /'
    echo
    if _ssh -o BatchMode=yes -o ConnectTimeout=5 "$ALIAS" 'true' 2>/dev/null; then
      ok "✓ 远程可免密连通"
    else
      err "✗ 远程不通(或没免密)"
    fi
  else
    err "ssh_config 未生成,先跑 switch"
  fi
}

usage() {
  cat <<EOF
用法: $(basename "$0") <command> [args]

▌新机器流程(每次换 AutoDL 跑这两步,一次密码后全免密):
  switch <host> [port]    切换到新机器,自动推公钥免密
  bootstrap               远程环境一次性准备(SDK / cmake / .musa_env)

▌日常:
  run <target>            同步 + 编译 + 跑(如 01_hello_world)
  build                   只同步 + 编译
  sync                    只同步
  shell                   ssh 到远程 code/ 目录
  clean                   清远程 build/
  pull                    远程 → 本地(罕用)
  status                  显示当前配置 + 连通性

▌环境变量:
  REMOTE_USER=$REMOTE_USER
  REMOTE_DIR=$REMOTE_DIR
  MUSA_PATH=$MUSA_PATH
  SSH_KEY=$SSH_KEY
EOF
}

case "${1:-}" in
  switch)            shift; cmd_switch    "$@" ;;
  bootstrap)         shift; cmd_bootstrap "$@" ;;
  sync)              shift; cmd_sync      "$@" ;;
  pull)              shift; cmd_pull      "$@" ;;
  build)             shift; cmd_build     "$@" ;;
  run)               shift; cmd_run       "$@" ;;
  shell)             shift; cmd_shell     "$@" ;;
  clean)             shift; cmd_clean     "$@" ;;
  status)            shift; cmd_status    "$@" ;;
  ""|help|-h|--help) usage ;;
  *)                 usage; exit 1 ;;
esac
