#!/usr/bin/env bash
#
# config名を1つ受け取り、以下をまとめて実行するスクリプト。
#   1. outputs/<config名>/ と outputs/<config名>/logs/ を作成
#   2. タイムスタンプ付きのログパスを決定
#   3. configs/<config名>.yaml を指定して akita_soc_fig2.py を実行
#
# 使い方:
#   bash scripts/run_akita_soc.sh <config名> [akita_soc_fig2.py への追加引数...]
#
# 例:
#   bash scripts/run_akita_soc.sh akita_soc_autapse
#   bash scripts/run_akita_soc.sh akita_soc_delay --seed 2
#
set -euo pipefail

# --- 引数チェック ---------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_akita_soc.sh <config名> [追加引数...]" >&2
  echo "  例: bash scripts/run_akita_soc.sh akita_soc_autapse" >&2
  exit 1
fi

CONFIG_NAME="$1"
shift  # 残りは akita_soc_fig2.py へそのまま渡す追加引数

# --- パス類の決定 ---------------------------------------------------------
# スクリプトの場所からリポジトリルートを特定（どこから呼んでも動くように）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="configs/${CONFIG_NAME}.yaml"
OUT_DIR="outputs/${CONFIG_NAME}"
LOG_DIR="${OUT_DIR}/logs"

# config ファイルが存在するか先に確認
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: 設定ファイルが見つかりません: ${CONFIG_PATH}" >&2
  exit 1
fi

# 出力ディレクトリとログディレクトリを作成
mkdir -p "${LOG_DIR}"

# ログパス（タイムスタンプ付き）
LOG_PATH="${LOG_DIR}/run_$(date +%Y%m%d-%H%M%S).log"

# --- 実行 -----------------------------------------------------------------
# デフォルトの記録条件。追加引数で上書き・追加可能。
DEFAULT_ARGS=(
  --record-hours-range 0 12
  --record-window-ms 600000
  --record-buffer-ms 10000
)

echo "config     : ${CONFIG_PATH}"
echo "out-dir    : ${OUT_DIR}"
echo "log        : ${LOG_PATH}"
echo "extra args : $*"
echo "実行を開始します..."

python scripts/akita_soc_fig2.py \
  --config "${CONFIG_PATH}" \
  --out-dir "${OUT_DIR}" \
  "${DEFAULT_ARGS[@]}" \
  "$@" \
  > "${LOG_PATH}" 2>&1

echo "完了しました。ログ: ${LOG_PATH}"
