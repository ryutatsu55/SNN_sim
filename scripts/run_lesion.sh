#!/usr/bin/env bash
#
# 親 run を 1 つ受け取り、以下をまとめて実行するスクリプト。
#   1. <親run>/lesion/ と <親run>/lesion/logs/ を作成
#   2. タイムスタンプ付きのログパスを決定
#   3. scripts/lesion.py を実行
#
# 切断条件と記録条件は**このファイル内の設定ブロックで指定する**。CLI で渡すのは親 run だけ。
#
# 使い方:
#   bash scripts/run_lesion.sh <親runディレクトリ>
#
# 例:
#   bash scripts/run_lesion.sh outputs/axon_growth_grid/b20_seg100_gDefF
#
set -euo pipefail

# ==========================================================================
# 設定ブロック — ここを書き換えて実験条件を決める
# ==========================================================================

# --- 切断条件 -------------------------------------------------------------
# CUT_SPECS に切断 spec を並べる。**複数書くと合成される** (既定は or = どちらかに
# 該当すれば切る。CUT_COMBINE=and にすると両方に該当するものだけ切る)。
#
# 指定できる spec は 4 種類:
#
#   bridge:kind=inter_cluster            軸索が実際にクラスタ間ブリッジ(BX)を通った結合
#   bridge:kind=intra_cluster            同じくクラスタ内ブリッジ(BC)
#   bridge:kind=any                      全ブリッジ
#   bridge:parts=BX1-3+BX2-3             ブリッジを名指し (part 名は area.png / layout_axes で確認)
#   bridge:kind=any,samples=64           線分サンプル数 (既定 32)。細いブリッジや長い
#                                        セグメントでは上げる。判定は必ず線分で行う
#     ※ bridge は axon_geometry.npz を要求する (connection: axon_growth 系の run のみ)
#     ※ 判定範囲は soma→接触点まで。接触後に通ったブリッジは数えない
#
#   between:axis=module                  module 軸で別グループ間の結合 (トポロジー的定義)
#   between:axis=module,pairs=C0-M0-C1-M0    グループ対を名指し
#     ※ bridge (幾何的定義) とは本数が一致しない。同じモジュール内に戻ってくる軸索が
#        あるため。どちらの意味で「モジュール間」と言うかは主張に直結する
#
#   hub:metric=participation,top=3,direction=out
#   hub:metric=betweenness,top=3,direction=both
#   hub:metric=out_degree,top=5,direction=out
#     metric    = participation | within_module_z | betweenness |
#                 out_degree | in_degree | degree | out_strength | in_strength
#     top       = 上位何ニューロンを選ぶか
#     direction = out (出力を切る) | in (入力) | both
#     ※ participation は connector hub (モジュールをまたぐハブ) を選ぶ指標
#
#   synapses:pairs=86-194+112-87         (pre,post) を名指し
#   synapses:file=cuts.csv               pre,post 列を持つ CSV
#     ※ 存在しないペアを書くとエラーになる (黙って 0 本切る事故を防ぐため)
#
# 実測の目安 (outputs/axon_growth_grid/b20_seg100_gDefF, 全 1596 本):
#   bridge:kind=inter_cluster  ->  115 本 (7.2%)   クラスタが分断される。本命
#   bridge:kind=intra_cluster  ->  736 本 (46.1%)
#   bridge:kind=any            ->  771 本 (48.3%)  ※ samples=64 推奨 (32 では 1 本取りこぼす)
#   between:axis=module        ->  254 本
#   hub:metric=participation,top=3,direction=out ->  27 本
#   hub:metric=betweenness,top=3,direction=both  ->  47 本
#
CUT_SPECS=(
  # "bridge:kind=inter_cluster"
  # "hub:metric=participation,top=3,direction=out"
  # "between:axis=module"
  "bridge:parts=BX1-3"
)
CUT_COMBINE="or"        # or | and (CUT_SPECS が 2 つ以上のときだけ意味を持つ)

# --- 記録条件 -------------------------------------------------------------
# probe 間隔 1h / 窓 600s は develop.py の記録条件に合わせてある
# (親 run の metrics.csv と同じ土俵で読めるように)。
FROM_HOUR=""            # 引き継ぐ記録時刻 [h]。空なら親 run の最終記録
SETTLE_MS=60000         # 復元直後・切断直後に落ち着かせる時間 [ms]
RECOVERY_HOURS=12       # 切断後に観察する総時間 [h]
PROBE_INTERVAL_HOURS=1  # probe の間隔 [h] (等間隔)
PROBE_WINDOW_MS=600000  # 1 probe の記録窓 [ms]
RECORD_BUFFER_MS=10000  # GeNN のスパイク記録バッファ [ms]

# --- 解析条件 -------------------------------------------------------------
# ハブ判定の within-module degree z 閾値 (Guimera-Amaral の既定 2.5)。
# 2.5 は代謝ネットワーク由来なので、次数の小さい網では下げないとハブが 0 個になる。
# 実測 (b20_seg100_gDefF, N=256, 平均次数 6.2): z の最大 2.60 / 99%点 2.21 なので、
# 2.5 では provincial hub が 2 個。役割別の内訳が空なら 2.0 前後まで下げて確認すること。
HUB_Z=2.5

# --- 実行オプション -------------------------------------------------------
# DRY_RUN=1 にすると GeNN を触らず、切断本数と検証結果だけ出して終わる。
# **本番を投げる前に一度は 1 で通すこと。** spec のタイプミスで数時間を捨てずに済む。
DRY_RUN=0
NO_BETWEENNESS=0        # 1 にすると媒介中心性を計算しない (N が大きいと重い)
NO_CLUSTERING=0
NO_STRUCTURE_FIGURES=0

# ==========================================================================
# ここから下は通常さわらない
# ==========================================================================

# --- 引数チェック ---------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_lesion.sh <親runディレクトリ>" >&2
  echo "  例: bash scripts/run_lesion.sh outputs/axon_growth_grid/b20_seg100_gDefF" >&2
  echo "  切断条件はこのスクリプト内の CUT_SPECS で指定します。" >&2
  exit 1
fi

PARENT_RUN="${1%/}"   # 末尾のスラッシュを落とす (タブ補完で付くため)

# --- パス類の決定 ---------------------------------------------------------
# スクリプトの場所からリポジトリルートを特定（どこから呼んでも動くように）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d "${PARENT_RUN}" ]]; then
  echo "Error: 親 run ディレクトリが見つかりません: ${PARENT_RUN}" >&2
  exit 1
fi
if [[ ${#CUT_SPECS[@]} -eq 0 ]]; then
  echo "Error: CUT_SPECS が空です。このスクリプト内の設定ブロックで切断条件を指定してください。" >&2
  exit 1
fi

# config.yaml と重み記録は run 直下か data/ のどちらかにある (organize_output の前後で変わる)。
SOURCE_DIR=""
for SUBDIR in "" "/data"; do
  if [[ -f "${PARENT_RUN}${SUBDIR}/config.yaml" ]]; then
    SOURCE_DIR="${PARENT_RUN}${SUBDIR}"
    break
  fi
done
if [[ -z "${SOURCE_DIR}" ]]; then
  echo "Error: config.yaml が見つかりません: ${PARENT_RUN} (と ${PARENT_RUN}/data)" >&2
  exit 1
fi
if ! compgen -G "${SOURCE_DIR}/weights_*h.npz" > /dev/null; then
  echo "Error: weights_*h.npz が見つかりません: ${SOURCE_DIR}" >&2
  echo "  引き継げるのは develop.py が重みを記録した run だけです。" >&2
  exit 1
fi
# ブリッジ切断は軸索の幾何を要求する (axon_growth 系の run にしか無い)。
for SPEC in "${CUT_SPECS[@]}"; do
  if [[ "${SPEC}" == bridge:* && ! -f "${SOURCE_DIR}/axon_geometry.npz" ]]; then
    echo "Error: axon_geometry.npz が無いので bridge 切断はできません: ${SOURCE_DIR}" >&2
    echo "  connection: axon_growth 系の run を使うか、between:axis=module を使ってください。" >&2
    exit 1
  fi
done

# 出力は親 run の中に置く。その run から派生したものが 1 か所にまとまり、
# 親の config.yaml / weights_*h.npz と取り違えようがない
# (locate() は run 直下と data/ しか見ないので、lesion/ は親の解析に干渉しない)。
OUT_DIR="${PARENT_RUN}/lesion"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ログ名に切断条件を残す (同じ親 run に複数条件をかけるため)
CUT_TAG="$(printf '%s_' "${CUT_SPECS[@]}" | sed 's/_$//' | tr -c 'A-Za-z0-9._-' '_')"
LOG_PATH="${LOG_DIR}/lesion_$(date +%Y%m%d-%H%M%S)_${CUT_TAG}.log"

# --- 引数の組み立て -------------------------------------------------------
ARGS=(
  --out-dir "${OUT_DIR}"
  --cut-combine "${CUT_COMBINE}"
  --settle-ms "${SETTLE_MS}"
  --recovery-hours "${RECOVERY_HOURS}"
  --probe-interval-hours "${PROBE_INTERVAL_HOURS}"
  --probe-window-ms "${PROBE_WINDOW_MS}"
  --record-buffer-ms "${RECORD_BUFFER_MS}"
  --hub-z "${HUB_Z}"
)
for SPEC in "${CUT_SPECS[@]}"; do
  ARGS+=(--cut "${SPEC}")
done
[[ -n "${FROM_HOUR}" ]]          && ARGS+=(--from-hour "${FROM_HOUR}")
[[ "${DRY_RUN}" == "1" ]]        && ARGS+=(--dry-run)
[[ "${NO_BETWEENNESS}" == "1" ]] && ARGS+=(--no-betweenness)
[[ "${NO_CLUSTERING}" == "1" ]]  && ARGS+=(--no-clustering)
[[ "${NO_STRUCTURE_FIGURES}" == "1" ]] && ARGS+=(--no-structure-figures)

# --- 実行 -----------------------------------------------------------------
echo "parent run : ${PARENT_RUN}"
echo "cut        : ${CUT_SPECS[*]}  (combine: ${CUT_COMBINE})"
echo "out-dir    : ${OUT_DIR}"
echo "log        : ${LOG_PATH}"
[[ "${DRY_RUN}" == "1" ]] && echo "mode       : DRY RUN (GeNN は実行しません)"
echo "実行を開始します..."

# 失敗はログの中にしか出ないので、末尾を拾って端末にも見せる
# (数時間かかる run が黙って落ちると、原因に辿り着くまでが遠い)。
if ! python scripts/lesion.py "${PARENT_RUN}" "${ARGS[@]}" > "${LOG_PATH}" 2>&1; then
  echo "Error: 実行に失敗しました。ログ: ${LOG_PATH}" >&2
  echo "--- ログ末尾 ---" >&2
  tail -20 "${LOG_PATH}" >&2
  exit 1
fi

echo "完了しました。ログ: ${LOG_PATH}"
# 直近の結果ディレクトリを教える (lesion.py が日時付きサブディレクトリを作るため)
LATEST="$(ls -dt "${OUT_DIR}"/2*/ 2>/dev/null | head -1 || true)"
[[ -n "${LATEST}" ]] && echo "結果: ${LATEST}"
