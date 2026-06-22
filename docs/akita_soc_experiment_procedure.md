# Akita SoC実験シミュレーション実行手順

このドキュメントは、`scripts/akita_soc_fig2.py` を使って Akita先生の実験相当のSNNシミュレーションを実行するための手順書です。

## 前提

対象スクリプトと設定ファイルは次の通りです。

```text
scripts/akita_soc_fig2.py
configs/akita_soc_fig2.yaml
configs/components/tasks.yaml
```

現在の `akita_soc_fig2` 設定では、シミュレーション内の時間として72時間相当を扱います。

```yaml
akita_soc_fig2:
  duration: 259200000.0  # 72 h [ms]
  record_hours: [0, 6, 72]
  record_window_ms: 600000.0  # 10 min
  record_buffer_ms: 10000.0
```

ここでの72時間は、実際にPC上で72時間かかるという意味ではありません。神経回路が72時間ぶん活動した状態を、計算上で進めるという意味です。実際の実行時間は、CPU、GPU、メモリ、PyGeNN/GeNNの環境、ネットワーク規模、記録量に依存します。

現在の設定では `dt = 0.1 ms` なので、72時間相当を進める場合のステップ数は次の通りです。

```text
72 h = 259,200,000 ms
259,200,000 ms / 0.1 ms = 2,592,000,000 steps
```

かなり長い実行になる可能性があるため、本番実行前に短時間ベンチマークを行い、`tmux` などでSSH切断に耐える形で実行してください。

## 事前確認

リポジトリのルートへ移動します。

```bash
cd /home/yamada/SNN_sim
```

仮想環境を有効化します。

```bash
source .venv/bin/activate
```

設定ファイルの主要条件を確認します。

```bash
sed -n '1,80p' configs/akita_soc_fig2.yaml
sed -n '7,12p' configs/components/tasks.yaml
```

特に確認する項目は次の通りです。

- `simulation.dt`: 時間刻み。現在は `0.1 ms`
- `neurons`: ニューロン数。現在は興奮性80個、抑制性20個
- `record_hours`: 記録するシミュレーション時刻。現在は `0, 6, 72`
- `record_window_ms`: 各記録点でスパイクを記録する窓。現在は10分
- `record_buffer_ms`: 記録バッファ長。現在は10秒

## 短時間ベンチマーク

いきなり72時間相当を回さず、まず短い条件で実行時間を測ります。

```bash
mkdir -p outputs/akita_soc_benchmark
time python scripts/akita_soc_fig2.py \
  --record-hours 0 \
  --record-window-ms 60000 \
  --out-dir outputs/akita_soc_benchmark/record_0h_1min
```

この例では、0時間時点から1分間だけ記録します。GeNNの初回ビルド時間も含まれるため、2回目以降の実行時間とは差が出る場合があります。

6分相当や1時間相当の進行時間を測りたい場合は、`record-hours` を使って到達時刻を増やします。

```bash
mkdir -p outputs/akita_soc_benchmark
time python scripts/akita_soc_fig2.py \
  --record-hours 0 0.1 \
  --record-window-ms 60000 \
  --out-dir outputs/akita_soc_benchmark/record_0h_0p1h
```

`0.1 h` はシミュレーション内時間で6分です。この結果から、72時間相当の実行時間を概算します。

```text
72時間相当の概算実行時間 = 0.1時間相当の実行時間 × 720
```

厳密には、記録窓やファイル保存、初回ビルドの影響があるため完全な線形にはなりません。本番前の目安として使ってください。

## tmuxでバックグラウンド実行する

SSH接続が切れると通常の端末上で動いているプロセスは停止する可能性があります。長時間実行では `tmux` を使ってください。

新しい `tmux` セッションを作成します。

```bash
tmux new -s akita72
```

セッション内で仮想環境を有効化します。

```bash
cd /home/yamada/SNN_sim
source .venv/bin/activate
```

セッションから一時的に抜けるには、次のキー操作を行います。

```text
Ctrl-b を押してから d
```

SSHに再接続した後、実行中のセッションへ戻るには次を実行します。

```bash
tmux attach -t akita72
```

実行中のセッション一覧を確認するには次を使います。

```bash
tmux ls
```

## 本番実行

本番実行では、標準出力と標準エラーをログファイルに残します。後から条件を確認しやすいように、出力先ディレクトリも明示します。

```bash
mkdir -p outputs/akita_soc_72h/logs
LOG_PATH="outputs/akita_soc_72h/logs/run_seed1_$(date +%Y%m%d-%H%M%S).log"
python scripts/akita_soc_fig2.py \
  --record-hours 0 6 72 \
  --record-window-ms 600000 \
  --record-buffer-ms 10000 \
  --out-dir outputs/akita_soc_72h \
  > "$LOG_PATH" 2>&1
```

別のseedで実行する場合は、`--seed` と出力先を変えます。

```bash
mkdir -p outputs/akita_soc_72h/logs
LOG_PATH="outputs/akita_soc_72h/logs/run_seed2_$(date +%Y%m%d-%H%M%S).log"
python scripts/akita_soc_fig2.py \
  --record-hours 0 6 72 \
  --record-window-ms 600000 \
  --record-buffer-ms 10000 \
  --seed 2 \
  --out-dir outputs/akita_soc_72h \
  > "$LOG_PATH" 2>&1
```

`record-hours-range`: `START STOP [STEP=1]` で等間隔リストを生成できる。例: `--record-hours-range 0 72 1` で1時間刻み73点
```bash
python scripts/akita_soc_fig2.py \
  --config configs/akita_soc_autapse.yaml \
  --record-hours-range 0 72 \
  --record-window-ms 600000 \
  --record-buffer-ms 10000 \
  --out-dir outputs/autapus \
  > "$LOG_PATH" 2>&1
```

tmuxセッションの立ち上げ、configファイルの指定、以下でディレクトリの作成、log_pathの指定、python実行まですべてやってくれる
```bash
tmux new -s delay   -d 'bash scripts/run_akita_soc.sh akita_soc_delay'
```

`--out-dir` は実験グループのベースディレクトリとして扱われます。実際の成果物は、その直下に作られる `YYYYmmdd-HHMMSS/` ディレクトリへ保存されます。

`scripts/akita_soc_fig2.py` には `--duration-hours` もありますが、現在の処理では実際にどこまで進めるかは主に `--record-hours` の最大値で決まります。72時間相当を回す場合は、`--record-hours 0 6 72` を明示してください。

## 実行状況の確認

`tmux` に戻って標準出力を見る場合は次を実行します。

```bash
tmux attach -t akita72
```

ログファイルだけ確認する場合は、別のSSH端末で次を実行します。

```bash
tail -f "$(ls -t outputs/akita_soc_72h/logs/run_seed1_*.log | head -n 1)"
```

GPU使用状況を確認する場合は次を使います。

```bash
nvidia-smi
```

CPUやメモリを確認する場合は次を使います。

```bash
top
```

## 実行を止める方法

`tmux` の実行画面に戻ります。

```bash
tmux attach -t akita72
```

実行中のPythonプロセスを止めるには、端末上で次を入力します。

```text
Ctrl-c
```

止めた後、`tmux` セッション自体を終了するには次を実行します。

```bash
exit
```

## 出力ファイルの確認

本番実行が完了すると、指定した出力先に結果が保存されます。

```bash
ls -lh outputs/akita_soc_72h/run_seed1
```

主な出力は次の通りです。

- `config.yaml`: 実行時に解決された設定
- `source_config.yaml`: 元の設定ファイル
- `weights_0h.npz`: 0時間時点の重み
- `weights_6h.npz`: 6時間時点の重み
- `weights_72h.npz`: 72時間時点の重み
- `spikes_0h.npz`: 0時間時点からの記録窓内スパイク
- `spikes_6h.npz`: 6時間時点からの記録窓内スパイク
- `spikes_72h.npz`: 72時間時点からの記録窓内スパイク
- `raster_*.png`: ラスタープロット
- `avalanche_*.png`: avalanche分布プロット
- `metrics.csv`: 発火率や臨界性指標などの集計値

`metrics.csv` の先頭を確認します。

```bash
head outputs/akita_soc_72h/run_seed1/metrics.csv
```

## rcloneでオンラインストレージへアップロードする

この環境では、`outputs/` の結果を `rclone` でオンラインストレージへアップロードできます。設定済みのリモート名は次で確認します。

```bash
rclone listremotes
```

この環境では `wasedabox:` が設定されています。以降の例では、ローカルの `outputs/` 全体を `wasedabox:研究室関連/各研究グループ関連/CELL-G/00_進行中プロジェクト/SNN_sim/yamada/outputs` にアップロードします。別の保存先にしたい場合は、パス部分を変更してください。

### 実験終了後に一括アップロードする

最も安全な方法は、シミュレーションが正常終了した後に `outputs/` 全体をまとめてアップロードする方法です。`rclone copy` は転送先に同じファイルがある場合、通常は未転送または更新されたファイルだけをコピーします。

```bash
rclone copy outputs \
  wasedabox:研究室関連/各研究グループ関連/CELL-G/00_進行中プロジェクト/SNN_sim/yamada/ \
  --progress
```

`outputs/akita_soc_72h/logs` も `outputs/` 配下にあるため、この1コマンドでログも一緒にアップロードされます。`copy` は、ローカルにあるファイルをリモートへコピーします。通常はリモート側の余分なファイルを削除しないため、実験結果の退避用途では `sync` より安全です。

アップロード結果は次で確認します。

```bash
rclone lsf wasedabox:研究室関連/各研究グループ関連/CELL-G/00_進行中プロジェクト/SNN_sim/yamada/
```

### 実験終了後に自動でアップロードする

実験が終わったら続けてアップロードしたい場合は、同じ `tmux` セッション内で次のように実行します。

```bash
mkdir -p outputs/akita_soc_72h/logs
LOG_PATH="outputs/akita_soc_72h/logs/run_seed1_$(date +%Y%m%d-%H%M%S).log"
python scripts/akita_soc_fig2.py \
  --record-hours 0 6 72 \
  --record-window-ms 600000 \
  --record-buffer-ms 10000 \
  --out-dir outputs/akita_soc_72h \
  > "$LOG_PATH" 2>&1

rclone copy outputs \
  wasedabox:研究室関連/各研究グループ関連/CELL-G/00_進行中プロジェクト/SNN_sim/yamada/ \
  --progress \
  > "outputs/akita_soc_72h/logs/rclone_run_seed1_$(date +%Y%m%d-%H%M%S).log" 2>&1
```

この書き方では、Python実行が終了した後に `rclone copy` が実行されます。Python実行がエラーで終了した場合でも次のコマンドへ進むため、ログを含めてアップロードしたい用途に向いています。

エラー終了時にはアップロードしたくない場合は、実験コマンドと `rclone copy` を `&&` でつなぎます。

```bash
mkdir -p outputs/akita_soc_72h/logs
LOG_PATH="outputs/akita_soc_72h/logs/run_seed1_$(date +%Y%m%d-%H%M%S).log"
python scripts/akita_soc_fig2.py \
  --record-hours 0 6 72 \
  --record-window-ms 600000 \
  --record-buffer-ms 10000 \
  --out-dir outputs/akita_soc_72h \
  > "$LOG_PATH" 2>&1 && \
rclone copy outputs \
  wasedabox:研究室関連/各研究グループ関連/CELL-G/00_進行中プロジェクト/SNN_sim/yamada/ \
  --progress
```

### 実験中に定期アップロードする

実験中の途中成果物も退避したい場合は、シミュレーション用とは別の `tmux` セッションで定期的に `rclone copy` を実行します。

```bash
tmux new -s rclone-sync
```

セッション内で次を実行します。

```bash
cd /home/yamada/SNN_sim
while true
do
  date
  rclone copy outputs \
    wasedabox:研究室関連/各研究グループ関連/CELL-G/00_進行中プロジェクト/SNN_sim/yamada/ \
    --progress
  sleep 1800
done
```

この例では30分ごとに `outputs/` 全体を差分アップロードします。`Ctrl-b` のあと `d` で `tmux` から抜けても同期ループは継続します。

定期アップロードを止める場合は、同期用セッションに戻って `Ctrl-c` を押します。

```bash
tmux attach -t rclone-sync
```

### copyとsyncの注意点

実験結果の退避では、基本的に `rclone copy` を使ってください。

- `rclone copy`: ローカルからリモートへコピーする。リモート側の余分なファイルは通常削除しない。
- `rclone sync`: ローカルとリモートを同期する。条件によってはリモート側のファイル削除が発生する。

`sync` は便利ですが、指定先を間違えるとオンラインストレージ側の既存ファイルを消す可能性があります。使う場合は、先に `--dry-run` で何が起きるか確認してください。

```bash
rclone sync outputs \
  wasedabox:研究室関連/各研究グループ関連/CELL-G/00_進行中プロジェクト/SNN_sim/yamada/ \
  --dry-run
```

## 注意点

72時間時点の記録では、72時間に到達した後、さらに `record_window_ms` の分だけスパイク記録を行います。現在の設定では `record_window_ms = 600000 ms` なので、72時間時点から追加で10分間の活動を記録します。

`tmux` はSSH切断には有効ですが、次の問題は防げません。

- マシンの再起動
- GPUドライバの停止
- メモリ不足
- ディスク容量不足
- PythonまたはGeNN側の例外終了

本番前に、短時間ベンチマーク、ディスク容量確認、ログ保存先確認を必ず行ってください。

ディスク容量は次で確認できます。

```bash
df -h .
```

## 推奨する実行順

1. `source .venv/bin/activate` で環境を有効化する。
2. `record-hours 0` で最小実行を確認する。
3. `record-hours 0 0.1` で6分相当のベンチマークを取る。
4. 実行時間を72時間相当に外挿する。
5. `tmux` セッションを作成する。
6. `--record-hours 0 6 72` とログ保存付きで本番実行する。
7. `tail -f`、`nvidia-smi`、出力ディレクトリで進捗と結果を確認する。
8. 実験終了後、`rclone copy` で `outputs/` の結果をオンラインストレージへアップロードする。
