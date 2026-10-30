# LIF可塑性ネットワークの基準探索

オータプスを導入する前の比較基準を用意する実験です。論文の厳密再現を目的にせず、既存の確率LIF・STDP・STPを使います。

## ネットワークと探索

既定は1000細胞（E800/I200）、方向付きランダム結合p=0.1、自己結合なし、初期重み0、空間構造なしです。細胞数を減らすことなく短い検証から始められます。

escape-noise LIFを用い、静止付近でも低頻度で発火します。膜時定数30ms、静止電位−74mV、発火確率の参照電位−54mV、gain 4mV、不応期E3/I2ms。硬い閾値を超えるだけで必ず発火する通常のLIFとは異なります。

シナプスはコンダクタンス型でE/Iの反転電位0/−80mV、減衰2/4ms。コンダクタンスの値は漏れコンダクタンスに対する無次元比です。`g_max` をµSやnSとして解釈しないでください。遅延設定はEE1.5ms、その他0.8ms。既存の軸索遅延実装を使います。

E/I-STDPは既存の到着時刻基準nearest方式を明示して使用します。Eは時定数20ms、Iは10/20ms、学習振幅0.01、βE=1.05、重み範囲0〜1。STPは利用率0.4、回復150ms。重みを固定しても、STPによる実効伝達効率の変化は継続します。

既定の27条件はg_max=1.5/2.5/4.0、βI=1.05/1.15/1.25、seed=11/22/33。静止付近の発火率は0.4Hzで、`rest_rates_hz` に複数値を入れると走査軸に追加できます。同じseedの条件は同じ配線と発火乱数の初期シードを使います。発火列自体は重み・条件の違いにより変化します。

発達の後、再ビルドや状態初期化をせずにSTDPだけを停止して活動を観測します。固定後の重みは毎チャンク完全一致を検査します。既定の発達1時間と固定後60秒は探索開始条件であり、成熟・臨界状態を保証しません。臨界性は単一指標で自動判定しません。

## 計画の確認

リポジトリ直下から、既存のPyGeNN導入済みvenvを使用します。

```bash
.venv/bin/python scripts/lif_network.py plan --preset smoke
.venv/bin/python scripts/lif_network.py plan --preset pilot
.venv/bin/python scripts/lif_network.py plan --preset full
```

`plan` はGeNNのモデルを構築しません。`--config` を省略すると `configs/lif_network.yaml` を使います。未知フィールド、非有限値、重複シード、遅延や記録間隔の時間不整合を拒否します。

| プリセット | 細胞数 | 条件数 | 発達 | STDP停止後 |
| --- | ---: | ---: | ---: | ---: |
| smoke | 設定値、既定1000 | 1 | 1秒 | 1秒 |
| pilot | 設定値、既定1000 | 1 | 60秒 | 10秒 |
| full | 設定値、既定1000 | 設定の直積、既定27 | 設定値、既定1時間 | 設定値、既定60秒 |

smoke/pilotは指定リストの中央のg_maxとβI、先頭seed・発火率を選びます。小さい設定を検証に使う場合も、主実験のYAMLを上書きする必要はありません。

## バックグラウンド起動

以下のremoteは例です。実際のrclone設定済みremoteと保存先に置き換えます。転送不要なら `--remote` を省略します。

```bash
.venv/bin/python scripts/lif_network.py start --preset pilot \
  --remote 'REMOTE:SNN_sim/lif_network'
```

起動するとJSONで絶対パスの `run_dir` とPIDを表示して、ランチャは終了します。管理プロセスは独立したセッションで動き、入力を端末につながず、出力を `run.log` に保存します。VSCodeや統合端末が終了しても計算・解析・転送を継続する構成です。Windows/WSLの終了・再起動・電源断では計算は停止し、スリープ中は進みません。

保存先を事前に決めたい場合は、まだ存在しないディレクトリを `--run-dir outputs/lif_network/任意の実行名` で指定します。既存ディレクトリの上書きは拒否します。CPU/CUDAはYAMLの `backend` で指定できますが、今回の検証はCPUです。

pilotを確認してから本実験を起動します。

```bash
.venv/bin/python scripts/lif_network.py start --preset full \
  --remote 'REMOTE:SNN_sim/lif_network'
```

発達時間を延長するときはYAMLを複製し、`development_ms` を変更して `--config` で指定します。発達・固定後観測・重み保存間隔は `chunk_ms` の整数倍が必要です。1条件の実時間上限は `case_timeout_s` で、既定86400秒です。集団平均発火率がいずれかのE/Iで `max_rate_hz` を超えた条件は停止・失敗として保存します。この上限は過剰活動の運用上の判定であり、臨界性の判定ではありません。

## 進捗・停止・再開

以下のパスを、起動時に表示された実際の `run_dir` に置き換えます。

```bash
.venv/bin/python scripts/lif_network.py status --run-dir outputs/lif_network/実行名
tail -f outputs/lif_network/実行名/run.log
```

`status` は計画の状態、管理プロセスの生存、現在条件、シミュレーション時間、経過時間、推定残時間、転送状態を表示します。コンパイル中や最初の記録前には残時間の推定値はありません。条件の詳細ログは `cases/<ケースID>/attempt_*.log` です。推定残時間は既に実行した条件からの目安で、活動量やビルド時間により変わります。

```bash
# 現在の条件を完了してから停止する。長い条件の場合は終了まで待つ。
.venv/bin/python scripts/lif_network.py stop --run-dir outputs/lif_network/実行名

# 実行中の条件を中断して停止する。
.venv/bin/python scripts/lif_network.py stop --force --run-dir outputs/lif_network/実行名

# 完了した条件を検証・スキップし、残りを再実行する。
.venv/bin/python scripts/lif_network.py resume --run-dir outputs/lif_network/実行名
```

再開は**条件単位**です。発達途中のGeNN内部状態から再開しません。中断・失敗条件は先頭から新しいattemptとして実行し、元のデータ・ログ・終了コードは残します。重みだけから同じ発達を継続できるとは扱いません。再開時は設定・計画・関連ソースのハッシュと成功条件の全成果物を検証します。改変・欠損があれば停止するため、コード更新後は新しい実行として起動してください。

PIDに加えてLinuxのプロセス開始情報を確認し、ロックで二重起動を防ぎます。停止は管理プロセスへの要求ファイルで行います。強制停止・timeoutはケースのプロセスグループに適用し、コンパイラ等の子孫も回収します。

## 出力

| 場所 | 内容 |
| --- | --- |
| `report.md` / `summary.csv` | 日本語報告、条件一覧、成功・失敗・中断・未実行数、固定後の指標 |
| `overview_*.png` | 発火率・ΔCr・バースト指標のパラメータマップ。成功seedの平均、欠測は空欄 |
| `settings.json` / `plan.json` | 実際の設定とケース一覧 |
| `manifest.json` / `source/` | Git情報、依存バージョン、実行に使った関連ソースとハッシュ |
| `artifact_manifest.json` | 管理ログ等を除いた確定成果物のサイズ・SHA-256 |
| `cases/<ID>/outcome.json` | 最新attempt、終了状態・終了コード・所要時間 |
| `cases/<ID>/attempt_*.exit.json` / `.log` | 各試行の終了記録と標準出力・標準エラー |
| `cases/<ID>/attempt_*/resolved.yaml` | 細胞・シナプス・STDP/STPの全定数を含む構築設定 |
| 同 `connectivity.npz` | 行優先COO、遅延、E/Iの細胞数。IDはEが先頭、次にI |
| 同 `chunks/000000.npz` 等 | 全細胞の発火時刻・ID。時刻はチャンク相対ms、絶対開始時刻と期間も保存 |
| 同 `chunks/state_*.npz` | チャンク終了時の代表細胞の膜電位・不応期・シナプス入力項 |
| 同 `weights/` | 初期・定期・STDP停止時のCOO順のdouble重み |
| 同 `freeze_state.npz` / `final_state.npz` | 境界の全細胞状態とSTP資源等。完全な再開用チェックポイントではない |
| 同 `metrics.csv` / `frozen_metrics.json` | 時間発展と固定後の全観測を用いた記述指標 |
| 同 `activity.png` / `diagnostics.png` | 発火率・重みの時間発展、固定後ラスター、重み分布、アバランチ分布、スペクトル |
| 同 `artifacts.json` | 完了した条件の全成果物のチェックサム。再開で検証する |
| `upload/` | 転送状態、rcloneのログ・終了コード・試行履歴 |

NPZはNumPyの圧縮形式です。`allow_pickle=True` は不要です。全細胞の毎ステップ電位は標準保存せず、全細胞の発火は全期間保存します。細かい膜電位波形を調べたい場合は別の短い検証を行います。シナプス入力項は膜方程式内の値であり、物理単位のnA電流ではありません。

固定後のスパイクをチャンク間で連結してから解析し、保存単位の境界をアバランチの境界にしません。既存の平均スパイク間隔による切り出しと、サイズ1〜Nを台とする指標を使用します。べき乗を支持するLLRだけで臨界性を確定せず、未学習の低活動、同期バースト、少数細胞の反復などをラスターと合わせて評価してください。

## 終了後のrclone転送

計画内の全条件と解析が終了した後、バックグラウンド処理が `rclone copy` を実行します。一部条件が失敗しても得られた結果と失敗記録を転送します。一時停止では転送を保留します。保存先は `<指定remote>/<実行ディレクトリ名>/` です。

本体を転送し、次に転送結果と本体ログを追加転送し、最終状態の受領記録も送ります。rcloneの終了コードで結果を判定します。rcloneがバックエンドに応じて行う照合とは別の、全ファイルのリモート再読込み検証は行いません。成果物チェックサム一覧は一緒に転送します。最後の転送自身のログはローカルに保持します。

計算の成否と転送の成否は別です。`status.json` は計算状態、`upload/status.json` は転送状態を持ちます。転送失敗時もローカル結果を保持し、保存先を指定して転送だけ再実行できます。

```bash
.venv/bin/python scripts/lif_network.py upload --run-dir outputs/lif_network/実行名 \
  --remote 'REMOTE:SNN_sim/lif_network'
```

`resume --remote ...` でも転送先を更新できます。本体・最終転送はそれぞれ既定7200秒を上限とし、rcloneには最大3回の再試行を指定します。転送時間上限は実行ディレクトリの `job_options.json` の `upload_timeout_s` で、停止中に変更できます。認証設定はコピーせず、生成コード・バイナリは `genn_code/lif_network/` にローカル保存します。転送先やローカル結果の削除は行いません。

別実行をまとめて手動転送する場合は、書込みが終わっていることを確認した上で次を使えます。

```bash
rclone copy outputs/lif_network/ 'REMOTE:SNN_sim/lif_network/' --retries 3 --progress
```

## 再解析と検証

```bash
.venv/bin/python scripts/lif_network.py analyze --run-dir outputs/lif_network/実行名
.venv/bin/python -m pytest test/utils/test_lif_network.py test/test_akita_model.py -q
```

再解析は終了後に実行し、図・指標・成果物一覧を更新します。追加のシミュレーションは行いません。更新後のファイルは必要なら `upload` で再転送してください。

## 2026-10-22の実装検証

新規22テストと既存42テストが通過しました。1000細胞でdt=0.1/0.05msをそれぞれ2回実行し、各dt内で発火列の再現、学習中の重み変化、STDP停止後の重み完全一致、STPの継続、自己結合なしを確認しています。異なるdt間で発火列の一致や長時間の数値収束を確認したという意味ではありません。

60秒発達＋10秒固定観測のpilotも完走しました。100,209結合、固定後4,092スパイク、平均0.4092Hz、980細胞が参加し、固定後の重みは完全一致しました。まだ低活動の初期段階です。臨界状態の獲得や成熟は未検証です。

pilotの計算部分は約14.3秒、解析・図生成を含む管理処理は約16.2秒でした。条件別フォルダは約39.4MBで、pilotは重みを1秒ごとに保存しています。本実験は既定で60秒ごとの保存です。活動が発達すると速度・保存量が変わるため、これを長時間実験の保証値にはしません。

結果は `outputs/lif_network/validation_pilot_20261022/`、詳しい検証記録は `outputs/lif_network/validation_20261022.md` です。実際の外部アップロードと長時間27条件の本実験はまだ実行していません。
