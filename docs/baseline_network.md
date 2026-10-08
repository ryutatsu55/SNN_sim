# PQN基準ネットワークの実行方法

オータプスと可塑性を入れないPQNネットワークの予備探索を行う。主モデルは `PQN_float` の `RSexci` / `RSinhi`。配線は既存の `NetworkBuilder` へ接続する。本CLIの状態記録はこの2モードを対象とし、他モデルは設定段階で拒否する。既存レジストリのモデル切替機能は維持している。

## 実行

リポジトリ直下で、PyGeNNを導入済みの環境を使う。既定はCPU。CUDAバックエンドは設定できるが、今回の確認はCPUのみ。

```bash
# 配線のみ作成
.venv/bin/python scripts/baseline_network.py prepare

# 独立細胞の入力応答・抑制性シナプス対照
.venv/bin/python scripts/baseline_network.py calibrate --warmup-ms 2000 --dt-ms .025

# 100細胞の実装確認
.venv/bin/python scripts/baseline_network.py run --n 100 --post-ms 200

# 1000細胞、配線3種×刺激対象3種×結合倍率6種＋無刺激3条件
.venv/bin/python scripts/baseline_network.py sweep

# 静止状態からの探索では、確認結果を踏まえて待機を延長する
.venv/bin/python scripts/baseline_network.py sweep --warmup-ms 2000 --dt-ms .025
```

`configs/baseline_network.yaml` の初期探索値は待機500 ms、刺激20 ms、刺激後2000 ms、dt=0.05 ms。**500 msでは回復変数に残る変化を確認したため、定常な静止状態を比較する次の探索には待機2000 ms以上を用いる。** 初期探索の再現性のため既定値は保存している。dt=0.05 msと0.025 msで数値安定性・発火数が変わる条件もあり、細刻みの再確認は必須。

単条件では `--multiplier`、`--inhibition-factor`、`--stimulus-seed`、`--no-stimulus` を指定できる。走査ではYAMLの `multipliers` と `inhibition_factors` を変更する。倍率0を含める。抑制倍率はI→EとI→Iのみに掛ける。

```bash
# 保存済みの配線を再使用する（サイズ・空間条件・遅延は設定と一致させる）
.venv/bin/python scripts/baseline_network.py run --snapshot outputs/baseline_network/実行名/wiring_11.npz --multiplier 2 --inhibition-factor .5

# 完了した単条件を再検証する。既定ではdtを半分、刺激後観測を2倍にする
.venv/bin/python scripts/baseline_network.py verify --reference-run outputs/baseline_network/実行名/case_0003

# dtと観測時間を固定し、刺激振幅だけを変更する例
.venv/bin/python scripts/baseline_network.py verify --reference-run outputs/baseline_network/実行名/case_0003 --dt-ms .05 --post-ms 2000 --stimulus-amplitude .3
```

`verify` は元の設定、保存配線、刺激シード、結合倍率を引き継ぐ。刺激割合を変えると保存順列の先頭から選ぶため、対象集合は入れ子になる。元の配線ファイルと実行記録は一緒に保管する。走査全体の途中再開機能はなく、失敗した条件は保存設定から別ディレクトリで再実行する。シミュレーション開始前のSIGSEGVのみ最大2回自動再試行し、失敗ディレクトリとログを `*_build_failure_*` に保存する。開始後のクラッシュや数値不安定は再試行しない。

## 条件と記録

- IDは興奮性0〜799、抑制性800〜999。自己結合・重複結合は禁止。
- 空間は既定 `none`。`uniform_2d` を選ぶと配置シード・単位・範囲・開放境界を保存する。座標は今回の配線・遅延には影響しない。
- 配置、配線、刺激選択、将来のE/I別オータプス対象順序の乱数系列を分離する。
- 初期刺激は興奮性40細胞への振幅0.4の矩形電流。入力・膜電位はPQNモデル単位、時刻はms。発火を強制する指定ではない。
- 送信側が興奮性なら正、抑制性なら負の静的ExpCurr。時定数はE=2 ms、I=5 ms。
- 実効到着遅延1.5 msはGeNNの更新順序を含めて設定する。内部遅延を1ステップ減らす。dtの整数倍かつ1〜256ステップを受け付ける。
- 各試行は独立プロセスで新しいモデルを生成する。状態・残留電流・遅延キューを引き継がない。
- 演算と状態保存はdouble。既存の重み初期化はfloat32丸めを含むため、丸め後の実際の値も記録する。
- 全細胞のスパイクを保存し、代表細胞のV/N/Qを記録する。校正の独立細胞は全細胞・全ステップを保存する。
- `pre`、`during`、`post` は左閉右開区間。スパイクと入力は更新開始時刻、状態は更新終了時刻。

## 出力

各実行は `outputs/baseline_network/時刻/`、生成コードは `genn_code/baseline_network/時刻/` へ保存する。

| ファイル | 内容 |
| --- | --- |
| `settings.yaml` / `manifest.json` | 設定、CLI引数、Gitコミット、完了・失敗状態 |
| `wiring_*.npz` | COO、実効遅延、IDと極性、座標、対象順序、メタデータとSHA-256 |
| `summary.csv` / `sweep.png` | 正常終了した試行の集計と図。配線内反復と配線間反復を区別して解釈する |
| `numerical_failures.json` | 数値不安定で除外した試行。0発火として扱わない |
| `case_*/metadata.json` | 配線識別子、対象数・割合、絶対重み、精度、ソースハッシュ |
| `case_*/resolved.yaml` / `neuron_constants.json` | 構築用設定、細胞定数と初期状態 |
| `case_*/stimulus_*.npy` | 刺激IDと全細胞の入力振幅 |
| `case_*/spikes.npz` | 全発火時刻 `times_ms` と細胞 `ids` |
| `case_*/traces.npz` | 状態時刻、代表ID、V/N/Q、サンプル時のE/I別最大入力絶対値 |
| `case_*/rates.csv` / `metrics.json` / `activity.png` | E/I発火率、参加細胞、刺激後・末尾活動、ラスター |
| `case_*/warmup_drift.json` | 待機終了前50 msのV/N/Q変化の全細胞最大値 |
| `case_*/failed_state.npz` | 非有限値を検出した時刻・状態変数・全細胞値（検出時のみ） |
| `case_*/status.json` / `case_*.log` / `build_retries.json` | 単条件の実行状態、ログ、開始前再試行履歴 |
| `calibration.csv` / `calibration_waveforms.png` | 独立細胞の発火応答と500 ms入力の波形 |
| `inhibition_comparison.csv` | 同じ外部入力にRSinhiからの負の指数電流を加えた対照 |

非有限値は `sample_ms` ごとに検査する。数値失敗時に完全なスパイク履歴を取得することは保証しない。設定・失敗状態・ログは残す。通常のネットワーク波形はサンプル記録であり、個々のスパイク形状の確認には細かいサンプル間隔を指定する。

末尾100 msの発火は「観測末尾まで活動がある」という指標であり、無限時間の持続や臨界性の判定ではない。ビン内参加割合、集団カウントCV、スペクトルピークは記述指標とし、過渡応答のスペクトルから周期振動を断定しない。数値不安定も活動持続には数えない。

## テスト

```bash
.venv/bin/python -m pytest test/utils/test_baseline_network.py test/utils/test_autapse_protocol.py test/utils/test_autapse_analysis.py -q
```

保存・改変検出、座標変更時の配線不変性、対象集合、重み極性、無入力、刺激停止、独立試行の一致、dt=0.05/0.025 msでの実効遅延と指数減衰を確認する。

## 実行環境の既知の制約

この環境のGeNNでは、シミュレーションを行わずコード生成だけを繰り返す最小再現でもSIGSEGVが起きた。ネイティブスタックではシナプス更新の `EnvironmentGroupMergedField::addField` 周辺の型コピー処理を確認したが、C++側の根本原因の修正までは行っていない。試行分離、開始前限定の再試行、失敗記録で対処する。再試行が上限に達した場合は走査を失敗として終了し、正常結果に混ぜない。
