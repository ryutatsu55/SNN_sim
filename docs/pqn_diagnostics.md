# PQN応答・時間刻み診断

`configs/pqn_diagnostics.yaml` はPQN基準ネットワークと単一細胞応答を、4種類の時間刻みで比較する標準条件を定義します。モデル定数、方程式、閾値は変更しません。

```bash
.venv/bin/python scripts/pqn_diagnostics.py plan --config configs/pqn_diagnostics.yaml
.venv/bin/python scripts/pqn_diagnostics.py run --config configs/pqn_diagnostics.yaml
.venv/bin/python scripts/pqn_diagnostics.py analyze --run-dir outputs/pqn_diagnostics/<実行ID>
```

`plan` は22ケースの条件をJSONで表示し、GeNNを初期化しません。`run` はCPUで逐次実行し、特定のケースだけ試す場合は `--case-id` を繰り返し指定します。`analyze` は実行ディレクトリの保存済み計画・データを読みます。

標準22ケースは、独立細胞の入力20/500 msと4つのdt（8件）、待機を4000 msにした細胞・ネットワーク各1件、ネットワークの刺激あり・無刺激・重み0と4つのdt（12件）です。独立細胞はRSexci/RSinhiそれぞれ12振幅を同時に記録します。ネットワーク比較は保存配線・入力・実効遅延を一致させ、dtまたは待機時間だけを変えます。

暫定基準はYAMLに明記し、実行計画にも保存します。細胞は区間別発火数、初回潜時、V/N/Q統計、閾値交差とイベントの対応、刺激開始からのV/N/Q波形NRMSEを比較します。波形は共通時刻の記録点を使い、補間しません。ネットワークは刺激後・末尾の発火数、参加細胞数、最後の発火潜時とactivity statusを比較します。待機延長ケースは保存済み `warmup_drift.json` も判定材料に含みます。

実行結果は `outputs/pqn_diagnostics/<実行ID>/` に保存します。`plan.json` と `manifest.json` は実行条件・比較ペア・ソース版を記録します。各 `cases/<case-id>/` にはスパイク、状態波形、実効入力、設定、ログ、状態を保存します。`cell_metrics.csv`、`network_metrics.csv`、`comparisons.json`、`decision.json`、`cell_waveforms.png`、`report.md` は保存データから作り直せます。再解析時は `analysis_manifest.json` に解析コードのハッシュを記録します。

ケースの未選択、未実行、非有限値、異常終了、Python例外は区別して保存します。失敗ケースを0発火として扱いません。比較対象の片方が欠ける場合や保存した入力・配線・設定が一致しない場合、その比較は判定不能です。ケース失敗がある実行は終了コード1、入力検証エラーは2です。数値基準の不一致は実験プログラムの失敗ではなく、比較結果に記録します。

閾値は探索用の暫定基準であり、数値比較が通っても収束の証明や臨界性の確定にはなりません。RSinhiの応答が少ない、または長い脱分極が残る場合も、この診断だけで生物学的な妥当性や実装不具合を断定できません。

クラッシュの根本原因は今回調査しません。既存runnerの開始前SIGSEGV再試行とログ保存を利用し、追加の再試行やGeNN環境変更は行いません。
