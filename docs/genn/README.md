# GeNN 利用ガイド（SNN_sim 向け詳細ドキュメント）

このディレクトリは、SNN シミュレータ **GeNN**（および Python フロントエンド **pygenn**）の
できること・使い方を、SNN_sim プロジェクトのメンバー向けに日本語で詳細にまとめたものです。

- 対象 GeNN バージョン: **5.4.0**（`/home/tanii/genn/version.txt`）
- Python パッケージ名: `pygenn`
- 本文中のコード例・API 名・組み込みモデルの仕様は、すべて `/home/tanii/genn` の実ソース
  （`include/genn/genn/*.h`, `pygenn/*.py`, `docs/*.rst`, `tests/features/*.py`, `userproject/*.py`）
  を根拠に記載しています。

> GeNN は「ネットワークを Python で記述 → C++/CUDA コードを**生成**してコンパイル → GPU/CPU で実行」
> という **コード生成型** のシミュレータです。この性質が他の SNN シミュレータとの最大の違いであり、
> ニューロン・シナプスの挙動を **GeNNCode**（C 風の文字列）で自由に定義できます。

---

## 目次

| 章 | ファイル | 内容 |
|----|----------|------|
| 1 | [01_overview.md](01_overview.md) | GeNN とは／SNN 向け／コード生成方式／バックエンド／バッチング・非同期実行 |
| 2 | [02_installation.md](02_installation.md) | 前提（コンパイラ・CUDA・libffi）／pip インストール／SNN_sim の依存関係 |
| 3 | [03_quickstart.md](03_quickstart.md) | 最小の動く例と基本ワークフロー（create→build→load→step_time→pull） |
| 4 | [04_model_building.md](04_model_building.md) | `GeNNModel` とポピュレーション追加 API の詳細 |
| 5 | [05_builtin_models.md](05_builtin_models.md) | 組み込みニューロン／重み更新／後シナプス／電流源／初期化スニペット一覧 |
| 6 | [06_custom_models.md](06_custom_models.md) | GeNNCode 言語仕様と `create_*` でのカスタムモデル定義 |
| 7 | [07_simulation_recording.md](07_simulation_recording.md) | 実行ループ・変数アクセス・スパイク記録・バッチ・動的パラメータ・計測 |
| 8 | [08_advanced.md](08_advanced.md) | 遅延／カスタム更新／カスタム結合更新／変数参照／EGP／（pre_spike_code 遅延の話） |
| 9 | [09_api_reference.md](09_api_reference.md) | API 早見表（メソッド・`create_*`・`init_*`・主要列挙型） |
| 10 | [10_snn_sim_integration.md](10_snn_sim_integration.md) | SNN_sim 内での GeNN 利用（NetworkBuilder/Simulator/Registry/Config） |

---

## 読み方ガイド

- **まず動かしたい人**: [03_quickstart.md](03_quickstart.md) →[04_model_building.md](04_model_building.md)。
- **モデルを自作したい人**: [06_custom_models.md](06_custom_models.md)（GeNNCode）と
  [05_builtin_models.md](05_builtin_models.md)（既存モデルの実装を参考にする）。
- **学習則・遅延・カスタム更新が必要な人**: [08_advanced.md](08_advanced.md)。
- **SNN_sim のコードを読む／拡張する人**: [10_snn_sim_integration.md](10_snn_sim_integration.md)。

## GeNN の処理フロー（全体像）

```
Python (pygenn)                         生成された C++/CUDA               実行
─────────────────                       ──────────────────               ──────
GeNNModel("float", "net")
  .add_neuron_population(...)   ─┐
  .add_synapse_population(...)   │ build() で
  .add_current_source(...)       ├──▶ コード生成 + コンパイル ──▶ load() ──▶ step_time() ×N
  .add_custom_update(...)       ─┘                                            │
                                                                  push/pull で変数を CPU⇄GPU 転送
                                                                  spike_recording_data で発火を取得
```

各章末尾にナビゲーションリンクがあります。次は [01_overview.md](01_overview.md) へ。
