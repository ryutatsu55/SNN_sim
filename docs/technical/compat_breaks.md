# 非互換変更ログ

**過去の run と結果が比較できなくなる変更**をここに記録する。同じシード・同じバックエンドでも
実現値が変わる種類の変更が対象。後方互換フラグは置かない方針（`docs/utils_refactor_plan.md`）
なので、ここが唯一の追跡手段になる。

> **ルール:** 既存 run と結果が変わる変更をしたら、必ずこのファイルに追記すること。
> 書式は「何が変わったか / 影響の大きさ（実測） / 古い run をどうするか」。

---

## `axon_growth`: 境界処理と結合抽選の変更

`axon_growth` の実現値は以下の変更をまたいで**一切比較できない**。互換フラグは無い。

- 壁での advance-and-slide 方式への変更（旧方式は disk で長さの 28% を切り落としていた）
- `_CONTAINS_TOL = 1e-9` の導入
- 結合抽選が「セグメントごと」から「invasion ごと」へ

**凸領域も例外ではない** — 境界ハンドラと抽選の両方が変わったため。

実測の桁感: `axon_growth_modular` の同一 config が、抽選を per-invasion に移した時点で
**34,319 → 14,243 シナプス**。

詳細 → `docs/technical/axon_growth_confinement.md`

---

## 解析層が COO 専用化 → 重み統計が変わった

不在の結合がもう 0 を寄与しない。`block_values()` は `(weights, row, col, layout)` を取り、
実在するシナプスだけを見る。したがって対角ブロック（EE, II）— 以前は空の自己結合セルを
含んでいた — の平均が上がる。

実測: `test/test_akita_soc.py` の4ニューロンのフィクスチャで **ee 0.3 → 0.6**、**ii 0.4 → 0.8**。

**全結合でない config では、古い run の指標は新しい run と比較できない。**

---

## Akita run ディレクトリが COO 専用に。旧密形式のリーダは無い

各 run は `connectivity.npz`（row/col/shape）を1度書き、加えて `data` だけを持つ
`weights_{h}h.npz` を書く。

`runio.load_weight_values()` は legacy の `weights` キーを持つ npz に対して、どの 0 がシナプス
だったかを推測せずに **raise する**。

**旧密形式の run は、解析するには再実行するしかない。**

---

## NetworkLayout 導入: ID 割り当てが変わった

`group_info`（dict）が `NetworkLayout` に置き換わった際、ID 割り当てがランダム散布から
config 順の連番に変わった（その後 `layout.assignment` として明示的な選択肢になった）。
シード → ネットワーク実現が非互換。Akita 実験に影響。

`sequential` ↔ `random` の切り替え一般については `docs/technical/reproducibility.md` §5。

---

## 領域(area)の幾何を変えると重み・遅延もずれる

汎用 `Area.sample()` は棄却サンプリングなので draw 回数が形状に依存し、`NetworkBuilder.rng` の
下流（重み・遅延）が全部ずれる。`allow_soma: false` / `soma_in_bridge: false` も同様。

→ `docs/technical/reproducibility.md` §4
