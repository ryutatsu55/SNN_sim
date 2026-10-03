### 検討事項: GeNN の `matrix_type="DENSE"` 対応（未着手）

ホスト側の結合生成は COO だけで、GeNN への登録 (`add_synapse_population(matrix_type=...)`) も
[NetworkBuilder.py:593](../src/core/NetworkBuilder.py#L593) で `"SPARSE"` 固定。

```
_generate_global_matrices()   connector → (rows, cols)、weight / delay → 1 結合 1 値
        ↓
global_coo()  (GlobalCOO, row-major ソート済み)
        ↓ _pair_coo(src, tgt) が集団ごとに切り出す
sg.set_sparse_connections(local_src_idx, local_tgt_idx)   ← 常に matrix_type="SPARSE"
```

#### DENSE を入れる場合の作業量（見積り）

`matrix_type="DENSE"` を選べるようにするのは**軽くない**。GeNN の DENSE は
「N_pre × N_post を全部持つ」形式で、疎とはインターフェースが根本的に違う。

| 影響箇所 | 内容 | 重さ |
|---|---|---|
| `set_sparse_connections()` | DENSE では**呼べない**（結合は暗黙に全対全）。疎で表現していた「結合なし」を weight=0 で表す設計に変える必要がある | 中 |
| `SynapseIndex` | 疎は「実結合の並び」、DENSE は「全ペアの行優先」。`local_src/local_tgt/global_src/global_tgt` の意味と長さが変わる | 中 |
| `pull_synapse_flat` / `pull_synapse_coo` | `values.size == index.num_synapses` の検証が前提から崩れる。DENSE では `n_pre * n_post` になる | 中 |
| `max_dendritic_delay_timesteps` / `num_threads_per_spike` | 到着 kernel の並列度は**行長（= 疎の fan-out）**から決めている。DENSE では行長 = N_post 固定になり、`_arrival_threads_per_spike` の前提が崩れる | **大** |
| per-synapse 遅延 | fork の `pre_arrival_syn_code` が DENSE で同じ意味を持つか未検証 | **大（要調査）** |

**得られる利益は小さい。** DENSE が有利なのは「ほぼ全結合 かつ シナプス変数が少ない」場合に
限られるが、本プロジェクトは STDP で per-synapse 変数（重み・トレース・遅延）を持つため、
結合が全対全でも DENSE の利点はほとんど出ない。逆に結合が疎なら VRAM を大量に無駄にする。

**結論: 当面 `"SPARSE"` 固定でよい。** 将来 DENSE を入れるなら、上表の「大」2 件
（到着 kernel の並列度・per-synapse 遅延）の調査が前提になる。

---

## 導入予定: ハブのグラフ理論的定義による評価（未着手）

**方針: 損傷部位の「ハブ／周辺」を、幾何学的定義だけでなくグラフ理論的定義でも評価する。**

### 動機

Watanabe et al., *Commun. Eng.* **5**:150 (2026)
（`docs/private/watanabe.pdf`, doi:10.1038/s44172-026-00686-5）は、
`hierarchical_modular_grid` の実物にあたる 16 モジュールのマイクロ流体デバイスで
ハブ結合と周辺結合をレーザー切断し、**急性期の被害は同等（平均相関 −35.7% vs −33.3%）
なのに回復のタイムスケールが 3 倍違う（3 日 vs 1 日）**ことを示している。

この論文の「ハブ」「周辺」は**デバイスの設計図から決まる幾何学的な定義**であって、
グラフ指標から導かれたものではない。本プロジェクトでも同じで、
`BridgeSelector(kind="inter_cluster")` はクラスタ間ブリッジ `BX*` を part 名の接頭辞で
選んでいるだけである。

一方でグラフ側の部品は既に揃っている:

| 部品 | 場所 |
|---|---|
| participation coefficient / within-module z-score | `analysis/graph.py::module_participation` |
| Guimerà–Amaral の役割分類 | `analysis/graph.py::classify_roles` |
| 媒介中心性 | `analysis/graph.py::betweenness_centrality` |
| グラフ指標でのハブ選択 | `scripts/lesion/analysis/selectors.py::HubSelector` (`hub:metric=participation` 等) |
| 切断シナプスごとの participation / z の記録 | `scripts/lesion/analysis/metrics.py` |

**足りないのは部品ではなく、2 つの定義を突き合わせる評価軸のほう。**
`bridge:kind=inter_cluster`（幾何）と `hub:metric=participation`（グラフ）は現状まったく
無関係なセレクタとして並んでおり、両者が同じものを指しているかを確かめる手段がない。

### やること

1. **一致度の検証（最優先）**
   `BX*` を通る軸索を持つニューロンが、`classify_roles` で実際に `connector_hub`
   （高 P・高 z）に分類されるかを見る。`BC*` しか通らないニューロンが
   `provincial_hub` / `peripheral` 側に落ちるかも同様。
   一致すれば「幾何学的ハブ＝グラフ理論的ハブ」が本モデルで成り立つことの検証になり、
   論文の定義をそのまま借りてよい根拠になる。**ずれた場合のほうが情報量が大きい** ——
   デバイス設計上のハブが、生成された配線ではハブになっていないという意味で、
   `axon_growth` のパラメータ（`segment_length` とブリッジ幅の関係、注記 16d）を
   疑う入口になる。

2. **評価軸としての利用**
   グラフ指標を「どこを切るか」（セレクタ）だけでなく「切った結果どう変わったか」
   （評価）にも使う。損傷前後で participation / z / betweenness / `component_sizes` が
   どう動いたかを比較すれば、論文が transfer entropy で見ていた再編を
   構造側から測ることになる。`metrics.py` は現在**切断前**の値しか持たないので、
   切断後の再計算と差分が要る。

3. **論文との対応づけ**
   論文が示していないのはまさにこの突き合わせなので、一致度の検証は
   再現ではなく**補強**にあたる。結果は論文比較のドキュメントに残す。

### 注意

- `classify_roles` の閾値（`hub_z=2.5`, `provincial_p=0.62`, `connector_p=0.80`）は
  Guimerà–Amaral の原論文の値で、**16 モジュール・数千ニューロンの本モデルで妥当かは
  未検証**。一致度を見る前に、P と z の分布そのものを確認すること。
- `module_participation` はモジュールラベル（`module` 軸）を要求する。
  `soma_in_bridge: true` だとブリッジ上のニューロンが `BC*` / `BX*` というラベルを持ち、
  「モジュール」の意味が壊れる。`hierarchical_modular_grid` では
  `soma_in_bridge: false` が前提（CLAUDE.md の該当節、および論文のマイクロチャネルが
  幅 4.5 µm でソーマを通さないという実験事実に対応）。

---

## 検討事項: `src/data/` (DataLoader) と入力設定の在処（放置中）

**結論だけ先に:** `src/data/` は共有層に置かれた汎用入力レイヤのように見えるが、実態は
**モデル手動テストの刺激治具**である。3 実験のどれも使っていない。気持ち悪さの正体は
「実験ごとに扱いがバラバラ」ではなく「そもそも共有されていない」ほう。

以下は 2026-09-22 時点の調査。行数・件数は当時のもの。

### 現状

| 使っている | 使っていない |
|---|---|
| `scripts/tools/pipeline_check.py` | `scripts/develop/` |
| `scripts/tools/spike_animation.py` | `scripts/akita_soc/` |
| `test/models/**` の手動テスト 7 本 | `scripts/lesion/` |

`run_one.py` に `push` も `generate` も現れない。実験のループは `sim.step(n)` だけ。

登録されている DataLoader:

| プロファイル | 実体 |
|---|---|
| `pqn_test` / `lif_test` / `stdp_test` | 全ニューロンに一定電流を矩形で入れるだけ |
| `spatial_recognition` | **どの config からも選ばれず、import もされていない** ([spatial_loader.py](../src/data/spatial_loader.py) 103 行) |
| `audio_loader.py` | **0 バイト** |

`src/data/` 370 行のうち、生きているのは単体ニューロン試験用の矩形電流だけ。

### 実験が使わないのは偶然ではない

駆動方式が 2 種類あり、DataLoader は片方しかカバーしていない。

| | 誰が作る | いつ | config の在処 |
|---|---|---|---|
| **ホスト駆動** | `DataLoader` が numpy 配列を作って `push()` | 毎ステップ | `task.input` |
| **自走** | `NetworkBuilder` が GeNN の CurrentSource を張る + ニューロン内部の escape noise | build 時に 1 回 | `inputs.GaussianNoise` |

3 実験は全部後者。**ホストが押し込むものが無い**ので DataLoader を通る理由がない。

「入力形式の扱いがバラバラ」に見えるのはここで、**入力の設定が `inputs:` と `task:` の
2 箇所に分かれ、どちらを読むかが駆動方式で決まる**。`task.yaml` のコメントが言う
「『いつ記録するか』と『どんなタスクか』が同じ棚に載っている」の後者がこれ。実験は
自分の半分を `scripts/<実験>/task.yaml` へ持ち出したので、`configs/components/tasks.yaml`
に残っているのはほぼ DataLoader 側の半分になっている。

### 実害が出ているのは `duration` の二重定義

[base_loader.py](../src/data/base_loader.py#L31) は `total_steps = duration / dt` とする
= **「duration = run の長さ」**。

一方、実験では duration は run 長を決めない（決めるのは `record_hours` の最大値 +
`record_window_ms`）。**同じキーが、DataLoader を通るかどうかで別の意味になっている。**
これは整理の問題ではなく単純に直すべき箇所。

### やること（優先度順）

1. **死んでいるものを消す。** `audio_loader.py`（空）/ `spatial_loader.py`（参照ゼロ）/
   `configs/components/tasks.yaml` の `spatial_recognition` / [NetworkBuilder.py の
   `__main__` ブロック](../src/core/NetworkBuilder.py#L838)（`ConfigManager(config_src,
   "pqn_test")` と呼んでおり、引数なしコンストラクタの現在は動かない）。
   これだけで「汎用レイヤに見えるが中身が無い」状態は消える。
2. **残ったものに正しい名前を付ける。** `src/data/` の冒頭に「ホスト側から電流を押し込む
   刺激。3 実験はどれも自走系なので使っていない」と書く。
3. **`duration` の二重定義を解消する。**
4. **入力の在処を 1 つにまとめるのは、host-driven の実験を実際にやる時でよい。**
   今やっても使われない抽象が 1 段増えるだけ。

### 動かす条件

**入力刺激を与える実験を実際に始めるとき。** 当面 develop / akita_soc / lesion の路線なら
4 は後回しでよい。

その時が来たら、プロジェクト自身のルール（「まとめる単位は呼び出し元の数ではなく
何の都合で変わるか」, CLAUDE.md）に従えば **刺激プロトコルは `scripts/<実験>/` 側**になる。
図と同じ扱いで、共有するのは数式だけ。`spatial_recognition` が共有層で干からびたのは、
置き場所が最初から逆だったためと見ている。

### 注意

- **`BaseDataLoader` という枠は残す。** CLAUDE.md の不変条件 3 が挙げる 8 つの拡張点の
  1 つであり、拡張点として正しい。問題は枠の存在ではなく、枠が「シミュレータの入力層」を
  名乗っていること。
- `scripts/tools/pipeline_check.py` は Config → Registry → Builder → Simulator →
  DataLoader → 作図の疎通確認そのものなので、**DataLoader が `src/` から import できる
  必要は残る**（`test/` へ移すと `scripts/tools/` から参照できなくなる）。
