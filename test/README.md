# test/ — 何がどこにあり、いつ走らせるか

プロジェクト全体の概要はルートの `README.md` を参照。ここでは `test/` 配下の構成と、
各テストが何を保証しているかをまとめる。

---

## 分類軸: 何を検証しているか（`src/` の構造をそのまま鏡写しにする）

```
test/
├── conftest.py             pytest 共通の環境セットアップ (sys.path / MPLCONFIGDIR / Agg)
├── test_*.py                層をまたぐ不変条件         → src/ 全体
├── models/
│   ├── network/    test_*.py  接続・重み・遅延の生成ロジック → src/models/network/
│   ├── neurons/    <名前>/    ニューロンモデルの発火動態    → src/models/neurons/
│   └── plasticity/ <名前>/    可塑性モデルの動態           → src/models/plasticity/
├── utils/          test_*.py  解析と描画                  → src/utils/
└── archive/                   完了済みの一回性検証。走らせなくてよい
```

**ディレクトリは検証対象で決まる。実行方法（pytest か GeNN 手動実行か）はディレクトリを
決めない** — それは各ファイルの性質であり、下の一覧表の列で表現する。結果として
`models/network/` は pytest 形式、`models/neurons/` と `models/plasticity/` は GeNN 手動形式に
偏っているが、それは対象の性質上そうなっているだけで、たとえば `models/neurons/PQN_test/` の
`PQN_Euler_test.py` は GeNN 不要な pytest 未満のスクリプトでも同じ場所に置く
（`pqn_single_test.py` と同じ「PQN ニューロンの正しさ」を検証しているため）。

---

## 見分け方: pytest が拾うか / 手で叩くか

| | 形 | 依存 | 実行 |
|---|---|---|---|
| **自動** | フラットな `test_*.py` | 純 numpy。GeNN も GPU も不要 | `pytest test/ -q` |
| **手動** | `<名前>/test.py` + `test.yaml` + 出力 png のディレクトリ1セット | **GeNN ビルド必須**。図を出して目視する | `python <パス>/test.py` |

手動テストは `if __name__ == "__main__"` で守られ、ファイル名も `test.py`
（pytest の `test_*.py` / `*_test.py` に一致しない）ので、**`pytest test/` は手動テストを
一切収集しない**。1 本ずつ明示的に走らせる。

---

## `test/` 直下 — 層をまたぐ不変条件 [自動]

| ファイル | 壊れたら何が起きるか | 件数 / 時間 |
|---|---|---|
| `test_global_coo.py` | 疎/密の生成経路が同じ COO を返さなくなる。build 以降を COO 一本にできる根拠そのもの | 5 / 0.4s |
| `test_axon_growth.py` | 軸索が領域外へ出る・接していない部分領域が結合する・area が RNG を消費する・`DiskArea` の draw 順が `RandomCircle2DSpace` とずれる | 93 / **107s** |
| `test_area_plotting.py` | `plot_area()` が境界を閉じない / bbox 全体を塗る / 座標の有無で誤判定する。SDF 実装で踏んだ罠 3 つを固定 | 13 / 3.5s |
| `test_lesion_builder.py` | `replace_global_coo()`（損傷実験の構造的除去の土台）が壊れる | 25 / 1.3s |
| `test_akita_soc.py` | Akita モデルの素の数式（escape noise / conductance LIF / STDP kernel / STP）と replot 経路 | 31 / 2.1s |
| `experiments/develop/test_develop.py` | develop 実験。「論文の 100」が N に追従すること / task プロファイルを config が選ぶこと / seed 範囲の展開 / `config.yaml` が build を通った記録に限られること / 本番と再解析が同じ列を作ること | 30 / 4.5s |

**所要時間の 9 割が `test_axon_growth.py` 1 本**（107 秒。残り全部で 11 秒）。
軸索・領域に触っていないなら `pytest test/ -q --ignore=test/test_axon_growth.py` で十分。

---

## `test/models/network/` — 接続・重み・遅延の生成ロジック [自動]

`src/models/network/` の connector / weight / delay クラス単体の引数バリデーション。

| ファイル | 検証対象 | 件数 |
|---|---|---|
| `test_block_random_topology.py` | `BlockRandomTopology` の引数バリデーション | 6 |
| `test_random_delay.py` | `RandomDelay` の引数バリデーションとクリップ | 4 |
| `test_offset_scaled_normal_weight.py` | `OffsetScaledNormalWeight` の引数バリデーション | 4 |

---

## `test/models/neurons/` — ニューロンモデルの発火動態 [主に手動]

GeNN をビルドして実際にスパイクを出す。**自動テストがカバーしていない唯一の経路**
（`NetworkBuilder` → GeNN 登録 → `simulator.push/step/pull/reset` → カスタムモデルの実発火）。

| ファイル | 何が分かるか | 判定 |
|---|---|---|
| `SingleNeuronTest/test.py` | LIF 単一ニューロンが入力に対して発火するか | 発火時刻の出力 + `neuron_test.png` を目視 |
| `AkitaEscapeLif/test.py` | `akita_escape_lif` の escape noise による確率発火 | 発火時刻の出力 + png を目視 |
| `PQN_test/pqn_single_test.py` | PQN ニューロンを GeNN 経由で回す | `PQN_V_test.png` を目視 |
| `PQN_test/PQN_Euler_test.py` | `PQNengine` の整数版と float Euler 版の一致（**GeNN 不要**） | `PQN_Euler_test.png` を目視 |

---

## `test/models/plasticity/` — 可塑性モデルの動態 [主に手動]

`TransTest` はシナプス結合そのものではなく、`src/models/plasticity/standard_models.py` の
`StaticPulseDendriticDelay`（樹状突起遅延を担う可塑性モデル）を検証しているのでここに置く。

| ファイル | 何が分かるか | 判定 |
|---|---|---|
| `TransTest/test.py` | 樹状突起遅延つきのシナプス伝播が効くか（`optional_connect` で 1→0、遅延 10 ms） | 2 ニューロンの発火時刻の差を目視 |
| `STPtest/test.py` | 短期可塑性 (STP) の資源枯渇が効くか | `neuron_test.png` を目視 |
| `STDPtest/test.py` | STDP ウィンドウの形（201 トライアル、1 トライアル = pre→post 1 ペア） | `stdp_window.png` を目視 |
| `STDPtest/test_multi_spike.py` | **複数スパイクが交錯したときの trace 蓄積と、伝播遅延に対するロールバック**。E-STDP / I-STDP × 5 シナリオを、到着時刻ベースの全ペア和リファレンスと突き合わせる | **自動判定** (`All scenarios PASSED.` / 非 0 終了) |

`STDPtest/test_multi_spike.py` だけは PASS/FAIL を自分で判定するので、目視不要で CI に載せられる。
ファイル名が `test_*.py` なので pytest には import されるが、テスト関数を持たないので 0 件収集になる。
「custom_Akita のロールバックは in-transit の pre スパイクが 1 本という前提（delay < TauRefrac が
必要）」を守っているのはこのテストだけなので、遅延・不応期まわりを触ったら必ず走らせること。

### `models/neurons/` `models/plasticity/` の config は「保存済み config」として扱われる

各ディレクトリの `test.yaml` は `ConfigManager.load_resolved()` で読まれる（`resolve()` ではない）。
つまり**解決済みの形**で書く必要があり、以下が必須:

- `simulation.seed` と `simulation.backend` に具体値（`null` や欠落は `NetworkBuilder` で raise）
- `layout.assignment`
- `neurons.*.polarity`
- `synapses.*.source` は**リスト**

現在は全ファイル `seed: 42` / `backend: cpu` に固定してある（どれも数ニューロンなので CPU で十分、
かつ CUDA/WSL の環境差を避けられる）。

---

## `test/utils/` — 解析と描画 [自動]

| ファイル | 壊れたら何が起きるか | 件数 |
|---|---|---|
| `test_connection_probability.py` | 群間結合確率の分母が「ありうるペア数」でなくなる / 対角から自己ペアが抜けない / プールが単純平均になる | 15 |
| `test_beggs_plenz.py` | べき指数・σ の推定量が既知の合成データを復元できなくなる | 19 |
| `test_criticality_index.py` | ΔCr の符号・スケール（Tetzlaff の \|Δp\|=0.195 と同じ土俵）が崩れる | 12 |
| `test_lesion_analysis.py` | `axons` / `graph` / `isi` と復元経路。area をダックタイピングで受ける契約の検証も兼ねる | 27 |
| `test_plotting_ordering.py` | 表示用並べ替え軸（単軸=極性、多軸=外でブロック化・内でソート） | 13 |
| `test_spike_animation.py` | スパイクアニメーションの減衰強度 | 2 |
| `test_spike_csv.py` | `export_spike_csv` の出力行 | 2 |

> **現在 `test_criticality_index.py` の一部が失敗することがある。** `criticality.py` の `smin` を
> 作業中に変更している間の一時的な状態であり、腐敗ではない。他のテストは常に通る状態を保つ。

**自動テスト全体でおよそ 280 件強 / 約 2 分**（`pytest test/ -q`）。

---

## 運用ルール

1. **新しいテストは対象で置き場所を決める。** 「何を検証するか」が `src/` のどこに対応するかを
   先に決め、実行方法（pytest か手動か）はその結果として決まる。逆に実行方法から置き場所を
   決めない。
2. **新しいテストの冒頭 docstring に「守りたい不変条件」を書く。**
   この表の「壊れたら何が起きるか」列はその 1 行要約にすぎない。詳細は docstring が正であり、
   表を詳しくして二重管理にしない。
3. **自動テストはフラットな `test_*.py`、手動テストは `<名前>/test.py` + `test.yaml` の
   ディレクトリ 1 セット。** この形が唯一の見分け方なので崩さない。手動テストを
   `test_*.py` の名前で直置きすると、pytest が 0 件収集するだけの無意味なファイルになる。
4. **意味を失ったテストはコメントアウトせず削除する。** git が履歴を持っている。
5. **一回性の検証スクリプトは `test/archive/` へ。** 完了日と結論を README に書いて凍結する。
   回帰テストとして毎回走らせる意味がないものを `test/` 直下に置かない。
6. **手動テストを壊す変更をしたら、その場で yaml を直す。**
   pytest が拾わないので、放っておくと壊れたことに気づけない。実際、NetworkLayout の
   リファクタの際に手動テストの yaml が複数取り残され、`polarity` / `source` / `layout` の
   3 項目で全滅したまま長期間気づかれなかったことがある。**モデルや config スキーマを
   変えたら、対応する `test/models/{neurons,plasticity}/<名前>/test.py` を 1 回走らせること。**
7. **環境セットアップは `conftest.py` に任せる。** `sys.path` 通し・`MPLCONFIGDIR`・
   matplotlib の Agg 固定は conftest が済ませるので、新しい自動テストに書かない
   （手動テストと `test/archive/` は pytest 経由で走らないので各自で通す）。
8. **自動テストの命名は `test_*.py`。** `*_test.py` も pytest は拾うが、混在させない。
