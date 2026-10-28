# test — 何がどこにあり、いつ走らせるか

`test/` 配下の構成と、各テストが何を保証しているかをまとめる。プロジェクト全体の概要はルートの `README.md`。

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
├── experiments/    <実験>/    実験ディレクトリの記録規約と読み出し → scripts/<実験>/
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

| ファイル | 検証する対象 | 保証すること | 件数 / 時間 |
|---|---|---|---|
| `test_global_coo.py` | `NetworkBuilder.global_coo()` が返す `GlobalCOO` | row-major ソート済み・dtype・配列長の不変条件を満たす | 3 / 0.4s |
| `test_network_golden.py` | config と seed から NetworkBuilder が組み上げるネットワーク (結合・重み・遅延・座標・軸索幾何) | 同じ config・seed からビット単位で同じネットワークができる | 1 / 0.5s |
| `test_axon_growth.py` | 領域 (`area.py`) と軸索伸長 (`AxonGrowthTopology`) | 軸索は線分全体が領域内に留まり、接していない部分領域どうしは結合せず、area は乱数を消費せず、`DiskArea` の draw 順は `RandomCircle2DSpace` と一致する | 92 / **108s** |
| `test_area_plotting.py` | 領域と軸索の描画 (`plotting/area.py` など) | 境界が閉じ、塗りが領域の内側に限られ、描くかどうかを `is_bounded` で決め、描画が乱数に触れない | 13 / 3.5s |
| `test_config_loading.py` | YAML の読み込み (`load_yaml()`) とリポジトリにある config | 重複キーをエラーにし、実在する config・プロファイルに重複キーが無く、メイン config に書いた値がプロファイルより優先される | 11 / 0.2s |
| `test_akita_model.py` | Akita モデル (escape LIF・STP・STDP カーネルと trace) と指標 (べき乗フィット・E/I ブロック統計) | 実装が論文 (補足資料) の式どおりの値を返す | 22 / 0.5s |
| `experiments/test_report_registries.py` | 3 実験の出力の登録簿 (`scripts/<実験>/report/`) | 登録簿が表の形を保ち、名前・ファイル名が重複せず、報告の種類が `store/paths.py` に登録されている | 15 / 0.8s |
| `experiments/develop/test_develop.py` | develop 実験の記録規約と読み出し | 本番と再解析が同じ指標を同じ場所に作り、記録から読み直した時刻と `Built` が本番と一致する | 61 / 7s |
| `experiments/akita_soc/test_akita_soc.py` | akita_soc 実験の記録規約と読み出し | 記録ファイル名が往復し、時刻が npz から来て、指標の列が揃い、他の実験の記録と取り違えない | 11 / 1.4s |
| `experiments/lesion/test_lesion.py` | 損傷実験の切断 (`replace_global_coo()`・切断 spec)・run の命名・可塑性の凍結・記録の読み直し | 切断は指定どおりのシナプスだけを除いて COO の不変条件を保ち、切断後のネットワークと比較表が記録から作り直せる | 50 / 2.6s |
| `experiments/lesion/test_lesion_analysis.py` | 損傷実験用の解析 (ブリッジ横断・グラフ指標・ISI・重みの引き当て・迂回経路・駆動の収支) | 各指標が合成データに対して定義どおりの値を返し、重みは位置ではなく (pre, post) で引き当てられる | 54 / 0.5s |
| `experiments/lesion/test_compare.py` | run をまたぐ比較 (`scripts/lesion/compare`) | 低下・回復などの要約が定義どおりで、条件の違う run (未完了・可塑性を止めたもの) を混ぜない | 17 / 0.6s |

`experiments/no_space_100.yaml` は pytest ではなく、実験テストが共通で使うネットワーク設定 (N=100, `no_space`)。

**所要時間の 9 割が `test_axon_growth.py` 1 本**（108 秒。残り全部で 14 秒）。
軸索・領域に触っていないなら `pytest test/ -q --ignore=test/test_axon_growth.py` で十分。

---

## `test/models/network/` — 接続・重み・遅延の生成ロジック [自動]

`src/models/network/` の connector / weight / delay クラス単体の生成結果と引数バリデーション。

| ファイル | 検証する対象 | 保証すること | 件数 |
|---|---|---|---|
| `test_coo_generation.py` | 全 connector / weight / delay の `generate()` | 結合は int32・row-major・重複なし、重み・遅延は結合 1 本に 1 値の float32 で、同じ seed から同じ実現値と同じ rng 状態になる | 52 |
| `test_block_random_topology.py` | `BlockRandomTopology` の引数 | 不正な引数を生成前に弾く | 6 |
| `test_random_delay.py` | `RandomDelay` の引数とクリップ | 不正な引数を弾き、遅延を指定範囲にクリップする | 4 |
| `test_offset_scaled_normal_weight.py` | `OffsetScaledNormalWeight` の引数 | 不正な引数を弾く | 4 |

---

## `test/models/neurons/` — ニューロンモデルの発火動態 [主に手動]

GeNN をビルドして実際にスパイクを出す。**自動テストがカバーしていない唯一の経路**
（`NetworkBuilder` → GeNN 登録 → `simulator.push/step/pull/reset` → カスタムモデルの実発火）。

| ファイル | 検証する対象 | 保証すること | 判定 |
|---|---|---|---|
| `SingleNeuronTest/test.py` | LIF 単一ニューロン | 入力に対して発火する | 発火時刻の出力 + `neuron_test.png` を目視 |
| `AkitaEscapeLif/test.py` | `akita_escape_lif` | escape noise によって確率的に発火する | 発火時刻の出力 + png を目視 |
| `PQN_test/pqn_single_test.py` | PQN ニューロン (GeNN 経由) | GeNN 上で膜電位が時間発展する | `PQN_V_test.png` を目視 |
| `PQN_test/PQN_Euler_test.py` | `PQNengine` の整数版と float Euler 版（**GeNN 不要**） | 両者の軌道が一致する | `PQN_Euler_test.png` を目視 |

---

## `test/models/plasticity/` — 可塑性モデルの動態 [主に手動]

`TransTest` はシナプス結合そのものではなく、`src/models/plasticity/standard_models.py` の
`StaticPulseDendriticDelay`（樹状突起遅延を担う可塑性モデル）を検証しているのでここに置く。

| ファイル | 検証する対象 | 保証すること | 判定 |
|---|---|---|---|
| `TransTest/test.py` | 樹状突起遅延つきのシナプス伝播（`optional_connect` で 1→0） | 指定した遅延 (10 ms) で伝わる | 2 ニューロンの発火時刻の差を目視 |
| `STPtest/test.py` | 短期可塑性 (STP) | 資源枯渇で応答が減る | `neuron_test.png` を目視 |
| `STDPtest/test.py` | STDP ウィンドウ（201 トライアル、1 トライアル = pre→post 1 ペア） | ウィンドウが期待した形になる | `stdp_window.png` を目視 |
| `STDPtest/test_multi_spike.py` | 複数スパイクが交錯したときの trace 蓄積と、伝播遅延に対するロールバック (E-STDP / I-STDP × 5 シナリオ) | 到着時刻ベースの全ペア和リファレンスと一致する | **自動判定** (`All scenarios PASSED.` / 非 0 終了) |

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

| ファイル | 検証する対象 | 保証すること | 件数 |
|---|---|---|---|
| `test_connection_probability.py` | 群間結合確率 (`analysis/connectivity.py`) | 分母がありうるペア数 (対角は自己ペアを除く) で、方向を区別し、群のプールは分子と分母を別々に足して割る | 15 |
| `test_avalanche_criticality.py` | アバランシェ検出と臨界性指標 (Beggs & Plenz の 4 指標) | 指数・σ が既知の合成データから正しく復元される | 19 |
| `test_criticality_index.py` | ΔCr (Ikeda-Akita-Takahashi 2023) と LLR | ΔCr がべき乗で ~0・劣臨界で負になり、尺度が標本数で動かず、回帰直線が確率 1 を超えない | 18 |
| `test_correlation.py` | ニューロン間相関 (Watanabe et al. 2026 式 1) | `np.corrcoef` と一致し、行列の非対角平均と閉じた形が一致し、黙ったニューロンは `nan` として平均から除かれる | 8 |
| `test_synchrony.py` | 同期イベント・バースト・群ごとの χ (`analysis/synchrony.py`) | 参加率に発火率で決まる上限があり、それで割った値は発火率だけの変化では動かず、χ・バースト・全体イベントが定義どおりの値になる | 20 |
| `test_transfer_entropy.py` | 移動エントロピー GTE と RTE (Watanabe et al. 2026 式 2・3) | 素朴な数え上げと一致し、遅れて駆動する向きにだけ値が立ち、RTE は損傷前に固定した結合集合 C の上で測られる | 6 |
| `test_plotting_ordering.py` | 表示用の並べ替え (`plotting/ordering.py`) と図の引数 | 単軸では興奮性が先頭、多軸では外側の軸でブロック化して内側で並べ、図が `(view, out_path)` を取る | 13 |
| `test_spike_animation.py` | スパイクアニメーションの減衰強度 | 指数減衰し、同じニューロンでは最大値を保つ | 2 |
| `test_spike_csv.py` | `export_spike_csv` | 期待した行を書き、長さの合わない入力を弾く | 2 |

**自動テスト全体で 519 件 / 約 2 分**（`pytest test/ -q`、2026-10-27 実測 122 秒）。
