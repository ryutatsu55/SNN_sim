# RunView — 実験出力の読み出し契約

`src/utils/runview.py`

解析・描画関数が「develop のデータ」ではなく**契約**に対して書かれるようにするための取り決め。
これがあると、同じ関数が実験をまたいで動き、共有層と実験固有層の間の移動が
`git mv` 1 回で済む。

関連: `docs/architecture/experiment_structure.md`（この契約を使う側の構成） /
`docs/scripts_unification_plan.md`（なぜこの形に統一したか）

---

## 1. なぜ必要か

いまの `scripts/develop/` には、同じ図を描くための橋渡しが 2 か所にある。

- `run_one.py` … GeNN の in-memory の値から図へ
- `replot.py` … npz から図へ

[run_one.py の記録処理](../../scripts/develop/run_one.py) には、こういう行がある。

```python
# 再解析が npz から組み立てるのと同じ形にして渡す (描画経路を 1 本に保つため)。
window_trace = Trace(V=trace_v, I=trace_i, dt=dt, ...)
```

**本番の run が「npz から読んだらこうなるはずの形」を手で組み立てている。**
片方だけ直せば静かにずれる。`smax` / `wmax` / `order_axes` も両方が独立に計算している。

契約を決め、**本番も再解析も必ず一度ファイルに書いてから読み直す**ことにすれば、
この 2 経路が 1 本になる。

---

## 2. 基底 — `RunView`

```python
class RunView:
    """1 つの run を、ある時点から見たもの。"""
    run_dir: Path
    config                      # 解決済み config (Pydantic)
    layout: NetworkLayout
```

**持つデータはこの 3 つだけ。** 実験ごとに変わるものを基底に置かない。
他はすべて「その名前で呼べる」ことだけを決めたアクセサで、**既定は `MissingData` を投げる**。

```python
class MissingData(Exception):
    """この run はそのデータを持たない。バグではない。"""
```

### アクセサ一覧

| 名前 | 返すもの | 意味 |
|---|---|---|
| `wiring()` | `Wiring(row, col, shape)` | **どこに結合があるか。** run を通して不変 |
| `coo()` | `Coo(row, col, weights, delays, shape)` | **その時点の結合。** 時点が定まる view だけが持つ |
| `weights()` | `np.ndarray` (1D, `wiring` と index 整合) | その時点の重み |
| `coords()` | `np.ndarray` (N, 2) | soma の座標 |
| `area()` | `BaseArea` 相当 | ニューロンが置かれる領域 |
| `geometry()` | 軸索ジオメトリ | 軸索の折れ線 |
| `spikes()` | `Spikes(times, ids)` | スパイク列 |
| `trace()` | `Trace(V, I, dt, neuron_id, ...)` | 膜電位トレース |
| `metrics()` | `pd.DataFrame` | 指標の表 |

`wiring()` と `coo()` を分けてあるのは、**片方が run 不変でもう片方が時点依存**だから。
`wiring()` はどの view でも同じものを返し、`coo()` は「いつの重みか」が決まる view にしか無い。

> **決定の修正:** 会話では `connectivity` を `coo` へ改名すると決めたが、
> 実装に落とす段で「run 不変の配線」と「時点ごとの重み」を分けた方が素直だと分かったので
> 2 つに割った。名前の衝突 (`analysis/connectivity.py`) を消すという当初の目的は満たす。

### エラー処理

図は分岐を書かない。**無いものを要求して、投げられたまま素通しする。**

```python
# figures/network.py
def network(view, out_path) -> None:
    coords = view.coords()          # 無ければ MissingData がそのまま上へ
    ...
```

受けるのは登録簿（`report/__init__.py` の `guard()`）。

```python
def guard(label, fn, *args):
    try:
        fn(*args)
    except MissingData as missing:
        print(f"  skip {label}: {missing}")           # 無いから出ない
    except Exception as error:
        print(f"  Warning: {label} failed: {error}")  # バグで出ない
```

**「無い」と「壊れた」を分けるのが要点。** いまは両方 `Warning:` で出るので、
`no_space` の config で空間依存の図が出ないのと、コードのミスで落ちるのが、
ログ上で見分けられない。

---

## 3. 3 つの view

実験の進行は 3 つの時点に分かれ、それぞれに view が対応する。

| view | いつ | 持つもの |
|---|---|---|
| `Built` | build 直後。まだ回していない | `wiring` `coo` `coords` `area` `geometry` |
| `Window` | 記録窓 1 つ | `hour` `record_start_ms` `wiring` `coo` `weights` `spikes` `trace` |
| `Series` | run 全体 | `windows` `wiring` `metrics` |

```python
class Built(RunView):   ...
class Window(RunView):  hour: float;  record_start_ms: float
class Series(RunView):  windows: tuple[Window, ...]
```

### `hour` の定義

```
hour: float = 基準時刻からの経過時間 [h]。基準を何に置くかは実験が決める。
```

**数値であって識別子ではない。** 点の間隔に意味がある:

- `figures/fig2c.py` … `ax.plot(hours, ...)` / `ax.set_xlim(0, hours.max())`
- `figures/fig2d.py` … `np.repeat(hours[:, None], n_neurons, axis=1)`
- `Series.windows` の並べ替えキー
- `metrics.csv` の `hour` 列

記録時刻は等間隔とは限らない (`record_hours: [0, 1, 6, 12]` が実際にある) ので、
識別子に置き換えると横軸が壊れる。

**正は npz が持つ。** `hour = record_start_ms / 3600000`。ファイル名の `{hour:g}` は
有効数字 6 桁なので、非整数の記録時刻は往復しない。

基準の例:

| 実験 | 基準 | `hour` の値 |
|---|---|---|
| develop | run の開始 | `0, 1, 2, … 72` |
| lesion | **切断時刻** | baseline が負 (`-0.167` など)、post が `0, 1, 2, … 12` |

lesion の `baseline` / `post` の区別は契約の外。**リーダーが `phase` として自分で足す。**
契約が要求するのは `hour` だけなので、共有の図は両方の実験で動く。

### 時刻の規約

**記録窓の中の時刻はローカル（窓の先頭 = 0）。**

```python
window.spikes().times        # ローカル [ms]
window.record_start_ms       # 絶対へ戻すための原点
```

**絶対時刻は返さない。** 以前は `times`（絶対）と `local_times` の両方を持ち、どちらを
使うかを呼び出し側が選んでいた。選び間違えてもエラーは出ず、burstiness のビン割りだけが
静かにずれる。絶対時刻を必要とする消費者が 1 つも無いことを確認したうえで落とした。

---

## 4. 分岐してよいもの・だめなもの

| 層 | 場所 | 分岐 |
|---|---|---|
| **契約** (`RunView` / `MissingData` / `guard` / `Wiring` / `Coo` / `Spikes` / `Trace`) | `src/utils/runview.py` | **だめ。** ここが drift したら意味が消える |
| `Built` の実装 (`BuiltNetwork`) | `src/utils/runview.py` | **だめ。** build 直後に手元にあるのはどの実験でも同じ `NetworkBuilder` 1 つで、分岐する余地が無い |
| リーダーの実装 (`Window` / `Series`) | 各実験の `store/` | 実験ごと。記録ファイル名の規約が違う |
| 解析・描画関数 | 契約にしか依存しない → **どちらにも置ける** | 図ごとに判断 |
| ステージ登録簿 | 各実験の `report/` | 実験ごと |

**「図は実験固有」と「図は共有」の二者択一が、図ごとの判断に変わる。** これが契約の主な効果。

### 実際にどう分けたか (2026-09-22)

**図は実験へ、契約は共有層へ。**

- `src/utils/plotting/` にあるのは**道具 (`scripts/tools/`) が使う図**だけ。全関数が
  `(view, out_path)` を取る。例外は 2 つ —— `ax` を取るプリミティブ (`draw_area` /
  `draw_discrete_distribution` / `draw_block_boundaries`) と `model_test.py`
  (`PQN_test` / `neuron_test` / `stdp_window`)。後者が契約を取らないのは**描く対象が
  run ではない**から (DataLoader を 1 本回して `pull()` した配列で、run ディレクトリも
  記録窓も持たないので `run_dir` と `layout` が埋まらない)
- 実験 (`scripts/develop/` / `akita_soc/` / `lesion/`) は自分の `figures/` を持ち、
  必要なら共有層から**複製する**
- 数式 (`src/utils/analysis/`) は逆に共有したまま。**図は分岐が目的、数式は分岐がバグ**

契約が `scripts/tools/` ではなく `src/utils/` にあるのは、`src/utils/plotting/` の図が
契約に対して書かれているから。`scripts/` 側に置くと `src/utils` → `scripts` という
逆向きの依存ができる。契約自体は「どのファイルをどう読むか」を一切知らない
(NetworkBuilder も matplotlib も import しない) ので、シミュレータ側の語彙として置ける。

### U1 が解ける

損傷実験は develop の run ディレクトリを読むので、記録ファイル名の規約は
develop の内部事情ではなく **develop と lesion の契約**だった
(U1)。横 import を禁じた以上、これをどう共有するかが
問題になっていた。

契約があれば、lesion は契約の実装を使うだけで develop の中身を import しない。

実際にそうなった: `scripts/lesion/store/parent.py` が「develop 形式の記録を読む」実装を
**lesion 側に**持つ。読むのは 5 つ (`config.yaml` / `connectivity.npz` /
`weights_{h}h.npz` / `layout_axes.npz` / `axon_geometry.npz`) で、これは develop の内部
事情ではなく develop と lesion の契約。規約が変わったら直す場所は 1 つで、そのことが
モジュールの docstring に書いてある。

---

## 5. テストへの影響

図が配列ではなく view を受け取るので、**合成配列での単体テストはできなくなる。**
小さい npz を書いた run ディレクトリを作るフィクスチャ方式になる。

主な守りは `test/experiments/develop/reference_figures.md5`（PNG のバイト一致）なので
実害は小さい。各実験のテストが `_make_run()` で小さな run を組み立てる。

**共通フィクスチャへは上げなかった。** 記録ファイル名の規約が実験ごとに違う以上、
`_make_run()` は実験の持ち物になる。共有したのは**ネットワーク設定**だけで、
`test/experiments/no_space_100.yaml` を 3 つの実験のテストが使う —— 実験の config を
借りると、実験の条件を変えるたびに別の実験のテストが落ちるため。

配列だけを渡す単体テストが要るときは `MemoryWindow` で契約に載せられる
(`test/utils/test_plotting_ordering.py` がそうしている)。ただし**実験の本番経路では
使わない** —— 「一度書いてから読み直す」ことで本番と再解析の描画経路を 1 本に保つのが
develop で得た形なので、そこを近道すると元に戻る。
