# develop 実験

Akita SoC Fig.2 相当の発達実験。ネットワークを長時間走らせ、指定した時刻に記録窓を取って
スパイク・重み・指標・図を残す。

---

## 走らせる

```bash
python -m scripts.develop --config axon_growth_grid --condition gmax25_baseline
```

**引数はこの 2 つだけ。**

| 引数 | 意味 |
|---|---|
| `--config` | config ファイル名。**名前だけなら `scripts/develop/` の中**を見る。パスを含めれば `configs/` のものも指定できる |
| `--condition` | 条件名。`outputs/develop/<条件名>/` になる |

受けるのは「**どの config か**」と「**条件名**」だけ。seed も並列度も
記録条件もトレースも、すべて config 側にある。

**条件名は「条件の記述」ではなくラベル。** 何で走ったかは run の `config.yaml` が持つので、
名前にすべての条件を詰め込む必要はない。人が後で見分けられればよい。

**同名の条件が既にあると拒否して止まる。** 前の結果と混ざらないようにするため。やり直す
なら別名を付けるか、古い方を手で消す。

---

## 条件の変え方

**config ファイルは複製しない。** `scripts/develop/<名前>.yaml` を直接書き換えて走らせる。
過去にやった実験の条件は run ディレクトリの `config.yaml` に残るので、条件ごとのファイルを
溜める必要がない。

メイン config もこの実験の持ち物なので `scripts/develop/` に置く (1 実験 = 1 ディレクトリ)。
`configs/` に残っているものを使いたいときは `--config configs/akita_soc.yaml` のように
パスで渡す。

> **注意: 複数の run を時間差で起動する最中に config を書き換えないこと。**
> config が読まれるのは起動時の 1 回だけなので、走行中の run は後の編集に影響されない。
> 危険なのは起動の途中で書き換えたときで、後から起動した run だけ条件が変わる。

### seed

`simulation.seed` に**スカラーでも範囲でも**書ける。

```yaml
simulation:
  seed: 5              # seed 5 の run を 1 本
  seed: [1, 10]        # seed 1〜10 の run を 10 本 (両端を含む)
  seed: [1, 10, 2]     # STEP 付き = 1, 3, 5, 7, 9
  parallel: 8          # 同時に走らせる run の数
```

**`[1, 10]` は「1 と 10 の 2 本」ではなく「1 から 10 まで」。**

seed が複数のときは seed ごとに子プロセスを立てて並列に走らせる。各 run の `config.yaml`
には展開後の**スカラーの seed** が入り、範囲指定は `source_config.yaml` (入力の逐語コピー)
に残る。

#### 並列度 (`simulation.parallel`) の決め方

**backend で決まる。**

| backend | N | 推奨 `parallel` | 理由 |
|---|---|---|---|
| `cpu` | 小 (100〜256) | 4〜8 | GeNN の CPU バックエンドは 1 プロセス 1 コア。上限を決めるのはコア数ではなく、seed ごとに走る GeNN のコンパイルが食うメモリ |
| `cuda` | 大 | 1 | 1 run で GPU を埋めるので、並列にする意味がない |
| `cuda` | 小 | 1 | 1 枚の GPU に複数プロセスを載せてもドライバが時分割するだけ。小さい N ではカーネル起動レイテンシ律速なので、むしろ遅くなる |

`backend: cuda` かつ `parallel > 1` のときは警告が出る。seed の本数を超える値を書いても
本数で頭打ちになる。

### 記録条件

記録時刻・窓幅・バッファは `scripts/develop/task.yaml` を書き換える。
**どのプロファイルで走るかはメイン config の `task:` が決める。**

```yaml
# scripts/develop/axon_growth_grid.yaml
task: develop          # task.yaml のプロファイル名
```

基本は `develop` の 1 つを手で書き換えて使う。記録プロトコルとして本質的に別物のとき
(発達の追跡ではなく短時間のスモークなど) だけプロファイルを足し、config 側で選ぶ。

### 膜電位トレース

1 ニューロンの V と Isyn_rec を記録窓の先頭で採るかどうかも `task.yaml` で決める。

```yaml
trace_neuron: null     # null なら採らない (既定)。採るならニューロン ID
trace_window_s: 10.0   # 記録窓の先頭から何秒ぶん
```

採ると記録窓の先頭を 1 ステップずつ進めて毎ステップ `pull()` するので遅くなる。長い run
では効く。そのぶん、同じ窓のラスターと時間軸が揃った V / I が残る。

**引数にしていないのは、採ったかどうかがその run の記録の一部だから。** 引数にすると、
完走した run に `trace_*.npz` が無いときに「採らない設定だった」のか「採ろうとして失敗
した」のかが区別できなくなる。範囲外のニューロン ID は起動直後に弾く (長い run を回し
切ってから `IndexError` で落ちないように)。

---

## 出力

```
outputs/develop/<条件>/            seed が 1 つ  -> ここが run ディレクトリそのもの
outputs/develop/<条件>/seed01/     seed が複数   -> ここが run ディレクトリ
                     /seed02/
```

run ディレクトリの中身:

```
<run>/
├── config.yaml           **build を通った run の記録**。seed も sparse も実値
├── source_config.yaml    入力 YAML の逐語コピー (seed の範囲指定はここに残る)
├── run.log               この run の標準出力・標準エラー
├── data/
│   ├── connectivity.npz      結合構造 (row/col/shape)。run に 1 回だけ
│   ├── layout_axes.npz       外部軸 (layer / module …)
│   ├── axon_geometry.npz     軸索の折れ線 (axon_growth 系の config のみ)
│   ├── metrics.csv           記録時刻ごとの指標。**1 行ずつ追記される**
│   ├── weights_{h}h.npz      重みの値ベクトル (connectivity と index 整合)
│   ├── spikes_{h}h.npz       スパイク (絶対時刻 ms)
│   └── trace_{h}h.npz        膜電位トレース (task.trace_neuron を指定した run のみ)
└── figures/
    ├── structure/        area / connection_mask / network_sample / 各種分布
    ├── raster/
    ├── avalanche/
    ├── trace/            task.trace_neuron を指定した run のみ
    └── overview/         figure2c / figure2d / weight_matrix_*
```

**ディレクトリ名からは何も読み取らないこと。** seed も条件も `config.yaml` の中にある。

`metrics.csv` は記録時刻ごとに 1 行ずつ書かれるので、途中で落ちた run もそこまでの指標が
読める。

### `config.yaml` と `pending_config.yaml`

走っている最中の run には、もう 1 つ `pending_config.yaml` がある。**ランチャから run 本体
への引き継ぎ**で、run 本体が build を通して `config.yaml` を書いた時点で消える。

分けてあるのは役割が違うから。`network.sparse` の実値 (`on` / `off`) を決めるのは
NetworkBuilder の生成時なので、ランチャが起動前に書ける config は `auto` のまま —— それを
`config.yaml` の名前で置くと「この run はこうして走った」という記録の意味が崩れる。

おかげで **`config.yaml` があれば build を通った run**、という読み方ができる。

| run ディレクトリの状態 | 意味 |
|---|---|
| `pending_config.yaml` だけ | まだ走っていない / build の前に落ちた |
| `config.yaml` だけ | build を通った |

---

## 失敗した run だけやり直す

run ディレクトリを指して子プロセスを直接叩く。config を書き換える必要はない。

```bash
python -m scripts.develop.run_one outputs/develop/gmax25_baseline/seed03
```

ランチャは失敗した run を一覧で出すので、そのパスをそのまま渡せばよい。
`pending_config.yaml` が残っていればそれを、完走した run をもう一度走らせるなら
`config.yaml` を読む。

---

## 図と指標を作り直す

```bash
python -m scripts.develop.replot <run ディレクトリ> [--smax N]
```

GeNN は使わない。`spikes_*h.npz` と `weights_*h.npz` から指標を計算し直し、図を描き直す。
本番と**同じ関数** (`metrics.build_row`) を通るので、列も値も一致する。

書き出し先は本番と同じ `metrics.csv` で、**上書きする**。同じ関数が同じ列を作るので別名に
逃がす理由がない (逃がしておく方がむしろ危険で、図がどちらを読むかを明示しないと例外を
出さずに古い値が描かれる)。

`connectivity.npz` を持たない古い run (密形式で保存された run) は**エラーで止まる**。列を
減らして続行はしない —— 列の欠けた `metrics.csv` で上書きすると、あとで図と突き合わせた
ときに原因が追えなくなる。

`organize_output` 時代の古い run (npz が `data/` にあるものも run 直下にあるものも) は
そのまま渡せる。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `__main__.py` | ランチャ (親)。seed を展開し、run ディレクトリを作り、子を並列起動する |
| `run_one.py` | 1 run を走らせる (子)。引数は run ディレクトリ 1 つ |
| `replot.py` | 再解析 |
| `<名前>.yaml` | メイン config (ネットワーク・seed・並列度・使う task プロファイル) |
| `task.yaml` | 記録プロトコル (記録時刻・窓幅・バッファ・トレース) |
| `paths.py` | run ディレクトリの内部構造。`data/` と `figures/` と引き継ぎファイル名を知るのはここだけ |
| `records.py` | 記録ファイルの名前の規約 (`weights_{h}h.npz` など) と読み書き。`metrics.csv` の逐次書き出しもここ |
| `metrics.py` | 1 記録窓の指標を 1 行の dict にする。**本番も再解析もここを通る** |
| `panels.py` | 記録窓ごとの図 (ラスター / アバランチ / トレース) |
| `fig2c.py` `fig2d.py` `weight_track.py` | run 全体をまとめた図 |

**依存の向きは `scripts/develop/ → src/utils/{analysis,plotting} → src/core` の一方向だけ。**
他の実験 (lesion など) を import しない。共有したいものが出てきたら、実験どうしで import
し合うのではなく `src/utils/` へ上げる。

テストは `test/experiments/develop/test_develop.py`。
