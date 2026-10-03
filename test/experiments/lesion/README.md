# lesion: 本番と再解析が一致することを確かめる

`scripts/lesion/figures/` は実験固有の複製なので、**共有層のテスト
(`test/utils/test_plotting_ordering.py` など) は lesion の図を守りません。**
代わりに確かめられるのは「**同じ run から 2 通りの経路 (本番 / `replot`) で同じ png が
出ること**」で、これは run の中で完結するので基準ファイルを持ちません。

> **図の md5 基準 (`reference_figures.md5`) は廃止した。** 理由は
> `test/experiments/develop/README.md` と同じ。

develop / akita_soc と同じ手順ですが、**親 run が要る**のがこの実験だけの事情です。

## 手順

### 1. 親 run を作る

`scripts/develop/` の図の基準と同じ手順で短い run を 1 本作ります
(`test/experiments/develop/README.md` の `smoke` プロファイル)。条件名は `_parent`。

```bash
python -m scripts.develop --config _smoke --condition _parent
```

**空間構造を持つ config で作ること。** `bridge:` の切断 spec はブリッジを持つエリア
(`modular_grid` 系) でないと 0 本になります。

### 2. 損傷プロトコルを足して走らせる

`scripts/lesion/task.yaml` の末尾に一時的なプロファイルを足します。

```yaml
smoke:
  from_hour: null
  cut: ["bridge:kind=inter_cluster"]
  cut_combine: or
  settle_ms: 2000.0
  record_hours: [0, 0.005, 0.01]
  record_window_ms: 5000.0
  record_buffer_ms: 1000.0
  hub_z: 2.5
  include_betweenness: true
  include_clustering: true
```

**出力は親 run の下**です (`--condition` はありません。ディレクトリ名は切断 spec から
決まる)。親を 1 本だけ作ったので、run ディレクトリは
`outputs/develop/_parent/bridge-inter_cluster/` になります。

```bash
python -m scripts.lesion --parent outputs/develop/_parent --task smoke
RUN=outputs/develop/_parent/bridge-inter_cluster
find "$RUN" -name "*.png" | sort | xargs md5sum > /tmp/lesion_before.md5
```

### 3. 再解析して突き合わせる

`replot` は構造図も本番と同じく切断直後の記録から描くので、全部の png が一致するはずです。
`--verify` を付けると、親と同じネットワークを再ビルドして記録どおりに切り、記録と同じ
結合と軸索幾何が出ることも併せて確かめます (切断マスクは `lesion_cut.npz` の (row, col)
から引き当てる)。

```bash
python -m scripts.lesion.replot "$RUN" --verify
find "$RUN" -name "*.png" | sort | xargs md5sum | diff - /tmp/lesion_before.md5
```

## 後片付け

終わったら `task.yaml` の `smoke` プロファイル・`outputs/develop/_parent`
(損傷 run はこの下にあるので一緒に消える)・`genn_code/lesion_*` を消してください。

## 時間がかかる

**GeNN のコンパイルが run につき 1 回**走り、そのうえ `_generate_global_matrices()` が
2 回 (切断対象の決定 / 本番の build) 走ります。`axon_growth` 系の config では
ここが支配的なので、このチェックは十数分見ておくこと。
