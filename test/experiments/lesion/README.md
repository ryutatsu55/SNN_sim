# lesion の図の回帰チェック

`scripts/lesion/figures/` は実験固有の複製なので、**共有層のテスト
(`test/utils/test_plotting_ordering.py` など) は lesion の図を守りません。**
代わりに「同じ run から同じ png が出ること」をバイト単位で確かめます。

`reference_figures.md5` がその基準。develop / akita_soc と同じ仕組みですが、
**親 run が要る**のがこの実験だけの事情です。

## 作り直し方

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
  baseline_window_ms: 5000.0
  recovery_hours: 0.01
  probe_interval_hours: 0.005
  record_window_ms: 5000.0
  record_buffer_ms: 1000.0
  hub_z: 2.5
  include_betweenness: true
  include_clustering: true
  duration: 120000.0
```

```bash
python -m scripts.lesion --parent outputs/develop/_parent --condition _ref --task smoke
find outputs/lesion/_ref -name "*.png" | sort | xargs md5sum \
  | sed 's|/_ref/|/RUN/|' > test/experiments/lesion/reference_figures.md5
```

比較するときは条件名を `_ref` にすること。

### 3. 本番と再解析が一致することも見る

`replot` は構造図のために**再ビルドして切り直す**ので、同じ run に対して 2 通りの経路で
図が作られることになります。切断マスクは `lesion_cut.npz` の (row, col) から引き当てる
ので、一致すれば**再ビルドが元の run と同じネットワークを作り、同じシナプスを切った**
証拠になります。

```bash
python -m scripts.lesion.replot outputs/lesion/_ref
find outputs/lesion/_ref -name "*.png" | sort | xargs md5sum | sed 's|/_ref/|/RUN/|' \
  | diff - test/experiments/lesion/reference_figures.md5
```

## 後片付け

終わったら `task.yaml` の `smoke` プロファイル・`outputs/lesion/_ref`・
`outputs/develop/_parent`・`genn_code/lesion_*` を消してください。

## 時間がかかる

**GeNN のコンパイルが run につき 2 回**走り、そのうえ `_generate_global_matrices()` が
3 回 (切断対象の決定 / Phase 1 / Phase 2) 走ります。`axon_growth` 系の config では
ここが支配的なので、基準の作り直しは数十分見ておくこと。
