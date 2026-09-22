# akita_soc の図の回帰チェック

`scripts/akita_soc/figures/` は実験固有の複製なので、**共有層のテスト
(`test/utils/test_plotting_ordering.py` など) は akita_soc の図を守りません。**
代わりに「同じ run から同じ png が出ること」をバイト単位で確かめます。

`reference_figures.md5` がその基準 (15 枚)。`scripts/develop/` と同じ仕組みで、
違うのは figure の顔ぶれだけ —— akita_soc は `area: no_space` なので空間の図
(area / network_sample / axon_network / connection_probability / distance_distribution)
を持ちません。

## 作り直し方

`scripts/akita_soc/task.yaml` の末尾に一時的なプロファイルを足します。

```yaml
smoke:
  duration: 20000.0
  record_hours: [0, 0.0013888888888888889]   # 0 h と 5 s。**非整数を含めるのが要点**
  record_window_ms: 5000.0
  record_buffer_ms: 1000.0
  trace_neuron: 0                            # トレース図も出す
  trace_window_s: 1.0
```

`scripts/akita_soc/akita_soc.yaml` を `_smoke.yaml` として複製し、`task: smoke` に
変えて走らせます。

```bash
python -m scripts.akita_soc --config _smoke --condition _ref
find outputs/akita_soc/_ref -name "*.png" | sort | xargs md5sum \
  | sed 's|/_ref/|/RUN/|' > test/experiments/akita_soc/reference_figures.md5
```

比較するときは条件名を `_ref` にすること。`weight_matrix_panel.png` と
`weight_delta_panel.png` は**タイトルに run ディレクトリ名が入る**ので、別名で走らせると
この 2 枚だけ必ず食い違います。

## 本番と再解析が一致することも見る

```bash
python -m scripts.akita_soc.replot outputs/akita_soc/_ref
find outputs/akita_soc/_ref -name "*.png" | sort | xargs md5sum | sed 's|/_ref/|/RUN/|' \
  | diff - test/experiments/akita_soc/reference_figures.md5
```

`metrics.csv` も本番と再解析で同じ関数を通るので、バイト単位で一致します。

## 後片付け

終わったら `task.yaml` の `smoke` プロファイル・`_smoke.yaml`・`outputs/akita_soc/_ref`・
`genn_code/akita_soc__ref_*` を消してください。
