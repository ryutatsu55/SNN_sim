# akita_soc: 本番と再解析が一致することを確かめる

`scripts/akita_soc/figures/` は実験固有の複製なので、**共有層のテストは akita_soc の図を
守りません。** 代わりに確かめられるのは「**同じ run から 2 通りの経路 (本番 / `replot`) で
同じ png が出ること**」で、これは run の中で完結するので基準ファイルを持ちません。

> **図の md5 基準 (`reference_figures.md5`) は廃止した。** 理由は
> `test/experiments/develop/README.md` と同じ。

akita_soc は `area: no_space` なので、空間の図 (area / network_sample / axon_network /
connection_probability / distance_distribution) は最初から出ません。

## 手順

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

`scripts/akita_soc/akita_soc.yaml` を `_smoke.yaml` として複製し、`task: smoke` に変えて
走らせます。

```bash
python -m scripts.akita_soc --config _smoke --condition _ref
find outputs/akita_soc/_ref -name "*.png" | sort | xargs md5sum > /tmp/akita_before.md5

python -m scripts.akita_soc.replot outputs/akita_soc/_ref
find outputs/akita_soc/_ref -name "*.png" | sort | xargs md5sum | diff - /tmp/akita_before.md5
```

`metrics.csv` も本番と再解析で同じ関数を通るので、バイト単位で一致します。

## 後片付け

終わったら `task.yaml` の `smoke` プロファイル・`_smoke.yaml`・`outputs/akita_soc/_ref`・
`genn_code/akita_soc__ref_*` を消してください。
