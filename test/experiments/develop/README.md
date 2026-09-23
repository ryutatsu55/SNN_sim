# develop: 本番と再解析が一致することを確かめる

`scripts/develop/figures/` は共有層の複製なので、**`test/utils/` のテストは develop の図を
守りません。** 代わりに確かめられるのは「**同じ run から 2 通りの経路 (本番 / `replot`) で
同じ png が出ること**」で、これは run の中で完結するので基準ファイルを持ちません。

> **図の md5 基準 (`reference_figures.md5`) は廃止した。** 指標や図の定義を直すたびに
> 基準の方も作り直すことになり、「意図して変えた」と「壊れた」を区別できなかったため。
> 残すのは下の自己一致チェックだけ。

**構造図のために再ビルドするので、一致すれば「再ビルドが元の run と同じ結合を作った」
証拠**にもなる (`connectivity.npz` との照合は `replot` 自身も毎回やるが、図の一致はより
強い検査)。

## 手順

`scripts/develop/task.yaml` の末尾に一時的なプロファイルを足します。

```yaml
smoke:
  duration: 20000.0
  record_hours: [0, 0.0013888888888888889]   # 0 h と 5 s。**非整数を含めるのが要点**
  record_window_ms: 5000.0
  record_buffer_ms: 1000.0
  trace_neuron: 0                            # トレース図も出す
  trace_window_s: 1.0
```

`scripts/develop/axon_growth_hierarchy.yaml` を `_smoke.yaml` として複製し、`task: smoke` /
`parallel: 1` に変えて走らせます (空間構造を持つ config なので、**構造図がフルセットで
出ます**)。

```bash
python -m scripts.develop --config _smoke --condition _ref
find outputs/develop/_ref -name "*.png" | sort | xargs md5sum > /tmp/develop_before.md5

python -m scripts.develop.replot outputs/develop/_ref
find outputs/develop/_ref -name "*.png" | sort | xargs md5sum | diff - /tmp/develop_before.md5
```

`metrics.csv` も本番と再解析で同じ関数を通るので、バイト単位で一致します。

## 後片付け

終わったら `task.yaml` の `smoke` プロファイル・`_smoke.yaml`・`outputs/develop/_ref`・
`genn_code/develop__ref_*` を消してください。
