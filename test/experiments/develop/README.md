# develop の図の回帰チェック

`scripts/develop/figures/` は `src/utils/plotting/` の複製なので、**共有層のテスト
(`test/utils/test_plotting_ordering.py` など) は develop の図を守りません。**
代わりに「同じ run から同じ png が出ること」をバイト単位で確かめます。

`reference_figures.md5` がその基準 (21 枚)。**リファクタで図が意図せず変わったかどうかは
これで分かります** —— 図の内容を変える変更をしたときだけ、意図を確認したうえで更新して
ください。

## 作り直し方

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
find outputs/develop/_ref -name "*.png" | sort | xargs md5sum \
  | sed 's|/_ref/|/RUN/|' > test/experiments/develop/reference_figures.md5
```

比較するときは条件名を `_ref` にすること。`weight_matrix_panel.png` と
`weight_delta_panel.png` は**タイトルに run ディレクトリ名が入る**ので、別名で走らせると
この 2 枚だけ必ず食い違います。

## 本番と再解析が一致することも見る

`replot` は構造図のために**再ビルド**するので、同じ run に対して 2 通りの経路で
21 枚が作られることになります。作り直しても md5 が動かないことを確かめてください。

```bash
python -m scripts.develop.replot outputs/develop/_ref
find outputs/develop/_ref -name "*.png" | sort | xargs md5sum | sed 's|/_ref/|/RUN/|' \
  | diff - test/experiments/develop/reference_figures.md5
```

構造図まで一致すれば、**再ビルドが元の run と同じネットワークを作った**証拠になります
(`connectivity.npz` との照合は `replot` 自身も毎回やりますが、図の一致はより強い検査です)。

`metrics.csv` も本番と再解析で同じ関数を通るので、バイト単位で一致します。

## 後片付け

終わったら `task.yaml` の `smoke` プロファイル・`_smoke.yaml`・`outputs/develop/_ref`・
`genn_code/develop__ref_*` を消してください。
