"""出力の登録簿 (`scripts/<実験>/report/`) が持つべき形を、全実験まとめて固定する。

**実験をまたぐ不変条件なので、実験ごとのディレクトリではなくここに置く。** 3 つの実験が
同じ形をしていること自体がこのプロジェクトの決めごと (ルート `CLAUDE.md` の
「実験は `scripts/<実験名>/` にまとめる」) で、1 つだけ形が崩れても他の 2 つのテストでは
気づけない。

見るのは 2 つ:

1. **登録簿が表であること。** `FIGURES` が `emit()` の外にあり、上から順に回されるだけ
   であること。ここが崩れると「何がいつ出るか」を読むのに関数本体を追うことになる。
2. **表の中身が壊れていないこと。** 名前もファイル名も重複せず、図の種類は
   `store/paths.py` が知っているものだけ。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

EXPERIMENTS = ("develop", "akita_soc", "lesion")


def _report(experiment: str, stage: str):
    return importlib.import_module(f"scripts.{experiment}.report.{stage}")


def _paths(experiment: str):
    return importlib.import_module(f"scripts.{experiment}.store.paths")


def _declared(experiment: str) -> set[tuple[str, str]]:
    """登録簿が出すと言っている `(図の種類, ファイル名 or テンプレート)` の全体。"""
    paths = _paths(experiment)
    declared = {(paths.STRUCTURE, name)
                for _label, _draw, name in _report(experiment, "structure").FIGURES}
    declared |= {(kind, template)
                 for _label, _draw, kind, template in _report(experiment, "panels").FIGURES}
    overview = _report(experiment, "overview")
    declared |= {(paths.OVERVIEW, name) for _label, _draw, name in overview.FIGURES}
    declared |= {(paths.OVERVIEW, template)
                 for _label, _draw, template in getattr(overview, "PER_WINDOW_FIGURES", ())}
    return declared


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_stage_declares_a_table(experiment):
    """3 つの段階すべてが `FIGURES` を持つこと。**一覧が関数の中に隠れない。**"""
    for stage in ("structure", "panels", "overview"):
        table = getattr(_report(experiment, stage), "FIGURES", None)
        assert table is not None, f"{experiment}/{stage} に FIGURES がありません"
        assert isinstance(table, tuple), f"{experiment}/{stage} の FIGURES は tuple にすること"
        assert table, f"{experiment}/{stage} の FIGURES が空です"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_rows_are_callable_with_a_label_and_a_name(experiment):
    """各行が `(名前, 呼べるもの, ファイル名…)` であること。"""
    rows = []
    rows += [(*row, "structure") for row in _report(experiment, "structure").FIGURES]
    rows += [(*row, "panels") for row in _report(experiment, "panels").FIGURES]
    rows += [(*row, "overview") for row in _report(experiment, "overview").FIGURES]
    for row in rows:
        label, draw, name = row[0], row[1], row[-2]
        assert isinstance(label, str) and label, f"{experiment}: 名前が空です ({row})"
        assert callable(draw), f"{experiment}: {label} の描画関数が呼べません"
        assert name.endswith(".png"), f"{experiment}: {label} の出力が png ではありません"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_labels_and_filenames_do_not_collide(experiment):
    """名前もファイル名も重複しないこと。

    名前が重なるとログの `skip X` / `Warning: X` がどちらの図の話か分からなくなる。
    ファイル名が重なると片方が黙って上書きされる。
    """
    labels, declared = [], list(_declared(experiment))
    for stage in ("structure", "panels", "overview"):
        module = _report(experiment, stage)
        labels += [row[0] for row in module.FIGURES]
        labels += [row[0] for row in getattr(module, "PER_WINDOW_FIGURES", ())]
    assert len(labels) == len(set(labels)), f"{experiment}: 名前が重複しています"
    assert len(declared) == len(set(declared)), f"{experiment}: 出力先が重複しています"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_figure_kinds_are_known_to_paths(experiment):
    """使っている図の種類が `store/paths.py` に登録されていること。

    未登録の種類は `fig_path()` が弾くが、それは**その図を描く直前**。表の時点で
    捕まえられれば、run を回す前に分かる。
    """
    known = set(_paths(experiment).FIG_KINDS)
    for kind, _name in _declared(experiment):
        assert kind in known, f"{experiment}: 未知の図の種類 {kind!r}"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_report_stays_a_registry(experiment):
    """**`report/` は matplotlib も numpy も import しない。**

    橋渡ししかしないので、計算も描画も入り込めない。破れていたら、その処理は
    `analysis/` か `figures/` へ行くべきもの。
    """
    for stage in ("__init__", "structure", "panels", "overview"):
        source = Path(_report(experiment, stage).__file__).read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.lstrip().startswith("#"))
        # docstring は除けないので、import 文だけを見る
        imports = [line for line in code.splitlines()
                   if line.startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "matplotlib" not in joined, f"{experiment}/{stage} が matplotlib を import"
        assert "numpy" not in joined, f"{experiment}/{stage} が numpy を import"
