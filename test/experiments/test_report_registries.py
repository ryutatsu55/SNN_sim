"""出力の登録簿 (`scripts/<実験>/report/`) が持つべき形を、全実験まとめて固定する。

**実験をまたぐ不変条件なので、実験ごとのディレクトリではなくここに置く。** 3 つの実験が
同じ形をしていること自体がこのプロジェクトの決めごと (ルート `CLAUDE.md` の
「実験は `scripts/<実験名>/` にまとめる」) で、1 つだけ形が崩れても他の 2 つのテストでは
気づけない。

見るのは 2 つ:

1. **登録簿が表であること。** 報告 (図の `FIGURES` と表の `TABLES`) が `emit()` の外にあり、
   上から順に回されるだけであること。ここが崩れると「何がいつ出るか」を読むのに
   関数本体を追うことになる。
2. **表の中身が壊れていないこと。** 名前もファイル名も重複せず、報告の種類は
   `store/paths.py` が知っているものだけ。図は png、表は csv。
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


REGISTRIES = ("FIGURES", "TABLES")


def _rows(experiment: str, stage: str) -> tuple:
    """その段階の報告の行 (図と表)。"""
    module = _report(experiment, stage)
    return tuple(module.FIGURES) + tuple(module.TABLES)


def _declared(experiment: str) -> list[tuple[str, str]]:
    """登録簿が出すと言っている `(報告の種類, ファイル名 or テンプレート)` の全体。"""
    paths = _paths(experiment)
    declared = [(paths.STRUCTURE, name)
                for _label, _make, name in _rows(experiment, "structure")]
    declared += [(kind, template)
                 for _label, _make, kind, template in _rows(experiment, "panels")]
    overview = _report(experiment, "overview")
    declared += [(paths.OVERVIEW, name) for _label, _make, name in _rows(experiment, "overview")]
    declared += [(paths.OVERVIEW, template)
                 for _label, _make, template in getattr(overview, "PER_WINDOW_FIGURES", ())]
    return declared


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_stage_declares_a_table(experiment):
    """3 つの段階すべてが `FIGURES` と `TABLES` を持つこと。**一覧が関数の中に隠れない。**

    `TABLES` は空でもよい (その段階に表が無いことを表に書いておく)。
    """
    for stage in ("structure", "panels", "overview"):
        for name in REGISTRIES:
            table = getattr(_report(experiment, stage), name, None)
            assert table is not None, f"{experiment}/{stage} に {name} がありません"
            assert isinstance(table, tuple), f"{experiment}/{stage} の {name} は tuple にすること"
        assert _report(experiment, stage).FIGURES, f"{experiment}/{stage} の FIGURES が空です"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_rows_are_callable_with_a_label_and_a_name(experiment):
    """各行が `(名前, 呼べるもの, ファイル名…)` であること。図は png、表は csv。"""
    for registry, suffix in (("FIGURES", ".png"), ("TABLES", ".csv")):
        for stage in ("structure", "panels", "overview"):
            for row in getattr(_report(experiment, stage), registry):
                label, make, name = row[0], row[1], row[-1]
                assert isinstance(label, str) and label, f"{experiment}: 名前が空です ({row})"
                assert callable(make), f"{experiment}: {label} の関数が呼べません"
                assert name.endswith(suffix), \
                    f"{experiment}: {label} の出力が {suffix} ではありません"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_labels_and_filenames_do_not_collide(experiment):
    """名前もファイル名も重複しないこと。

    名前が重なるとログの `skip X` / `Warning: X` がどちらの図の話か分からなくなる。
    ファイル名が重なると片方が黙って上書きされる。
    """
    labels, declared = [], _declared(experiment)
    for stage in ("structure", "panels", "overview"):
        module = _report(experiment, stage)
        labels += [row[0] for row in _rows(experiment, stage)]
        labels += [row[0] for row in getattr(module, "PER_WINDOW_FIGURES", ())]
    assert len(labels) == len(set(labels)), f"{experiment}: 名前が重複しています"
    assert len(declared) == len(set(declared)), f"{experiment}: 出力先が重複しています"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_figure_kinds_are_known_to_paths(experiment):
    """使っている報告の種類が `store/paths.py` に登録されていること。

    未登録の種類は `report_path()` が弾くが、それは**その報告を出す直前**。表の時点で
    捕まえられれば、run を回す前に分かる。
    """
    known = set(_paths(experiment).REPORT_KINDS)
    for kind, _name in _declared(experiment):
        assert kind in known, f"{experiment}: 未知の報告の種類 {kind!r}"


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
