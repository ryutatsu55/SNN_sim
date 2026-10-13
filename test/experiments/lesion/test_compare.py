"""run をまたぐ比較 (`scripts.lesion.compare`) の要約の定義と登録簿の形。

要約は**合成した metrics と lesion.json** で検算する (GeNN 不要)。view は
duck-typing の fake —— `build_row` が `Series` の型ではなく、`manifest()` / `metrics()` /
`config` / `run_dir` だけに依存していることの確認でもある。
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.lesion.analysis.summary import (
    LATE_HOURS, build_control_row, build_row, kind_of)


def _config(seed=3, backend="cpu"):
    return SimpleNamespace(simulation=SimpleNamespace(seed=seed, backend=backend))


def _lesion(rates, chis, *, specs=("random:fraction=0.1",), n_before=200, n_cut=20,
            weight_lost=0.08, hours=None):
    """baseline 1 行 + post (0..20 h) の fake 損傷 run。rates/chis の先頭が baseline。"""
    hours = list(range(21)) if hours is None else hours
    metrics = pd.DataFrame({
        "phase": ["baseline"] + ["post"] * len(hours),
        "hours_since_cut": [-0.17] + [float(h) for h in hours],
        "mean_rate_hz": rates,
        "chi_all": chis,
    })
    manifest = {"cut_specs": list(specs), "num_synapses_before": n_before,
                "num_synapses_cut": n_cut, "restored_from_hour": 20.0,
                "cut_profile": {"weight_lost_fraction": weight_lost}}
    return SimpleNamespace(run_dir=Path("fake"), config=_config(),
                           manifest=lambda: manifest, metrics=lambda: metrics)


@pytest.mark.parametrize("specs, kind", [
    (["bridge:parts=BX1-3"], "bridge"),
    (["random:fraction=0.05"], "random"),
    (["random:pre=exc,fraction=0.05"], "random_exc"),
    (["random:fraction=0.05,pre=exc"], "random_exc"),
    (["random:pre=inh,fraction=0.05"], None),
    (["random:post=exc,fraction=0.05"], None),
    (["bridge:parts=BX1-3", "random:fraction=0.05"], None),
])
def test_kind_is_decided_by_the_spec(specs, kind):
    assert kind_of(specs) == kind


def test_lesion_row_follows_the_definitions():
    # baseline 2.0、0 h で 1.5 に落ち、18–20 h は 1.6 / 1.7 / 1.8 (平均 1.7)
    post = [1.5] + [1.55] * 17 + [1.6, 1.7, 1.8]
    chis = [0.4] + [0.3] * 21
    row = build_row(_lesion([2.0] + post, chis))
    assert row["kind"] == "random"
    assert row["cut_pct"] == pytest.approx(10.0)
    assert row["cut_weight_pct"] == pytest.approx(8.0)
    assert row["drop"] == pytest.approx(0.25)
    assert row["chi_drop"] == pytest.approx(0.25)
    assert row["rec"] == pytest.approx((1.7 - 1.5) / 2.0)
    assert (row["late_from_h"], row["late_to_h"]) == LATE_HOURS


def test_unfinished_run_is_rejected():
    """回復の区間まで記録が届いていない run は要約しない (途中で止まった run)。"""
    hours = list(range(11))
    with pytest.raises(ValueError, match="揃っていません"):
        build_row(_lesion([2.0] + [1.5] * 11, [0.4] * 12, hours=hours))


def test_control_uses_the_same_formula_from_the_cut_hour():
    hours = [float(h) for h in range(41)]
    rates = [1.0] * 20 + [2.0] + [2.0] * 17 + [2.2, 2.2, 2.2]
    control = SimpleNamespace(run_dir=Path("ctrl"), config=_config(),
                              metrics=pd.DataFrame({"hour": hours, "mean_rate_hz": rates}))
    row = build_control_row(control, cut_hour=20.0)
    assert row["rate_0"] == pytest.approx(2.0)
    assert row["rec"] == pytest.approx(0.1)


def test_compare_registry_is_a_table():
    """登録簿の形 (test_report_registries.py と同じ約束を compare 段階にも)。"""
    from scripts.lesion.report import compare
    assert isinstance(compare.FIGURES, tuple) and compare.FIGURES
    labels = [row[0] for row in compare.FIGURES]
    names = [row[2] for row in compare.FIGURES]
    assert len(set(labels)) == len(labels) and len(set(names)) == len(names)
    assert all(callable(row[1]) and row[2].endswith(".png") for row in compare.FIGURES)
    imports = [line for line in Path(compare.__file__).read_text(encoding="utf-8").splitlines()
               if line.startswith(("import ", "from "))]
    assert not any("matplotlib" in line or "numpy" in line for line in imports)
