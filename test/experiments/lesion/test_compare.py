"""run をまたぐ比較 (`scripts.lesion.compare`) の要約の定義、迂回経路の集計、登録簿の形。

要約は**合成した metrics と lesion.json** で検算する (GeNN 不要)。view は
duck-typing の fake —— `build_row` が `Series` の型ではなく、`manifest()` / `metrics()` /
`config` / `run_dir` だけに依存していることの確認でもある。
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.lesion.analysis.summary import (
    LATE_HOURS, build_control_row, build_row, kind_of, write_summary)
from scripts.lesion.analysis.detours import condition_of, path_summary


def _config(seed=3, backend="cpu"):
    return SimpleNamespace(simulation=SimpleNamespace(seed=seed, backend=backend))


def _lesion(rates, chis, corrs=None, *, specs=("random:fraction=0.1",), n_before=200, n_cut=20,
            weight_lost=0.08, hours=None, frozen=None, run_dir="fake"):
    """baseline 1 行 + post (0..20 h) の fake 損傷 run。rates/chis の先頭が baseline。"""
    hours = list(range(21)) if hours is None else hours
    metrics = pd.DataFrame({
        "phase": ["baseline"] + ["post"] * len(hours),
        "hours_since_cut": [-0.17] + [float(h) for h in hours],
        "mean_rate_hz": rates,
        "chi_all": chis,
        "mean_correlation": chis if corrs is None else corrs,
    })
    manifest = {"cut_specs": list(specs), "num_synapses_before": n_before,
                "num_synapses_cut": n_cut, "restored_from_hour": 20.0,
                "cut_profile": {"weight_lost_fraction": weight_lost}}
    if frozen is not None:
        manifest["frozen_plasticity"] = list(frozen)
    return SimpleNamespace(run_dir=Path(run_dir), config=_config(),
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
    corrs = [0.08] + [0.06] * 21
    row = build_row(_lesion([2.0] + post, chis, corrs))
    assert row["kind"] == "random"
    assert row["cut_pct"] == pytest.approx(10.0)
    assert row["cut_weight_pct"] == pytest.approx(8.0)
    assert row["drop"] == pytest.approx(0.25)
    assert row["chi_drop"] == pytest.approx(0.25)
    assert row["corr_drop"] == pytest.approx(0.25)
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
    assert isinstance(compare.TABLES, tuple)
    rows = compare.FIGURES + compare.TABLES
    labels = [row[0] for row in rows]
    names = [row[-1] for row in rows]
    assert len(set(labels)) == len(labels) and len(set(names)) == len(names)
    assert all(callable(row[1]) and row[2].endswith(".png") for row in compare.FIGURES)
    assert all(callable(row[1]) and callable(row[2]) and row[3].endswith(".csv")
               for row in compare.TABLES)
    assert {row[3] for row in compare.TABLES} == {
        "detour_paths.csv", "detour_reverse.csv", "detour_last_hop.csv", "detour_turnover.csv",
        "flip_rule.csv", "flip_rule_classes.csv", "freeze_split.csv"}
    imports = [line for line in Path(compare.__file__).read_text(encoding="utf-8").splitlines()
               if line.startswith(("import ", "from "))]
    assert not any("matplotlib" in line or "numpy" in line for line in imports)


def test_detour_condition_comes_from_the_manifest():
    """条件の列は lesion.json から作る。`frozen_plasticity` を持たない古い run は止めていない扱い。"""
    assert condition_of({"cut_specs": ["random:fraction=0.1"]}) == \
        {"spec": "random:fraction=0.1", "frozen": ""}
    assert condition_of({"cut_specs": ["a", "b"], "frozen_plasticity": ["from_Inh", "from_Exc"]}) == \
        {"spec": "a;b", "frozen": "from_Exc-from_Inh"}


def test_path_summary_counts_targets_without_paths():
    """経路の本数の平均は、迂回経路を 1 本も持たない対象も分母に入れる。"""
    targets = pd.DataFrame(dict(
        spec="x", frozen="", seed=1, target=[0, 1], kind="cut", cls="strong",
        fwd_paths=[2, 0], rev_paths=[0, 0],
        fwd_strong_L2=[1, 0], fwd_strong_L3=[0, 0],
        rev_strong_L2=[0, 0], rev_strong_L3=[0, 0]))
    paths = pd.DataFrame(dict(
        spec="x", frozen="", seed=1, target=[0, 0], dir="fwd", length=[2, 3],
        bn_pre=[0.9, 0.2], bn_p0=[0.9, 0.2], bn_end=[0.9, 0.8],
        flip_first=[0, 0], flip_middle=[0, 0], flip_last=[0, 1]))
    row = path_summary(targets, paths).query("dir == 'fwd'").iloc[0]
    assert row.paths_per_target == pytest.approx(1.0)       # (2 + 0) / 2
    assert row.strong_per_target == pytest.approx(0.5)
    assert row.no_strong_fraction == pytest.approx(0.5)
    assert row.up_fraction == pytest.approx(0.5)
    assert row.flip_last == pytest.approx(1.0)


def test_path_summary_keeps_conditions_apart():
    """同じ spec でも止めた可塑性が違えば別の条件 (freeze の対を混ぜない)。"""
    def frame(**columns):
        return pd.DataFrame(dict(spec="s", seed=1, **columns))
    targets = pd.concat([frame(frozen=f, target=[0], kind="cut", cls="strong", fwd_paths=[n],
                               rev_paths=[0], fwd_strong_L2=[0], fwd_strong_L3=[0],
                               rev_strong_L2=[0], rev_strong_L3=[0])
                         for f, n in (("", 1), ("from_Inh", 0))], ignore_index=True)
    paths = frame(frozen="", target=[0], dir="fwd", length=[2], bn_pre=[0.2], bn_p0=[0.2],
                  bn_end=[0.2], flip_first=[0], flip_middle=[0], flip_last=[0])
    out = path_summary(targets, paths).query("dir == 'fwd'").set_index("frozen")
    assert out.loc["", "n_paths"] == 1 and out.loc["from_Inh", "n_paths"] == 0


def test_runs_with_frozen_plasticity_are_left_out(tmp_path):
    """切断後に可塑性を止めた run は、同じ spec でも用量反応の表に入れない。"""
    rates, chis = [1.0] + [0.8] * 21, [1.0] + [0.9] * 21
    runs = (_lesion(rates, chis, run_dir="plain"),
            _lesion(rates, chis, run_dir="no_freeze_key", frozen=[]),
            _lesion(rates, chis, run_dir="frozen", frozen=["from_Inh"]))
    collection = SimpleNamespace(cond_dir=tmp_path, lesion_runs=lambda: runs,
                                 control_runs=lambda: ())
    write_summary(collection, tmp_path)
    assert sorted(pd.read_csv(tmp_path / "runs.csv").run) == ["no_freeze_key", "plain"]


def _freeze_runs():
    """seed 1 は 3 条件がそろい、seed 2 は i-STDP を止めた run が欠けた合成の表。"""
    return pd.DataFrame({
        "spec": ["random:fraction=0.1"] * 5,
        "seed": [1, 1, 1, 2, 2],
        "rule": ["none", "e-STDP", "i-STDP", "none", "e-STDP"],
        "rec": [0.08, 0.01, 0.06, 0.05, 0.00],
    })


def test_freeze_split_is_the_recovery_with_the_other_rule_frozen():
    from scripts.lesion.analysis.freeze_split import split_table
    out = split_table(_freeze_runs())
    assert list(out.seed) == [1]
    row = out.iloc[0]
    assert row.delta_e == pytest.approx(0.06) and row.delta_i == pytest.approx(0.01)
    assert row.delta_sum == pytest.approx(0.07)


def test_freeze_rule_comes_from_the_plasticity_mode():
    from scripts.lesion.analysis.freeze_split import rule_of
    config = SimpleNamespace(synapses={
        "from_Exc": SimpleNamespace(plasticity=SimpleNamespace(mode="e-stdp_physical_fine")),
        "from_Inh": SimpleNamespace(plasticity=SimpleNamespace(mode="i-stdp_physical_fine"))})
    assert rule_of(config, []) == "none"
    assert rule_of(config, ["from_Exc"]) == "e-STDP"
    assert rule_of(config, ["from_Inh"]) == "i-STDP"
    with pytest.raises(ValueError):
        rule_of(config, ["from_Exc", "from_Inh"])
