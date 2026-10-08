"""PQN診断の純粋処理と保存データ解析の確認。"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from src.utils.experiments.pqn_diagnostics import (
    DiagnosticsSettings,
    _cell_comparison,
    _network_comparison,
    analyze_run,
    cell_rows,
    expand_cases,
    threshold_crossings,
)


def test_standard_plan_has_22_cases_and_smallest_dt_extended():
    settings = DiagnosticsSettings()
    cases = expand_cases(settings)
    assert len(cases) == 22
    assert len({case["case_id"] for case in cases}) == 22
    assert cases[-1]["dt_ms"] == min(settings.dt_values_ms)
    assert cases[-1]["warmup_ms"] == settings.warmup_check_ms


def test_threshold_crossings_are_upward_only():
    np.testing.assert_array_equal(
        threshold_crossings(np.array([0.0, 1.0, 1.1, 0.9, 1.2]), 0.5, 1.0),
        [1, 4],
    )


def test_missing_comparison_data_is_inconclusive(tmp_path):
    plan = {"cases": [], "criteria": {}}
    pair = {"kind": "dt", "coarse": "missing-a", "fine": "missing-b", "primary": True}
    assert _cell_comparison(pair, plan, tmp_path, {})[0]["status"] == "inconclusive"
    assert _network_comparison(pair, plan, {})["status"] == "inconclusive"


def test_network_comparison_reports_numerical_mismatch():
    pair = {"kind": "dt", "coarse": "a", "fine": "b", "primary": True}
    plan = {"criteria": {"network_count_relative": 0.05, "network_last_spike_abs_ms": 1.0,
                          "warmup_drift_abs": 1e-5}}
    base = {"activity_status": "active_at_observation_tail", "pre_spikes": 0,
            "during_spikes": 10, "post_spikes": 20, "post_exc_spikes": 15,
            "post_inh_spikes": 5, "post_participants": 10, "tail_spikes": 3,
            "last_post_latency_ms": 12.0, "no_stimulus": False, "multiplier": 2.0}
    other = {**base, "post_spikes": 40}
    result = _network_comparison(pair, plan, {"a": base, "b": other})
    assert result["status"] == "mismatch"
    assert result["differences"]["post_spikes"]["delta"] == -20


def test_no_stimulus_requires_zero_spikes_on_both_cases():
    pair = {"kind": "dt", "coarse": "a", "fine": "b", "primary": False}
    plan = {"criteria": {"network_count_relative": 0.05, "network_last_spike_abs_ms": 1.0,
                          "warmup_drift_abs": 1e-5}}
    quiet = {"activity_status": "no_post_activity", "pre_spikes": 0, "during_spikes": 0,
             "post_spikes": 0, "post_exc_spikes": 0, "post_inh_spikes": 0,
             "post_participants": 0, "tail_spikes": 0, "last_post_latency_ms": None,
             "no_stimulus": True, "multiplier": 2.0}
    noisy = {**quiet, "pre_spikes": 1}
    result = _network_comparison(pair, plan, {"a": quiet, "b": noisy})
    assert result["status"] == "mismatch"
    assert result["differences"]["no_stimulus_zero_spikes"] == {"coarse": True, "fine": False}


def test_short_cpu_cell_runs_reconstruct_recorded_spikes(tmp_path):
    pytest.importorskip("pygenn")
    from src.utils.experiments.baseline_network import BaselineSettings, make_snapshot, run_case

    root = Path(__file__).resolve().parents[2]
    base = yaml.safe_load((root / "test/utils/fixtures/baseline_small.yaml").read_text())
    base.update(n=2, n_exc=1, probability=0.0, warmup_ms=2.0, pulse_ms=1.0,
                post_ms=2.0, tail_ms=1.0, stimulus_fraction=1.0, sample_ms=0.1,
                rate_bin_ms=0.1,
                weights={"ee": 0.0, "ei": 0.0, "ie": 0.0, "ii": 0.0})
    snapshot = tmp_path / "shared_wiring.npz"
    cfg0 = BaselineSettings.model_validate({**base, "dt_ms": 0.1})
    make_snapshot(cfg0, 11, snapshot)
    for dt in (0.1, 0.05):
        cfg = BaselineSettings.model_validate({**base, "dt_ms": dt, "sample_ms": dt})
        case = {"case_id": f"cell_dt{dt}", "dt_ms": dt, "warmup_ms": 2.0,
                "pulse_ms": 1.0, "post_ms": 2.0}
        run_case(cfg, snapshot, tmp_path / case["case_id"], tmp_path / "build" / case["case_id"],
                 multiplier=0.0, amplitude_vector=np.array([0.4, 0.4]))
        rows = cell_rows(case, tmp_path / case["case_id"], amplitudes=[0.4])
        assert len(rows) == 2
        assert all(row["crossing_match"] for row in rows)


def test_analyze_marks_unselected_cases_and_writes_report(tmp_path):
    run = tmp_path / "fixture"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps({"cases": [], "settings": {}, "criteria": {}, "comparison_pairs": []}))
    (run / "case_results.json").write_text(json.dumps([{"case_id": "not-run", "status": "not_selected"}]))
    decision = analyze_run(run)
    assert decision["partial"] is True
    assert decision["counts"]["not_selected"] == 1
    assert (run / "report.md").exists()
    assert (run / "analysis_manifest.json").exists()
