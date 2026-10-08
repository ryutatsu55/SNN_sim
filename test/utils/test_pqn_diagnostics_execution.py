"""診断実行管理の比較計画、失敗継続、版・進捗保存を確認する。"""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.experiments import baseline_network, pqn_diagnostics
from src.utils.experiments.pqn_diagnostics_execution import create_plan, run_experiment

CONFIG = ROOT / "configs/pqn_diagnostics.yaml"


def test_resolved_plan_and_comparison_pairs():
    settings, base = pqn_diagnostics.load_settings(CONFIG)
    plan = create_plan(settings, base)
    assert len(plan["cases"]) == 22
    assert len(plan["comparison_pairs"]) == 17
    assert sum(p["primary"] for p in plan["comparison_pairs"]) == 7
    cases = {c["case_id"]: c for c in plan["cases"]}
    for pair in plan["comparison_pairs"]:
        a, b = cases[pair["coarse"]], cases[pair["fine"]]
        assert a["kind"] == b["kind"]
        assert a["kwargs"] == b["kwargs"]
        av, bv = dict(a["settings"]), dict(b["settings"])
        ignore = ("dt_ms", "sample_ms") if pair["kind"] == "dt" else ("warmup_ms",)
        for key in ignore:
            av.pop(key)
            bv.pop(key)
        assert av == bv
    assert plan["created_at"].endswith("+09:00")


def test_invalid_case_settings_fail_before_run():
    settings, base = pqn_diagnostics.load_settings(CONFIG)
    bad = settings.model_copy(update={"dt_values_ms": [0.05, 0.003125]})
    with pytest.raises(ValueError):
        create_plan(bad, base)


def test_failures_continue_without_false_zero_results(tmp_path, monkeypatch):
    settings, base = pqn_diagnostics.load_settings(CONFIG)
    ids = [c["case_id"] for c in create_plan(settings, base)["cases"][:3]]
    calls = []

    def fake_case(cfg, snapshot, out, build, **kwargs):
        calls.append(out.name)
        out.mkdir()
        if len(calls) == 1:
            (out / "status.json").write_text(json.dumps({
                "status": "failed", "phase": "before_simulation",
                "error": "実行プロセスが異常終了しました: exitcode=-11",
            }))
            (out / "build_retries.json").write_text(json.dumps([{"exitcode": -11}] * 2))
            raise RuntimeError("実行プロセスが異常終了しました: exitcode=-11")
        if len(calls) == 2:
            (out / "status.json").write_text('{"status":"failed"}')
            raise FloatingPointError("非有限値")
        (out / "status.json").write_text('{"status":"complete"}')

    monkeypatch.setattr(baseline_network, "run_case", fake_case)
    monkeypatch.setattr(pqn_diagnostics, "analyze_run", lambda path: {})
    out, code = run_experiment(CONFIG, tmp_path, ids)
    results = json.loads((out / "case_results.json").read_text())
    assert calls == ids
    assert code == 1
    assert len(results) == 22
    assert [r["status"] for r in results[:3]] == ["crashed", "non_finite", "complete"]
    assert results[0]["exitcode"] == -11
    assert results[0]["retry_count"] == 2
    assert results[0]["phase"] == "before_simulation"
    assert "post_spikes" not in results[0]
    assert all(r["status"] == "not_selected" for r in results[3:])


def test_partial_success_is_partial(tmp_path, monkeypatch):
    settings, base = pqn_diagnostics.load_settings(CONFIG)
    case_id = create_plan(settings, base)["cases"][0]["case_id"]

    def fake_case(cfg, snapshot, out, build, **kwargs):
        out.mkdir()
        (out / "status.json").write_text('{"status":"complete"}')

    monkeypatch.setattr(baseline_network, "run_case", fake_case)
    monkeypatch.setattr(pqn_diagnostics, "analyze_run", lambda path: {})
    out, code = run_experiment(CONFIG, tmp_path, [case_id])
    manifest = json.loads((out / "manifest.json").read_text())
    assert code == 0
    assert manifest["status"] == "partial"
    assert manifest["counts"]["complete"] == 1
    assert manifest["counts"]["not_selected"] == 21


def test_interrupted_results_keep_pending_cases(tmp_path, monkeypatch):
    def fake_case(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(baseline_network, "run_case", fake_case)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(CONFIG, tmp_path)
    out = next(tmp_path.iterdir())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["counts"]["pending"] == 22


def test_comparison_rejects_different_inputs(tmp_path):
    import numpy as np
    import yaml
    from src.utils.experiments.pqn_diagnostics_execution import validate_comparison_pair

    settings, base = pqn_diagnostics.load_settings(CONFIG)
    plan = create_plan(settings, base)
    pair = plan["comparison_pairs"][0]
    cases = {c["case_id"]: c for c in plan["cases"]}
    for cid in (pair["coarse"], pair["fine"]):
        directory = tmp_path / "cases" / cid
        directory.mkdir(parents=True)
        (directory / "settings.yaml").write_text(yaml.safe_dump(cases[cid]["settings"]))
        (directory / "metadata.json").write_text(json.dumps(dict(
            snapshot_fingerprint="identical", connectivity_hash="identical",
            absolute_weights={"ee": 0}, source_hashes={"model": "same"},
        )))
        (directory / "neuron_constants.json").write_text('{"V":0}')
        np.save(directory / "stimulus_ids.npy", [0, 1])
        np.save(directory / "stimulus_amplitudes.npy", [0.4, 0.4])
    validate_comparison_pair(tmp_path, plan, pair)
    changed = tmp_path / "cases" / pair["fine"]
    np.save(changed / "stimulus_amplitudes.npy", [0.4, 0.6])
    with pytest.raises(ValueError, match="入力が異なります"):
        validate_comparison_pair(tmp_path, plan, pair)
    np.save(changed / "stimulus_amplitudes.npy", [0.4, 0.4])
    values = dict(cases[pair["fine"]]["settings"])
    values["stimulus_fraction"] = 0.2
    (changed / "settings.yaml").write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="保存計画"):
        validate_comparison_pair(tmp_path, plan, pair)
