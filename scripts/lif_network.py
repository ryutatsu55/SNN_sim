"""LIF基準ネットワークの計画・バックグラウンド実行・管理。"""

import argparse
from datetime import datetime
import faulthandler
import json
import os
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))

from src.utils.experiments.lif_network import (
    NetworkSettings, analyze_case, inventory, load_settings, make_plan, preset_settings, run_case, save_json,
)
from src.utils.experiments import lif_network_job as job


def main():
    parser = argparse.ArgumentParser(description="LIF可塑性ネットワークを計画・実行・管理します")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "start"):
        item = sub.add_parser(name)
        item.add_argument("--config", type=Path, default=ROOT / "configs/lif_network.yaml")
        item.add_argument("--preset", choices=("smoke", "pilot", "full"), default="full")
        if name == "start":
            item.add_argument("--run-dir", type=Path)
            item.add_argument("--remote")
    for name in ("status", "stop", "resume", "upload", "analyze", "_manage", "_case"):
        item = sub.add_parser(name)
        item.add_argument("--run-dir", type=Path, required=True)
        if name in ("resume", "upload", "_manage"):
            item.add_argument("--remote")
        if name == "_manage":
            item.add_argument("--mode", choices=("run", "upload"), default="run")
        if name == "stop":
            item.add_argument("--force", action="store_true")
        if name == "_case":
            item.add_argument("--case-id", required=True)
            item.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    if args.action in ("plan", "start"):
        cfg = preset_settings(load_settings(args.config), args.preset)
        if args.action == "plan":
            print(json.dumps(dict(settings=cfg.model_dump(), cases=make_plan(cfg)), ensure_ascii=False, indent=2))
            return 0
        stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S-%f")
        out = args.run_dir or ROOT / "outputs/lif_network" / stamp
        out = job.prepare(out, cfg, args.remote)
        print(json.dumps(job.launch(out), ensure_ascii=False))
    elif args.action == "status":
        out = args.run_dir.resolve()
        status = job.read_json(out / "status.json")
        if status is None:
            raise ValueError("実行記録がありません")
        status.update(process_alive=job.running(out))
        if status["status"] == "running" and not status["process_alive"]:
            status["status"] = "interrupted"
            status["note"] = "管理プロセスがありません。保存済みの状態から条件単位で再開できます。"
        attempt = status.get("attempt")
        if attempt:
            status["progress"] = job.read_json(out / attempt / "progress.json")
        status["upload"] = job.read_json(out / "upload/status.json")
        print(json.dumps(status, ensure_ascii=False, indent=2))
    elif args.action == "stop":
        if not job.running(args.run_dir):
            raise ValueError("実行中の管理プロセスがありません")
        (args.run_dir / ("STOP_NOW" if args.force else "STOP")).touch()
        print("停止を要求しました。" + ("実行中の条件を中断します。" if args.force else "現在の条件が終わってから停止します。"))
    elif args.action in ("resume", "upload"):
        print(json.dumps(job.launch(args.run_dir, mode="run" if args.action == "resume" else "upload", remote=args.remote), ensure_ascii=False))
    elif args.action == "_manage":
        return job.manage(args.run_dir, args.mode, args.remote)
    elif args.action == "_case":
        faulthandler.enable()
        cfg = NetworkSettings.model_validate(job.read_json(args.run_dir / "settings.json"))
        case = next(row for row in job.read_json(args.run_dir / "plan.json") if row["case_id"] == args.case_id)
        build = ROOT / "genn_code/lif_network" / args.run_dir.name / args.case_id / args.attempt.name
        try:
            run_case(cfg, case, args.attempt, build)
        except Exception as exc:
            args.attempt.mkdir(parents=True, exist_ok=True)
            save_json(args.attempt / "error.json", dict(type=type(exc).__name__, error=str(exc)))
            raise
    elif args.action == "analyze":
        if job.running(args.run_dir):
            raise ValueError("実行中の結果は変更できません")
        for case in job.read_json(args.run_dir / "plan.json"):
            outcome = job.case_outcome(args.run_dir, case["case_id"])
            if outcome.get("status") == "complete":
                out = args.run_dir / outcome["attempt"]
                analyze_case(out)
                save_json(out / "artifacts.json", inventory(out, exclude=("artifacts.json",)))
        job.summarize(args.run_dir)
        job.plot_campaign(args.run_dir)
        save_json(args.run_dir / "analysis_manifest.json", dict(
            analyzed=job.now(), sources={str(path.relative_to(ROOT)): job.sha256(path)
                                        for path in job.source_paths()}))
        save_json(args.run_dir / "artifact_manifest.json", inventory(args.run_dir, exclude=(
            "upload", ".lock", "run.log", "job.json", "status.json", "artifact_manifest.json")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
