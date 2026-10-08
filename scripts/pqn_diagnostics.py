"""PQN応答と時間刻み収束の診断CLI。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.experiments.pqn_diagnostics import analyze_run, create_plan, load_settings, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "run":
            command.add_argument("--output-root", default="outputs/pqn_diagnostics")
            command.add_argument("--case-id", action="append")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            settings, resolved = load_settings(args.config)
            print(json.dumps(create_plan(settings, resolved), ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "run":
            path, code = run_experiment(args.config, args.output_root, args.case_id)
            print(path)
            return code
        result = analyze_run(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
