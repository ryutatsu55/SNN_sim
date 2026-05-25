import re
from datetime import datetime
from pathlib import Path


def _sanitize_dir_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "simulation"


def create_run_output_dir(
    simulation_name: str,
    base_dir: str | Path = "outputs",
    timestamp: str | None = None,
) -> Path:
    """シミュレーション1回分の出力ディレクトリを作成する。"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = Path(base_dir) / _sanitize_dir_name(simulation_name) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
