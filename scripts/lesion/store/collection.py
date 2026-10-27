"""損傷 run の束を読む。`compare` (run をまたぐ比較) のリーダー。

束ねる単位は**同じ親条件**: 親 run (develop) を並べた条件ディレクトリの下にある損傷 run
全部と、切らずに走り続けた対照 run の条件ディレクトリ。

    <条件>/                    ← cond_dir
    ├── seed01/                親 run (config.yaml を持つ)
    │   ├── bridge-BX1-3/      損傷 run (data/lesion.json を持つ)
    │   └── random-…/
    └── lesion_compare/        この束の表と図 (`paths.compare_dir()`)

**ここは読み出しだけ。** 要約の表を作るのは `analysis/summary.py`、それを読むのが
`runs()` / `control()`。損傷 run も対照 run も、**ファイルの有無で見つけ、
ディレクトリ名は見ない**。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core.config_manager import CONFIG_NAME, ConfigManager
from src.core.layout import NetworkLayout
from src.utils.runview import MissingData

from scripts.lesion.store import paths
from scripts.lesion.store.parent_run import Parent, load_parent, load_parent_metrics
from scripts.lesion.store.formats import (CONTROL_TABLE_NAME, MANIFEST_NAME,
                                          RUNS_TABLE_NAME, load_metrics)
from scripts.lesion.store.views import Series, open_run


@dataclass(frozen=True)
class ControlRun:
    """切らずに走り続けた run (develop 形式)。`metrics` の時刻の列は `hour`。"""
    run_dir: Path
    config: object
    metrics: pd.DataFrame

    def state_at(self, hour: float) -> Parent:
        """記録時刻 `hour` [h] の結合・重み・スパイク。読み方は親 run と同じ (`load_parent()`)。"""
        return load_parent(self.run_dir, hour)

    def layout(self) -> NetworkLayout:
        """この run の config から組んだ `NetworkLayout` (E/I などの軸)。"""
        return NetworkLayout.from_config(self.config)


def _runs_under(directory: Path) -> list[Path]:
    """`config.yaml` を持つ run を列挙する。ディレクトリそのものが run ならそれ 1 つ。"""
    if (directory / CONFIG_NAME).exists():
        return [directory]
    return sorted(child for child in directory.iterdir()
                  if child.is_dir() and (child / CONFIG_NAME).exists())


class Collection:
    """1 つの親条件の損傷 run と、その対照 run の束。"""

    def __init__(self, cond_dir: str | Path, control_dir: str | Path):
        self.cond_dir = Path(cond_dir)
        self.control_dir = Path(control_dir)

    @property
    def out_dir(self) -> Path:
        """この束の表の置き場所 (`<条件>/lesion_compare/`)。"""
        return paths.compare_dir(self.cond_dir)

    def lesion_runs(self) -> tuple[Series, ...]:
        """条件ディレクトリの下の損傷 run を全部開く。親 run ごと、名前順。"""
        found = [child
                 for parent in _runs_under(self.cond_dir)
                 for child in sorted(parent.iterdir())
                 if child.is_dir() and paths.data_path(child, MANIFEST_NAME).exists()]
        if not found:
            raise FileNotFoundError(
                f"{self.cond_dir} の下に損傷 run ({MANIFEST_NAME} を持つ run) がありません。")
        return tuple(open_run(run, require_windows=False) for run in found)

    def control_runs(self) -> tuple[ControlRun, ...]:
        """対照の条件ディレクトリの下の run を全部読む。"""
        found = _runs_under(self.control_dir)
        if not found:
            raise FileNotFoundError(f"{self.control_dir} の下に run がありません。")
        manager = ConfigManager()
        return tuple(ControlRun(run_dir=run,
                                config=manager.load_resolved(run / CONFIG_NAME),
                                metrics=load_parent_metrics(run))
                     for run in found)

    def runs(self) -> pd.DataFrame:
        """`runs.csv` (損傷 run 1 本 = 1 行) を読む。**呼ばれた時点で読む。**"""
        return self._table(RUNS_TABLE_NAME)

    def control(self) -> pd.DataFrame:
        """`control.csv` (対照 run 1 本 = 1 行) を読む。"""
        return self._table(CONTROL_TABLE_NAME)

    def _table(self, name: str) -> pd.DataFrame:
        path = self.out_dir / name
        if not path.exists():
            raise MissingData(name, f"{path} がありません")
        return load_metrics(path)


def open_collection(cond_dir: str | Path, control_dir: str | Path) -> Collection:
    """損傷 run の束を開く。run を実際に開くのは `lesion_runs()` を呼んだとき。"""
    cond_dir, control_dir = Path(cond_dir), Path(control_dir)
    for directory in (cond_dir, control_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} がありません。")
    return Collection(cond_dir, control_dir)
