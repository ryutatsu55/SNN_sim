"""**親 run** (develop 形式) を読む。損傷実験だけが持つ読み口。

損傷実験は「育ったネットワークを引き継いで切る」ので、自分の記録を読む前に
**他の実験が書いた run** を読む必要がある。ここがその唯一の場所。

## なぜ `scripts/develop/store/` を import しないのか

実験どうしの横 import を作らない、というのがこのプロジェクトの規則
(ルート `CLAUDE.md`)。develop の内部事情に lesion がぶら下がると、develop を直すたびに
lesion が壊れる。

代わりに **develop 形式の記録を読む実装をこちらが持つ。** 読むのは 5 つだけで、
develop の `store/records.py` 全体ではない:

    config.yaml            ネットワーク設定 (同じ seed から同じネットワークが出る)
    data/connectivity.npz  結合構造 (row / col / shape)
    data/weights_{h}h.npz  引き継ぐ時点の重み (値ベクトル)
    data/layout_axes.npz   外部軸 (module など)。任意
    data/axon_geometry.npz 軸索の折れ線。任意 (axon_growth 系のみ)

**この 5 つは develop と lesion の契約**であって develop の内部事情ではない、というのが
ここに書いてよい理由。ファイル名の規約が変わったら、ここも直す —— 直す場所が 1 つで、
そのことが docstring に書いてあれば、横 import より安全に保てる。

読めた値は `src/utils/runview.py` の形 (`Wiring`) で返す。**親も子も、読めた値の形は同じ。**
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.config_manager import ConfigManager
from src.core.output_manager import (AXES_NAME, AXONS_NAME, CONFIG_NAME, CONNECTIVITY_NAME,
                                     DATA_SUBDIR)
from src.utils.runview import Wiring

# develop / akita_soc の記録ファイル名 `<種別>_<時刻>h.npz`。
RECORD_PATTERN = re.compile(r"(?P<kind>[A-Za-z_]+)_(?P<hour>.+)h\.npz")
WEIGHTS = "weights"
METRICS_NAME = "metrics.csv"
# 親 run の終盤ドリフトとして manifest に添える行数。
DRIFT_TAIL_ROWS = 3


@dataclass(frozen=True)
class Parent:
    """引き継ぐ親 run の中身。"""
    run_dir: Path
    config: object
    hour: float
    wiring: Wiring
    weights: np.ndarray
    axes_path: Path | None
    geometry: object | None
    drift: dict


def _data_dir(run_dir: Path) -> Path:
    """親 run の `data/`。**旧レイアウト (run 直下に平置き) は受け付けない。**

    吸収しても意味がないため。記録窓の原点を持たない古い run はどのみち引き継げず、
    置き場所だけ合わせても後で別のエラーになる。
    """
    data_dir = run_dir / DATA_SUBDIR
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"{run_dir} に {DATA_SUBDIR}/ がありません。"
            " 旧レイアウトの run は引き継げないので、親を再実行してください。")
    return data_dir


def _discover_weight_records(data_dir: Path) -> list[tuple[float, Path]]:
    """`weights_{h}h.npz` を時刻の昇順で列挙する。"""
    found = []
    for path in data_dir.glob(f"{WEIGHTS}_*h.npz"):
        match = RECORD_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        try:
            found.append((float(match["hour"]), path))
        except ValueError:
            continue
    return sorted(found, key=lambda item: item[0])


def _load_weights(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "data" not in data.files:
        if "weights" in data.files:
            raise ValueError(
                f"{path} は密形式 (キー 'weights') で保存された古い run です。"
                " 引き継げるのは COO 形式 (キー 'data' + connectivity.npz) だけです。")
        raise KeyError(f"{path} にキー 'data' がありません。")
    return np.asarray(data["data"], dtype=np.float64).reshape(-1)


def _load_wiring(data_dir: Path) -> Wiring:
    path = data_dir / CONNECTIVITY_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} がありません。密形式で保存された古い run は引き継げません。")
    data = np.load(path, allow_pickle=True)
    shape = np.asarray(data["shape"], dtype=np.int64)
    return Wiring(row=np.asarray(data["row"], dtype=np.int64),
                  col=np.asarray(data["col"], dtype=np.int64),
                  shape=(int(shape[0]), int(shape[1])))


def _load_drift(data_dir: Path) -> dict:
    """親 run の `metrics.csv` の末尾数行。**結果を読むときのドリフトの目安。**

    sham が無いので「回復」と「損傷が無くても進んだ発達の続き」は分離できない
    (`scripts/lesion/README.md`)。
    """
    path = data_dir / METRICS_NAME
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[-DRIFT_TAIL_ROWS:]
    return {"note": "sham 無しなので、回復幅がこのドリフト幅と同オーダーなら結論を出さない",
            "tail_rows": rows}


def _load_geometry(data_dir: Path):
    """保存済みの軸索幾何。持たない run では None。

    `src/models` の import は幾何がある run でだけ走るよう関数内に置いてある。
    """
    path = data_dir / AXONS_NAME
    if not path.exists():
        return None
    from src.models.network.connectors import AxonGeometry
    return AxonGeometry.load(path)


def load_parent(run_dir: Path, from_hour: float | None = None) -> Parent:
    """親 run の config / 重み / 幾何を読む。

    Args:
        run_dir: 親の run ディレクトリ (`config.yaml` と `data/` を持つもの)。
        from_hour: 引き継ぐ記録時刻 [h]。None なら**最後の記録**。
    """
    run_dir = Path(run_dir)
    config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} がありません。build を通った run を指してください。")
    config = ConfigManager().load_resolved(config_path)

    data_dir = _data_dir(run_dir)
    records = _discover_weight_records(data_dir)
    if not records:
        raise FileNotFoundError(f"{WEIGHTS}_*h.npz が見つかりません: {data_dir}")
    if from_hour is None:
        hour, weights_path = records[-1]
    else:
        matches = [item for item in records if abs(item[0] - float(from_hour)) < 1e-9]
        if not matches:
            raise FileNotFoundError(
                f"{from_hour} h の重み記録がありません "
                f"(ある時刻: {[item[0] for item in records]})")
        hour, weights_path = matches[0]

    axes_path = data_dir / AXES_NAME
    return Parent(
        run_dir=run_dir,
        config=config,
        hour=hour,
        wiring=_load_wiring(data_dir),
        weights=_load_weights(weights_path),
        axes_path=axes_path if axes_path.exists() else None,
        geometry=_load_geometry(data_dir),
        drift=_load_drift(data_dir),
    )
