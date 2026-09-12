"""Akita SOC 実験の出力ファイル規約。

`weights_{h}h.npz` / `spikes_{h}h.npz` の "h" は 72 時間発達実験の記録時刻であって、
プロジェクト共通の概念ではない。よって `src/core/output_manager.py` (run ディレクトリ
構造・config.yaml・layout_axes.npz など) には置かず、この実験のモジュールが持つ。

**この規約を知っているのはここだけ**。書き出し (`record_filename`) と読み取り
(`parse_hour` / `discover_records`) が同じ 1 つの正規表現とテンプレートを共有するので、
片方だけ変えて食い違うことがない。
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.core.output_manager import CONNECTIVITY_NAME

# 記録ファイル名 `<種別>_<時刻>h.npz`。種別に数字を含めない前提で時刻と切り分ける。
RECORD_PATTERN = re.compile(r"(?P<kind>[A-Za-z_]+)_(?P<hour>.+)h\.npz")

WEIGHTS = "weights"
SPIKES = "spikes"


@dataclass(frozen=True)
class RecordFile:
    hour: float
    path: Path


def record_filename(kind: str, hour: float) -> str:
    """記録ファイル名を組み立てる。読み取り側と同じ規約を使うための唯一の入口。"""
    return f"{kind}_{hour:g}h.npz"


def parse_record_name(name: str) -> tuple[str, float]:
    """`weights_6.5h.npz` -> ("weights", 6.5)。規約に合わなければ ValueError。"""
    match = RECORD_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"記録ファイル名の規約に合いません: {name}")
    try:
        return match["kind"], float(match["hour"])
    except ValueError:
        raise ValueError(f"記録時刻を数値として読めません: {name}") from None


def parse_hour(path: Path | str, kind: str | None = None) -> float:
    """ファイル名から記録時刻 [h] を取り出す。

    `kind` を指定すると種別が一致することも確認する (spikes を weights として
    読んでしまう取り違えを防ぐ)。規約外なら **ValueError**。以前は失敗時に -1.0 を
    返す実装が混在していたが、それは「時刻 -1 の記録」として静かに紛れ込む。
    """
    found_kind, hour = parse_record_name(Path(path).name)
    if kind is not None and found_kind != kind:
        raise ValueError(f"{kind} の記録を期待しましたが {found_kind} でした: {path}")
    return hour


def record_glob(kind: str) -> str:
    """記録ファイルを列挙する glob パターン。存在確認だけしたい側もこれを使う。"""
    return f"{kind}_*h.npz"


def discover_records(directory: Path, kind: str) -> list[RecordFile]:
    """`directory` 直下の記録ファイルを時刻の昇順で返す。"""
    files = [RecordFile(hour=parse_hour(path, kind), path=path)
             for path in Path(directory).glob(record_glob(kind))]
    return sorted(files, key=lambda item: item.hour)


class Connectivity(NamedTuple):
    """run 全体で共通の結合構造 (`connectivity.npz`)。

    構造はシミュレーション中に変わらないので run につき 1 ファイル。各記録の
    `weights_{h}h.npz` は値ベクトル (`data`) だけを持ち、この row/col と index が整合する。
    """
    row: np.ndarray
    col: np.ndarray
    shape: tuple[int, int]


def load_connectivity(directory: Path) -> Connectivity:
    """`connectivity.npz` を読む。記録ごとではなく run につき 1 回だけ呼ぶこと。"""
    path = Path(directory) / CONNECTIVITY_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"結合情報 {path} が見つかりません。"
            " weights_*h.npz と対になる connectivity.npz が必要です"
            " (密形式で保存された古い run は再実行してください)。"
        )
    data = np.load(path, allow_pickle=True)
    shape = np.asarray(data["shape"], dtype=np.int64)
    return Connectivity(
        row=np.asarray(data["row"], dtype=np.int64),
        col=np.asarray(data["col"], dtype=np.int64),
        shape=(int(shape[0]), int(shape[1])),
    )


def load_weight_values(path: Path) -> np.ndarray:
    """重み npz から値ベクトル (1D) を読む。`connectivity.npz` の row/col と index 整合。

    密形式 (キー `weights` に (N,N) 行列) で保存された古い run は受け付けない。行列だけでは
    「結合が無い」と「重みが 0 まで下がった」を区別できず、統計に 0 が混ざるため。
    """
    data = np.load(path, allow_pickle=True)
    if "data" not in data.files:
        if "weights" in data.files:
            raise ValueError(
                f"{path} は密形式 (キー 'weights') で保存された古い run です。"
                " 解析は COO 形式 (キー 'data' + connectivity.npz) のみを受け付けます。"
                " 再実行してください。"
            )
        raise KeyError(f"{path} にキー 'data' がありません。")
    return np.asarray(data["data"], dtype=np.float64).reshape(-1)


def write_metrics_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hour",
        "block",
        "mean",
        "max",
        "nonzero_fraction",
        "ge_0p5_fraction",
        "ge_0p9_fraction",
        "at_1_fraction",
        "mean_delta_from_previous",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
