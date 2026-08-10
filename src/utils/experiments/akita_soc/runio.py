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

import numpy as np
import yaml

from src.core.output_manager import CONFIG_NAME, CONNECTIVITY_NAME, locate

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


# 疎 (COO) の重みファイルから密行列を復元してよい上限。これを超えると
# (N,N) が確保できないため、呼び出し側で COO のまま扱う必要がある。
DENSE_RECONSTRUCTION_LIMIT = 20000


def load_weight_matrix(path: Path) -> np.ndarray:
    """重み npz を密な (N,N) 行列として読む。

    2 つの形式を受け付ける:
      - 密形式: キー "weights" に (N,N) 行列
      - COO 形式: キー "data" (+ 同ディレクトリの connectivity.npz の row/col)
    COO の場合は N が大きすぎると復元しない (呼び出し側で COO のまま扱うこと)。
    """
    data = np.load(path, allow_pickle=True)

    if "weights" in data.files:
        weights = np.asarray(data["weights"], dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError(f"{path} must contain a square 2D weight matrix.")
        return weights

    if "data" not in data.files:
        raise KeyError(f"{path} does not contain 'weights' or 'data'.")

    # COO 形式。row/col は記録ごとに繰り返さず connectivity.npz に一度だけ持つ。
    values = np.asarray(data["data"], dtype=np.float64)
    if "row" in data.files and "col" in data.files:
        row = np.asarray(data["row"], dtype=np.int64)
        col = np.asarray(data["col"], dtype=np.int64)
        size = int(np.asarray(data["shape"])[0]) if "shape" in data.files else None
    else:
        connectivity_path = path.parent / CONNECTIVITY_NAME
        if not connectivity_path.exists():
            raise FileNotFoundError(
                f"{path} は COO 形式ですが、結合情報 {connectivity_path} が見つかりません。"
            )
        connectivity = np.load(connectivity_path, allow_pickle=True)
        row = np.asarray(connectivity["row"], dtype=np.int64)
        col = np.asarray(connectivity["col"], dtype=np.int64)
        size = int(np.asarray(connectivity["shape"])[0])

    if size is None:
        size = int(max(row.max(), col.max())) + 1
    if size > DENSE_RECONSTRUCTION_LIMIT:
        raise MemoryError(
            f"{path} は N={size} の疎行列です。密行列 ({size**2 * 8 / 2**30:.1f} GiB) には"
            " 復元しません。COO のまま扱ってください"
            " (src/utils/plotting/network.py の粗視化プロットを参照)。"
        )

    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[row, col] = values
    return matrix


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def connection_mask_from_config(run_dir: Path, size: int) -> np.ndarray | None:
    config_path = locate(run_dir, CONFIG_NAME)
    config = _load_yaml(config_path) if config_path is not None else {}
    connection = config.get("network", {}).get("connection", {})
    profile = connection.get("profile_name")
    allow_self = bool(connection.get("allow_self_connections", False))
    p = connection.get("p")

    if profile == "constant_prob_full" or p == 1.0:
        mask = np.ones((size, size), dtype=bool)
        if not allow_self:
            np.fill_diagonal(mask, False)
        return mask
    return None


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
