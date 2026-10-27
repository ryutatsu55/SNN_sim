"""**親 run** (develop 形式) を読む。損傷実験だけが持つ読み口。

損傷実験は「育ったネットワークを引き継いで切る」ので、自分の記録を読む前に
**他の実験が書いた run** を読む必要がある。ここがその唯一の場所。

## なぜ `scripts/develop/store/` を import しないのか

実験どうしの横 import を作らない、というのがこのプロジェクトの規則
(ルート `CLAUDE.md`)。develop の内部事情に lesion がぶら下がると、develop を直すたびに
lesion が壊れる。

代わりに **develop 形式の記録を読む実装をこちらが持つ。** 読むのは 7 つだけで、
develop の `store/formats.py` 全体ではない:

    config.yaml            ネットワーク設定 (同じ seed から同じネットワークが出る)
    data/connectivity.npz  結合構造 (row / col / shape)
    data/weights_{h}h.npz  引き継ぐ時点の重み (値ベクトル)
    data/spikes_{h}h.npz   同じ時点のスパイク = **損傷実験のベースライン窓**
    data/layout_axes.npz   外部軸 (module など)。任意
    data/axon_geometry.npz 軸索の折れ線。任意 (axon_growth 系のみ)
    data/metrics.csv       親 run の終盤ドリフト。任意 (無ければ理由が manifest に入る)。
                           `compare` の対照 (切らずに走り続けた run) の発火率もここから読む

**この 7 つは develop と lesion の契約**であって develop の内部事情ではない、というのが
ここに書いてよい理由。**この一覧が、develop 側で記録の規約を変える人にとっての索引になる。**
ファイル名の規約が変わったら、ここも直す —— 直す場所が 1 つで、そのことが docstring に
書いてあれば、横 import より安全に保てる。

したがって**ファイル名は下でローカルに宣言する。** `src/core` から import すると、
「lesion が親について何を仮定しているか」がこのファイルを読むだけでは分からなくなる。

読めた値は `src/utils/runview.py` の形 (`Wiring`) で返す。**親も子も、読めた値の形は同じ。**
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config_manager import ConfigManager
from src.utils.runview import Wiring

# ------------------------------------------------------------------------------------
# 親 (develop 形式) の run ディレクトリ規約。**上の 6 つの契約がここに対応する。**
#
# `store/paths.py` にも同名の定数があるが、あちらは**自分の run** の規約で、こちらは
# **親の run** の規約。値がたまたま一致しているだけなので、片方を変えても他方は動かない。
# ------------------------------------------------------------------------------------
CONFIG_NAME = "config.yaml"
DATA_SUBDIR = "data"
CONNECTIVITY_NAME = "connectivity.npz"
AXES_NAME = "layout_axes.npz"
AXONS_NAME = "axon_geometry.npz"
METRICS_NAME = "metrics.csv"

# develop / akita_soc の記録ファイル名 `<種別>_<時刻>h.npz`。
RECORD_PATTERN = re.compile(r"(?P<kind>[A-Za-z_]+)_(?P<hour>.+)h\.npz")
WEIGHTS = "weights"
SPIKES = "spikes"
# 親のスパイク npz が持つ鍵。**窓幅は持たない**ので、幅は親 config から取る。
RECORD_START_KEY = "record_start_ms"
# 親 run の終盤ドリフトとして manifest に添える行数。
DRIFT_TAIL_ROWS = 3

# **lesion が読む親 run (develop 形式) の `metrics.csv` の列。** develop の記録の列を
# 変えるときはここを見ること —— 実験どうしは import し合えないので、依存はここにしか書けない。
#
# 対照 run の回復を測る列 (`analysis/summary.py`)。無ければ止める。
CONTROL_METRIC_COLUMNS = ("hour", "mean_rate_hz")
# 終盤ドリフトとして `lesion.json` の `parent_drift` に写す列。親が持たない列は写さない。
DRIFT_METRIC_COLUMNS = ("hour", "mean_rate_hz", "mean_correlation", "bp_sigma_bp",
                        "delta_cr", "llr")


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
    # ベースライン窓。**この run では測らず、親の記録をそのまま使う。**
    spike_times: np.ndarray   # 窓の先頭を 0 とするローカル時刻 [ms]
    spike_ids: np.ndarray
    window_ms: float          # 親の記録窓の幅。lesion の record_window_ms とは違いうる


def _parent_data_dir(run_dir: Path) -> Path:
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


def _sibling_record(weights_path: Path, kind: str) -> Path:
    """同じ記録時刻の別の種別のファイル。**名前を組み立て直さず、置き換える。**

    `f"{hour:g}h"` を組み立て直すと、非整数の記録時刻で往復が合わない
    (`0.333333h` から読んだ float を書き戻すと別の名前になりうる)。
    """
    return weights_path.with_name(
        weights_path.name.replace(f"{WEIGHTS}_", f"{kind}_", 1))


def _load_spikes(path: Path, hour: float) -> tuple[np.ndarray, np.ndarray]:
    """親の記録窓のスパイクを、**窓の先頭を 0 とするローカル時刻**で読む。

    親は絶対時刻で書き、原点を `record_start_ms` に持つ。契約 (`runview.Spikes`) は
    ローカル時刻なので、ここで引く。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} がありません。{hour} h のスパイク記録が無いので、この時刻を"
            " ベースラインにできません (task.from_hour を別の時刻にするか親を再実行)。")
    with np.load(path, allow_pickle=False) as data:
        if RECORD_START_KEY not in data.files:
            raise KeyError(
                f"{path} に {RECORD_START_KEY} がありません。"
                " 記録窓の原点を持たない古い run は引き継げません。")
        start_ms = float(data[RECORD_START_KEY])
        times = np.asarray(data["times"], dtype=np.float64) - start_ms
        ids = np.asarray(data["ids"], dtype=np.int64)
    return times, ids


def _load_drift(data_dir: Path) -> dict:
    """親 run の `metrics.csv` の末尾数行の `DRIFT_METRIC_COLUMNS`。**結果を読むときのドリフトの目安。**

    親がこれを持たなくても run は完走するので、**取れなかったときは理由を返す。**
    空 dict だと「親に metrics.csv が無かった」と「読んだが行が無かった」が区別できず、
    `parent_drift` が空であることの意味が読めなくなる。
    """
    path = data_dir / METRICS_NAME
    if not path.exists():
        return {"unavailable": f"親 run に {METRICS_NAME} がありません: {data_dir}"}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[-DRIFT_TAIL_ROWS:]
    if not rows:
        return {"unavailable": f"親 run の {METRICS_NAME} に行がありません: {path}"}
    rows = [{key: row[key] for key in DRIFT_METRIC_COLUMNS if key in row} for row in rows]
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


def load_parent_geometry(run_dir: Path):
    """親 run の軸索幾何 (= 切断**前**の幾何)。持たない run では None。

    切断したシナプスの経路を描くときに使う。この run の `axon_geometry.npz` は切断後に
    絞ってあり、切ったシナプスの接触点を持たないため。
    """
    return _load_geometry(_parent_data_dir(Path(run_dir)))


def load_parent(run_dir: Path, from_hour: float | None = None) -> Parent:
    """親 run の config / 重み / ベースライン窓 / 幾何を読む。

    **重みとベースラインは同じ時刻から取る。** 別の時刻から取ると、ベースラインの
    活動と切断直前の重みが食い違う。

    Args:
        run_dir: 親の run ディレクトリ (`config.yaml` と `data/` を持つもの)。
        from_hour: 引き継ぐ記録時刻 [h]。None なら**最後の記録**。
    """
    run_dir = Path(run_dir)
    config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} がありません。親の run ディレクトリを指してください。")
    config = ConfigManager().load_resolved(config_path)

    data_dir = _parent_data_dir(run_dir)
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

    window_ms = getattr(config.task, "record_window_ms", None)
    if window_ms is None:
        raise ValueError(
            f"{config_path} の task に record_window_ms がありません。"
            " ベースライン窓の幅が決まらないので引き継げません。")
    spike_times, spike_ids = _load_spikes(_sibling_record(weights_path, SPIKES), hour)

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
        spike_times=spike_times,
        spike_ids=spike_ids,
        window_ms=float(window_ms),
    )


def load_parent_metrics(run_dir: Path) -> pd.DataFrame:
    """develop 形式の run の `metrics.csv` を丸ごと読む (時刻の列は `hour`)。

    損傷 run の対照 (切らずに走り続けた run) を読む口。
    """
    path = _parent_data_dir(Path(run_dir)) / METRICS_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} がありません。")
    table = pd.read_csv(path)
    missing = [column for column in CONTROL_METRIC_COLUMNS if column not in table.columns]
    if missing:
        raise KeyError(f"{path} に対照の回復を測る列がありません: {missing}")
    return table
