"""akita_soc 実験の記録ファイル名の規約。

`weights_{h}h.npz` / `spikes_{h}h.npz` の "h" は 72 時間発達実験の記録時刻であって、
プロジェクト共通の概念ではない。よって `src/core/output_manager.py` (run ディレクトリ
構造・config.yaml・layout_axes.npz など) には置かず、この実験のモジュールが持つ。

**この規約を知っているのはここだけ**。ファイル名 (`record_filename` / `discover_records`)
だけでなく、**npz の中の鍵の名前**も書き出しと読み取りが対で持つ (`save_spikes` /
`load_spikes` など)。片方だけ変えて食い違うことがない。

ファイルを**置く場所**は `paths.py` が持ち、**run 1 つを時刻順に読む**のは `series.py`。
このモジュールはファイル 1 つの読み書きまでを受け持つ。

返す値の形は `src/utils/runview.py` の契約 (`Spikes` / `Trace` / `Wiring`) に従う。
**この実験に固有なのはファイル名と npz の鍵名だけで、読めた値の形は全実験共通。**

**`scripts/develop/store/records.py` と同じ規約だが、別のファイルとして持つ。**
実験ごとに記録の中身は変わりうるので、共通化して片方の都合でもう片方が動くことを
避ける (図と同じ線引き)。共有するのは「読めた値の形」= 契約だけ。
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.output_manager import CONNECTIVITY_NAME
from src.utils.runview import Spikes, Trace, Wiring

# 記録ファイル名 `<種別>_<時刻>h.npz`。種別に数字を含めない前提で時刻と切り分ける。
RECORD_PATTERN = re.compile(r"(?P<kind>[A-Za-z_]+)_(?P<hour>.+)h\.npz")

WEIGHTS = "weights"
SPIKES = "spikes"
# 単一ニューロンの膜電位トレース (task.trace_neuron を指定した run だけが書く)。
# 記録窓の先頭を 1 ステップ刻みで採った V / Isyn_rec で、同じ窓のラスターと
# 時間軸が揃っている。図にしか残さないのは勿体ないので npz にも落とす。
TRACE = "trace"

# 記録時刻ごとの指標。本番の run も再解析も**同じこのファイル**に書く
# (どちらも metrics.build_row() を通るので列も値も一致する)。
METRICS_NAME = "metrics.csv"

MS_PER_HOUR = 60.0 * 60.0 * 1000.0

# スパイク npz に埋める記録窓の原点 [ms]。
#
# **ファイル名から時刻を復元しない**ための鍵。ファイル名は `f"{hour:g}h"` = 有効数字
# 6 桁なので、record_hours が非整数だと往復で元の値に戻らない (1/3 h なら約 1.2 ms
# ずれる)。原点がずれると burstiness_index のビン割りが変わり、本番と再解析で値が
# 食い違う。ファイル名は並べ替えと人間の目印だけに使い、**時刻の正は npz が持つ**。
RECORD_START_KEY = "record_start_ms"


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
    """ファイル名から記録時刻 [h] を取り出す。規約外なら **ValueError**。

    `kind` を指定すると種別の一致も確認する (spikes を weights として読む取り違えを防ぐ)。
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


def load_connectivity(directory: Path) -> Wiring:
    """`connectivity.npz` を読んで `Wiring` にする。run につき 1 回だけ呼ぶこと。

    構造はシミュレーション中に変わらないので run に 1 ファイル。各記録の
    `weights_{h}h.npz` は値ベクトル (`data`) だけを持ち、この row/col と index が整合する。
    """
    path = Path(directory) / CONNECTIVITY_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"結合情報 {path} が見つかりません。"
            " weights_*h.npz と対になる connectivity.npz が必要です"
            " (密形式で保存された古い run は再実行してください)。"
        )
    data = np.load(path, allow_pickle=True)
    shape = np.asarray(data["shape"], dtype=np.int64)
    wiring = Wiring(
        row=np.asarray(data["row"], dtype=np.int64),
        col=np.asarray(data["col"], dtype=np.int64),
        shape=(int(shape[0]), int(shape[1])),
    )
    _require_row_major(wiring, path)
    return wiring


def _require_row_major(wiring: Wiring, path: Path) -> None:
    """記録された COO が**行優先ソート済み**であることを確かめ、違えば止める。

    並びが違うと、位置で重みと対応づけたときに黙って別のシナプスに値が乗る。
    キーの狭義単調増加は「行優先かつ (pre, post) の重複なし」と同値。
    """
    if wiring.row.size < 2:
        return
    # **int64 で掛ける。** int32 のままだと N が大きいときに折り返り、別のペアが
    # 同じキーになって「ソート済み」に見えてしまう。
    key = wiring.row * int(wiring.shape[1]) + wiring.col
    bad = int(np.count_nonzero(np.diff(key) <= 0))
    if bad:
        raise ValueError(
            f"{path} の結合が行優先ソート済みではありません ({bad} 箇所)。\n"
            "  COO の並びを揃える前に作られた run です。位置で重みと対応づけると"
            " 別のシナプスに乗るので、読み込みを拒否しました。\n"
            "  再実行してください (docs/technical/compat_breaks.md 参照)。"
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


def save_connectivity(path: Path, row, col, shape) -> None:
    """結合構造を書く。**鍵名 (`row` / `col` / `shape`) を知る場所を 1 つにするための対。**

    構造はシミュレーション中に変わらないので run につき 1 回だけ呼ぶ。
    """
    np.savez_compressed(path, row=np.asarray(row), col=np.asarray(col),
                        shape=np.asarray(shape))


def write_table(rows: list[dict], out_path: Path) -> None:
    """dict の並びを CSV にする。**列は最初の行のキー順**、後の行で増えた分は末尾。

    `MetricsWriter` が「1 行ずつ追記する記録」なのに対し、こちらは**一度に確定する表**
    (構造の集計など) 用。列をここで決め打ちしないので、行を作る側が列を持てる。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_weight_values(path: Path, values: np.ndarray) -> None:
    """重みを値ベクトルとして書く。キー名を知っているのは読み書きのこの対だけ。"""
    np.savez_compressed(path, data=np.asarray(values))


def save_spikes(path: Path, times, ids, record_start_ms: float) -> None:
    """スパイクと**記録窓の原点**を書く。原点を一緒に残すのがこの関数の要点。"""
    np.savez_compressed(path, times=np.asarray(times), ids=np.asarray(ids),
                        **{RECORD_START_KEY: float(record_start_ms)})


def load_spikes(path: Path) -> Spikes:
    """スパイク npz を読み、**窓の原点を引いたローカル時刻**で返す。

    絶対時刻は返さない。必要なら `read_record_start_ms()` の値を足す。
    """
    with np.load(path) as data:
        if RECORD_START_KEY not in data.files:
            raise KeyError(
                f"{path} に {RECORD_START_KEY} がありません。"
                f" 記録窓の原点を持たない古い run です (再実行してください)。"
            )
        times = np.asarray(data["times"], dtype=np.float64)
        ids = np.asarray(data["ids"])
        start_ms = float(data[RECORD_START_KEY])
    return Spikes(times=times - start_ms, ids=ids)


def read_record_start_ms(path: Path) -> float:
    """スパイク npz から原点だけを読む。npz は鍵ごとに展開されるので配列は解かない。"""
    with np.load(path) as data:
        if RECORD_START_KEY not in data.files:
            raise KeyError(
                f"{path} に {RECORD_START_KEY} がありません。"
                f" 記録窓の原点を持たない古い run です (再実行してください)。"
            )
        return float(data[RECORD_START_KEY])


def save_trace(path: Path, V, I, *, dt: float, neuron_id: int, window_s: float,
               spike_times, spike_ids) -> None:
    """膜電位トレースを書く。図だけでなく生データも残す (後から作れないため)。"""
    np.savez_compressed(path, V=np.asarray(V), I=np.asarray(I), dt=float(dt),
                        neuron_id=int(neuron_id), window_s=float(window_s),
                        spike_times=np.asarray(spike_times),
                        spike_ids=np.asarray(spike_ids))


def load_trace(path: Path) -> Trace:
    with np.load(path) as data:
        return Trace(
            V=np.asarray(data["V"]), I=np.asarray(data["I"]),
            dt=float(data["dt"]), neuron_id=int(data["neuron_id"]),
            window_s=float(data["window_s"]),
            spike_times=np.asarray(data["spike_times"]),
            spike_ids=np.asarray(data["spike_ids"]),
        )


class MetricsWriter:
    """`metrics.csv` を **1 行ずつ追記**する。途中で落ちた run もそこまでが読める。

    ヘッダは**最初の行で確定**する。以降の行でキーが変わると例外。
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def append(self, row: dict) -> None:
        keys = list(row.keys())
        if self._fieldnames is None:
            self._fieldnames = keys
            with open(self.path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=keys).writeheader()
        elif keys != self._fieldnames:
            raise ValueError(
                "metrics の列が途中で変わりました。\n"
                f"  最初の行: {self._fieldnames}\n"
                f"  今回の行: {keys}"
            )
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._fieldnames).writerow(row)
