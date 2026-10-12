"""lesion 実験の記録ファイルの形式 (ファイル名と npz の鍵名)。

**この規約を知っているのはここだけ。** ファイル名だけでなく、**npz の中の鍵の名前**も
書き出しと読み取りが対で持つ (`save_spikes` / `load_spikes` など)。片方だけ変えて
食い違うことがない。

返す値の形は `src/utils/runview.py` の契約 (`Spikes` / `Wiring`) に従う。
**この実験に固有なのはファイル名と npz の鍵名だけで、読めた値の形は全実験共通。**

## develop / akita_soc と規約が違う理由

あちらは `{kind}_{hour}h.npz`、こちらは `{kind}_p{index:03d}.npz`
(ベースラインだけ `{kind}_baseline.npz`)。

1. **時刻をファイル名に埋めると精度を失う。** `f"{hour:g}h"` は probe 間隔が細かいと
   `1.66667e-05h.npz` になり、float の往復で元の時刻に戻らない。
2. **切断前の probe は負の時刻を持つ。** 損傷実験の記録軸は「切断からの経過」であって
   絶対時刻ではない。`{-0.17}h.npz` のような名前は読みにくいうえ並べ替えも壊れる。
3. **develop の解析 CLI に誤読されない。** あちらの glob は `spikes_*h.npz` なので
   `spikes_p003.npz` は掛からない。develop の replot に損傷 run を渡しても
   「見つからない」で止まる。

**時刻の正は npz が持つ** (`record_start_ms`)。ファイル名の index は並べ替えと人間の
目印にしか使わない。これは develop と同じ約束で、違うのは名前の付け方だけ。

**index は `task.record_hours` の添字**。`record_hours: [0..20]` なら p000..p020 で、
3 番目の記録時刻が p002 になる。ベースラインはこの並びに属さない (親 run 由来で、
窓幅も別) ので番号を持たず、`baseline` という名前で区別する。

## 結合構造が 2 つある

損傷実験は**同じ run の中でシナプス本数が変わる**唯一の実験。切断前と切断後で COO が
違うので、`connectivity.npz` を 2 本持つ。

    connectivity_pre.npz   切断前の結合 (全シナプス)
    connectivity.npz       切断後の結合 (生き残ったシナプス)

どちらを使うかは probe の phase で決まる (`views.py` の `Window.wiring()`)。
2 つをまたいで重みを比べるときは `src/utils/analysis/weights.py` の `align_subset_to_coo()` で
生存シナプスへ引き当てる —— **位置で対応づけてはいけない。**
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.runview import MissingData, NeuronTimeseries, Spikes, Wiring

# 記録ファイル名 `<種別>_p<番号>.npz` と `<種別>_baseline.npz`。
PROBE_PATTERN = re.compile(
    r"(?P<kind>[A-Za-z_]+)_(?:p(?P<index>\d+)|baseline)\.npz")

SPIKES = "spikes"
WEIGHTS = "weights"
# 記録窓の先頭で毎ステップ採った全ニューロンの変数。膜電位 V と入力電流 Isyn_rec を
# 別ファイルに書く。task.trace_neuron を指定した run だけが書き、長さは trace_window_s。
# スパイクは同じ窓の spikes の npz にある。
VOLTAGE = "voltage"
CURRENT = "current"

# 結合構造。**2 つあるのが lesion の特徴**で、切断の前後で別ファイルに書く。
POST_CONNECTIVITY_NAME = "connectivity.npz"      # 切断後 (生き残り)
PRE_CONNECTIVITY_NAME = "connectivity_pre.npz"   # 切断前 (全シナプス)

# soma の座標。結合構造と同じく run を通して不変なので 1 回だけ書く。
#
# **`no_space` の run は持たない。** config だけからは復元できない (空間コンポーネントが
# RNG を引く) ので、axes と同じ理由でここに残す。これがあると記録窓の view からも
# 空間の図が描ける。
COORDS_NAME = "coords.npz"

# 伝播遅延 [ms]。**切断後**の結合 (`connectivity.npz`) と index 整合する値ベクトル。
DELAYS_NAME = "delays.npz"

# **切断直後**の重み (親 run から復元して残存シナプスに流し込んだ値)。この run の
# build 直後の重みにあたる。`connectivity.npz` と index 整合。probe の名前規約
# (`weights_p*.npz` / `weights_baseline.npz`) には掛からない。
INITIAL_WEIGHTS_NAME = "weights_initial.npz"

MANIFEST_NAME = "lesion.json"
CUT_NAME = "lesion_cut.npz"
CUT_PROFILE_NAME = "cut_profile.csv"
METRICS_NAME = "metrics.csv"
# probe ごとの RTE (実効結合の損傷前比)。全 probe が揃ってから一度に書く表。
RTE_NAME = "rte.csv"
# run の束 (`compare`) の表。置き場所は `paths.compare_dir()`。
RUNS_TABLE_NAME = "runs.csv"         # 損傷 run 1 本 = 1 行
CONTROL_TABLE_NAME = "control.csv"   # 対照 run 1 本 = 1 行

PHASE_BASELINE = "baseline"
PHASE_POST = "post"

# ベースラインの probe 番号。**この run では測らず、親 run の記録時刻の窓をそのまま置く。**
#
# 負の番号にしてあるのは、**post の番号を `task.record_hours` の添字と一致させる**ため。
# baseline が 0 を取ると `record_hours: [0..20]` が p001..p021 になり、「3 番目の記録時刻」
# と「p003」が食い違う。baseline はそもそも別種 (親 run 由来・窓幅も別) なので、
# 番号を共有せずファイル名も `{kind}_baseline.npz` と分ける。
BASELINE_INDEX = -1
BASELINE_TOKEN = "baseline"

MS_PER_HOUR = 60.0 * 60.0 * 1000.0

# スパイク npz に埋める鍵。**ファイル名から復元しない**ためのもの。
RECORD_START_KEY = "record_start_ms"   # 切断時刻を 0 とする窓の原点 [ms]。baseline は負
RECORD_WINDOW_KEY = "record_window_ms"  # この窓の長さ [ms]。baseline だけ別幅にできる
PHASE_KEY = "phase"


@dataclass(frozen=True)
class ProbeFile:
    index: int
    path: Path


def probe_token(index: int) -> str:
    """probe を指す短い文字列。`p003` / baseline は `baseline`。

    **ファイル名にも図のファイル名にも同じものを使う。** 片方だけ規約が変わると、
    npz と図を突き合わせるのに読み替えが要る。
    """
    index = int(index)
    return BASELINE_TOKEN if index == BASELINE_INDEX else f"p{index:03d}"


def probe_filename(kind: str, index: int) -> str:
    """記録ファイル名を組み立てる。読み取り側と同じ規約を使うための唯一の入口。"""
    return f"{kind}_{probe_token(index)}.npz"


def parse_probe_name(name: str) -> tuple[str, int]:
    """`spikes_p003.npz` -> ("spikes", 3)、`spikes_baseline.npz` -> ("spikes", -1)。

    規約に合わなければ ValueError。
    """
    match = PROBE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"probe ファイル名の規約に合いません: {name}")
    index = BASELINE_INDEX if match["index"] is None else int(match["index"])
    return match["kind"], index


def discover_probes(directory: Path, kind: str) -> list[ProbeFile]:
    """`directory` 直下の probe を index 昇順で返す (baseline は負なので先頭)。

    glob は `p` + 数字と `baseline` に限る。同じ接頭辞を持つ probe 以外の記録を
    拾わないため。
    """
    directory = Path(directory)
    paths = [*directory.glob(f"{kind}_p[0-9]*.npz"),
             *directory.glob(f"{kind}_{BASELINE_TOKEN}.npz")]
    files = [ProbeFile(index=parse_probe_name(path.name)[1], path=path)
             for path in paths]
    return sorted(files, key=lambda item: item.index)


# ======================================================================================
# 結合構造
# ======================================================================================

def save_connectivity(path: Path, row, col, shape) -> None:
    """結合構造を書く。**鍵名 (`row` / `col` / `shape`) を知る場所を 1 つにするための対。**"""
    np.savez_compressed(path, row=np.asarray(row), col=np.asarray(col),
                        shape=np.asarray(shape))


def load_connectivity(path: Path) -> Wiring:
    """`connectivity*.npz` を読んで `Wiring` にする。

    ファイルの中身は `Wiring` そのもの (`row` / `col` / `shape`)。遅延と重みは別ファイル。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"結合情報 {path} が見つかりません。"
            " weights_p*.npz と対になる結合構造が必要です (再実行してください)。"
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
    """記録された COO が**行優先ソート済み**であることを確かめる。

    COO の並びはプロジェクト全体で 1 つ (CLAUDE.md の不変条件 4) だが、それを揃える前に
    作られた run は GeNN の格納順で書かれている。同じネットワークで本数も同じなので、
    **位置で対応づけると黙って別のシナプスに値が乗る。** 読んだ時点で弾く。

    狭義単調増加であることは「行優先かつ (pre, post) の重複なし」と同値。
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


def post_connectivity_path(directory: Path) -> Path:
    return Path(directory) / POST_CONNECTIVITY_NAME


def pre_connectivity_path(directory: Path) -> Path:
    return Path(directory) / PRE_CONNECTIVITY_NAME


# ======================================================================================
# probe 1 つぶん
# ======================================================================================

def save_coords(path: Path, coords) -> None:
    """soma の座標を書く。キー名を知っているのは読み書きのこの対だけ。

    `no_space` の run では呼ばないこと (座標が無いことと、`inf` で埋めた座標を
    保存したことを読む側から区別できなくなる)。
    """
    np.savez_compressed(path, data=np.asarray(coords, dtype=np.float64))


def load_coords(path: Path) -> np.ndarray:
    """`save_coords()` が書いた座標を読む。非有限値は「座標が無い」として扱う。"""
    with np.load(path) as data:
        coords = np.asarray(data["data"], dtype=np.float64)
    if not np.all(np.isfinite(coords)):
        raise MissingData("coords", "座標に有限でない値があります (no_space)")
    return coords


def save_weight_values(path: Path, values: np.ndarray) -> None:
    """重みを値ベクトルとして書く。キー名を知っているのは読み書きのこの対だけ。"""
    np.savez_compressed(path, data=np.asarray(values))


def load_weight_values(path: Path) -> np.ndarray:
    """重み npz から値ベクトル (1D) を読む。その probe の結合と index 整合。"""
    data = np.load(path, allow_pickle=True)
    if "data" not in data.files:
        raise KeyError(f"{path} にキー 'data' がありません。")
    return np.asarray(data["data"], dtype=np.float64).reshape(-1)


def save_delays(path: Path, values) -> None:
    """遅延 [ms] を値ベクトルとして書く。切断後の結合と index 整合。"""
    np.savez_compressed(path, data=np.asarray(values, dtype=np.float64))


def load_delays(path: Path) -> np.ndarray:
    """`save_delays()` が書いた遅延 [ms] を読む。"""
    with np.load(path) as data:
        return np.asarray(data["data"], dtype=np.float64).reshape(-1)


def save_spikes(path: Path, times, ids, *, record_start_ms: float,
                record_window_ms: float, phase: str) -> None:
    """スパイクと、**その窓が何であるか** (原点・窓幅・phase) を同じ npz に書く。

    `times` は**切断軸**の時刻 (切断の瞬間が 0)。GeNN の時計は settle のぶん先を指すので、
    書く側がそのずれを引いてから渡すこと。

    窓幅を窓ごとに持つので、baseline と post で幅が違っても発火率が正しく出る。
    """
    times = np.asarray(times, dtype=np.float64)
    _check_window_span(path, times, record_start_ms, record_window_ms)
    np.savez_compressed(path, times=times, ids=np.asarray(ids),
                        **{RECORD_START_KEY: float(record_start_ms),
                           RECORD_WINDOW_KEY: float(record_window_ms),
                           PHASE_KEY: str(phase)})


def _check_window_span(path: Path, times, record_start_ms: float,
                       record_window_ms: float) -> None:
    """スパイクが窓 `[原点, 原点+幅]` に収まっているか確かめる。

    時間軸を取り違えた窓 (GeNN の時計をそのまま書いた等) は、どこも例外を出さないまま
    **空のラスターとずれた burstiness** になるので、書く時点で落とす。
    """
    if times.size == 0:
        return
    lo = float(record_start_ms)
    hi = lo + float(record_window_ms)
    slack = 1.0  # [ms]。窓の端は dt 1 ステップぶん食い違いうる。
    if times.min() < lo - slack or times.max() > hi + slack:
        raise ValueError(
            f"{path.name}: スパイク時刻 [{times.min():.1f}, {times.max():.1f}] ms が"
            f" 窓 [{lo:.1f}, {hi:.1f}] ms の外です。時間軸が食い違っています。"
        )


def read_window_meta(path: Path) -> tuple[float, float, str]:
    """スパイク npz から `(原点 [ms], 窓幅 [ms], phase)` だけを読む。

    npz は鍵ごとに展開されるので、ここで配列は解かない。
    """
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in (RECORD_START_KEY, RECORD_WINDOW_KEY, PHASE_KEY)
                   if key not in data.files]
        if missing:
            raise KeyError(
                f"{path} に {missing} がありません。"
                " 窓の素性を持たない古い run です (再実行してください)。"
            )
        return (float(data[RECORD_START_KEY]), float(data[RECORD_WINDOW_KEY]),
                str(data[PHASE_KEY]))


def record_hour(record_start_ms: float) -> float:
    """記録窓の原点 [ms] を切断からの経過時間 [h] に直す。

    `Window.hour` はこれで作る。baseline は負になる。
    """
    return record_start_ms / MS_PER_HOUR


def load_spikes(path: Path) -> Spikes:
    """スパイク npz を読み、**窓の原点を引いたローカル時刻**にして返す。

    絶対時刻は返さない (契約の `Spikes.times` はローカル)。絶対時刻を既定にすると、
    アバランチ分割は同じでも burstiness のビン割りが静かにずれる。
    """
    start_ms, _window_ms, _phase = read_window_meta(path)
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        ids = np.asarray(data["ids"])
    return Spikes(times=times - start_ms, ids=ids)


def save_neuron_timeseries(path: Path, values, *, dt: float) -> None:
    """全ニューロンのある変数の時系列 (ニューロン数, ステップ数) を書く。行がグローバル ID。"""
    np.savez_compressed(path, values=np.asarray(values), dt=float(dt))


def load_neuron_timeseries(path: Path) -> NeuronTimeseries:
    with np.load(path) as data:
        return NeuronTimeseries(values=np.asarray(data["values"]), dt=float(data["dt"]))


# ======================================================================================
# 損傷条件の記録
# ======================================================================================

def save_manifest(directory: Path, manifest: dict) -> Path:
    """`lesion.json` を書く。**切断で何が起きたかの記録はこれが正。**"""
    path = Path(directory) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(directory: Path) -> dict:
    return json.loads((Path(directory) / MANIFEST_NAME).read_text(encoding="utf-8"))


def save_cut(directory: Path, per_synapse: dict) -> Path:
    """切断されたシナプスごとの素性を記録する。**何を失ったか**を後から数えられるように。

    切断本数だけでは「ハブの出力を集中的に落とした」のか「弱い結合を薄く広く落とした」
    のかが区別できないので、重み・次数・participation・ハブ役割まで 1 本ごとに残す。
    """
    path = Path(directory) / CUT_NAME
    np.savez_compressed(path, **{name: np.asarray(values)
                                 for name, values in per_synapse.items()})
    return path


def load_cut(directory: Path) -> dict:
    with np.load(Path(directory) / CUT_NAME, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


# ======================================================================================
# 表
# ======================================================================================

def write_table(rows: list[dict], out_path: Path) -> Path:
    """dict の並びを CSV にする。**列は最初の行のキー順**、後の行で増えた分は末尾。

    `MetricsWriter` が「1 行ずつ追記する記録」なのに対し、こちらは**一度に確定する表**。
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
    return out_path


def load_metrics(path: Path) -> pd.DataFrame:
    """`MetricsWriter` が書いた `metrics.csv` を読む。書きと読みの対。"""
    return pd.read_csv(path)


class MetricsWriter:
    """`metrics.csv` を **1 行ずつ追記**する。途中で落ちた run もそこまでが読める。

    ヘッダは**最初の行で確定**する。以降の行でキーが変わると例外。

    **`metrics.csv` は図ではなく記録ファイル**なので、npz と同じくこのモジュールに置く。
    run の途中で書かれ (`run_one.py`)、`load_metrics()` で読まれ (`Series.metrics()`)、
    再解析が作り直す。ここが知っているのは「どのファイルにどういう作法で書くか」だけで、
    列の中身は `analysis/metrics.py` が決める。書く側だけが `save_*` 関数ではなくクラスなのは、
    記録窓をまたいで「最初の行で確定したヘッダ」を持ち続けるため。
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
