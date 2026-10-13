"""ニューロンのグローバルインデックス割り当てと、任意カテゴリ軸を司る NetworkLayout。

**正準グローバルID空間 0..N-1** を唯一の基準とし、その上に任意個数の「軸 (axis)」を貼る。
軸とは長さ N のラベル配列であり、各グローバルIDに1つの値(カテゴリ名 or ソート用の数値)を
対応させる。カテゴライズ (`ids_by` / `ids_where`) とソート (`order_by`) はすべてこの
ラベル表の上で総称的に行われる。

軸は3種類ある:

- **`population` 軸**: このクラスの中核。各グローバルIDがどの GeNN population に属するかで、
  **ローカルID↔グローバルID変換表の実体そのもの**。`__init__` が必ず構築する。
- **`mode` 軸 / `polarity` 軸**: それぞれニューロンの動作モード (RSexci / LTS / …) と
  シナプス極性 (excitatory / inhibitory)。config 由来のメタデータなので、config を知る
  唯一の場所である `from_config` が貼る。この2つは独立で、`mode` は動特性、`polarity` は
  E/I を表す。E/I で切りたい側 (結合・遅延・可視化) は必ず `polarity` を見ること。
- **外部軸**: 構造層・空間モジュール・任意の解析カテゴリなど。`add_axis(name, values)` で
  **外部から与える**。NetworkLayout 自身は空間座標もデータファイルも知らない。
  ネットワークコンポーネント (space / connection / weight / delay) が `describe_axes()`
  で宣言し、NetworkBuilder が生成直後に注入する。

**責務の境界**: このクラスが持つのは正準ID空間・変換表・軸だけである。ニューロンモデルの
パラメータ (`config.neurons[name]`) やニューロン数といった config の内容は保持しない。
それらが必要な呼び出し側は config を直接参照すること。

population のグローバルID集合は**常に昇順**である (`ids_by` が `np.nonzero` で返すため
構造的に保証される)。GeNN は発信ニューロン順にシナプスを格納するため、`local_to_global`
と NetworkBuilder の疎/密経路一致がこの不変条件に依存している。一方、**どの軸にも連続性の
要求も特権も無い**。切り出しは常に fancy-index (`np.ix_`) で行う。

外部軸はビルド時にしか存在しないため、解析側で復元できるよう `save_axes()` /
`load_axes_file()` で npz に永続化する。ファイル名は `AXES_NAME`、**どの run ディレクトリへ
置くか**は呼び出し側が決める。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# random 割当用のシード派生オフセット。行列生成用 RNG(NetworkBuilder.self.rng)とは
# 別ストリームにし、E/I 割当と行列値の相関を避けるために seed に加算する定数。
# 外部軸を書き出す npz の名前。`save_axes()` と `load_axes_file()` が対で使う。
AXES_NAME = "layout_axes.npz"

_ASSIGN_SEED_OFFSET = 104729

# 誤上書きを防ぐため overwrite=True を要求する軸 = **config だけから再導出できる軸**。
# `population` は変換表の実体、`mode` / `polarity` は config の宣言そのもの。よってこの
# リストがそのまま「永続化しなくてよい軸」の定義になる (axes_to_dict が除外し、復元時は
# from_config が再構築する)。config 由来の軸を足すときはここに追記すること。
_AUTO_AXES = ("population", "mode", "polarity")


class NetworkLayout:
    """正準グローバルID空間の上に、任意個数のカテゴリ/ソート軸を持つレイアウト。"""

    def __init__(
        self,
        population_labels: Sequence[str],
        order: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            population_labels: 長さ N の配列。各グローバルIDが属する population 名。
                各IDがちょうど1つの population に属することが構造的に保証されるため、
                メンバーシップの重複や欠落を作れない。
            order: 正準 population 順(通常は config 宣言順)。`split_global_to_local`
                が返す dict のキー順を決める。省略時はグローバルID空間での初出順。

        Raises:
            ValueError: order が population_labels に現れる名前の集合と一致しない場合。
        """
        labels = self._normalize("population", population_labels)
        if labels.ndim != 1:
            raise ValueError(f"population_labels は1次元配列である必要があります (got {labels.shape})。")
        self._total: int = len(labels)

        # --- 軸テーブル (長さ N のラベル配列。これが唯一の真実の源) ---
        # dtype は numpy 推論に任せる (文字列なら '<U*'、数値なら float/int)。
        # object dtype を避けることで npz 直列化・np.unique・lexsort が素直に効く。
        self._axes: Dict[str, np.ndarray] = {"population": labels}
        # ids_by() の転置索引キャッシュ (ラベル表からの派生ビュー)。
        self._ids_by_cache: Dict[str, Dict[Any, np.ndarray]] = {}

        present = self._values_appearance(labels)
        if order is None:
            self._pop_order: List[str] = list(present)
        else:
            self._pop_order = list(order)
            if set(self._pop_order) != set(present):
                raise ValueError(
                    f"order {self._pop_order} が population_labels に現れる名前 "
                    f"{list(present)} と一致しません。"
                )

    # ==================================================================
    # 構築
    # ==================================================================
    @classmethod
    def from_config(cls, config) -> "NetworkLayout":
        """AppConfig から population / mode / polarity 軸を決定論的に構築する。

        **config を参照する唯一の場所**。`__init__` は config を知らないので、config
        由来のメタデータ (mode / polarity) を軸として貼るのはここの責務。構造層・空間
        モジュール等の軸は **外部から `add_axis()` で与える**。

        ここで貼る軸は `_AUTO_AXES` と一致する。永続化されないのはこのメソッドが
        config から再構築するからであり、両者は必ず対応していなければならない。

        `config.layout.assignment` は population のグローバルID割当方式:
          - "sequential": config 宣言順に連番
          - "random": 全体比率でランダムに散らす (seed 決定論的)

        **既定値はここに持たない**。未指定を無言で埋めると、その値が config にも保存物にも
        残らないまま E/I 配置が決まってしまい、既定を変えた日に過去の run が別の割当として
        読み直される。既定の適用は `ConfigManager._materialize_layout()` の仕事で、そこを
        通れば assignment は実値になっている。
        """
        pop_names = list(config.neurons.keys())
        counts = [config.neurons[name].num for name in pop_names]
        total = sum(counts)

        layout_cfg = getattr(config, "layout", None)
        assignment = getattr(layout_cfg, "assignment", None) if layout_cfg is not None else None
        if assignment is None:
            raise ValueError(
                "config.layout.assignment がありません。ConfigManager.resolve() /"
                " load_resolved() を通せば実値が入ります (手組み config の場合は"
                " layout.assignment に 'sequential' か 'random' を明示してください)。"
            )

        if assignment == "sequential":
            membership: Dict[str, np.ndarray] = {}
            offset = 0
            for name, c in zip(pop_names, counts):
                membership[name] = np.arange(offset, offset + c)
                offset += c
        elif assignment == "random":
            base_seed = getattr(config.simulation, "seed", None)
            if base_seed is None:
                # seed 無しで RandomState(None) を使うと OS エントロピーで初期化され、
                # 同じ config から毎回違う割当が出る。そうなると保存済み config からの復元が
                # 実行時と異なる E/I 割当を解析側へ返し、結果が静かに壊れる。
                raise ValueError(
                    "layout.assignment='random' には simulation.seed が必須です "
                    "(seed が無いと割当が再現できず、保存済み config からレイアウトを"
                    "復元できません)。seed を指定するか assignment='sequential' にしてください。"
                )
            assign_seed = (int(base_seed) + _ASSIGN_SEED_OFFSET) % (2 ** 32)
            rng = np.random.RandomState(assign_seed)
            perm = rng.permutation(total)
            membership = {}
            offset = 0
            for name, c in zip(pop_names, counts):
                membership[name] = np.sort(perm[offset:offset + c])
                offset += c
        else:
            raise ValueError(f"未知の layout.assignment: {assignment!r} (sequential | random)")

        # メンバーシップを population 軸のラベル配列へ畳み込む(これが変換表の実体)。
        pop_labels = np.empty(total, dtype=object)
        mode_labels = np.empty(total, dtype=object)
        polarity_labels = np.empty(total, dtype=object)
        for name in pop_names:
            pop_labels[membership[name]] = name
            mode_labels[membership[name]] = getattr(config.neurons[name], "mode", None) or ""
            polarity_labels[membership[name]] = cls._polarity_of(config, name)

        layout = cls(pop_labels, order=pop_names)
        # mode / polarity は config 由来のメタデータ。__init__ ではなくここで貼る。
        layout.add_axis("mode", mode_labels, overwrite=True)
        layout.add_axis("polarity", polarity_labels, overwrite=True)
        return layout

    @staticmethod
    def _polarity_of(config, name: str) -> str:
        """population の E/I 極性を config から読む。

        `mode` は動特性 (RSexci / LTS / …) を表すので極性の情報源にしない。E/I は
        `neurons.<name>.polarity` に明示的に書くこと。推測はしない
        (推測すると E として結合を張りながら GeNN には I を渡す、といった食い違いが
        無言で成立してしまう)。
        """
        polarity = getattr(config.neurons[name], "polarity", None)
        if not polarity:
            raise ValueError(
                f"population {name!r} に polarity がありません。"
                f" config の neurons.{name} に polarity: excitatory | inhibitory を"
                f" 追加してください (mode は動特性であり E/I の情報源にはなりません)。"
            )
        return str(polarity)

    # ==================================================================
    # 軸の外部定義
    # ==================================================================
    def add_axis(self, name: str, values: Sequence[Any], *, overwrite: bool = False) -> None:
        """カテゴライズ/ソートの軸を外部から追加する。

        Args:
            name: 軸名 (例: "layer", "module", "depth")。
            values: 長さ N の配列。カテゴリ名(文字列)でも、ソート基準の数値でもよい。
            overwrite: 既存の同名軸を上書きする。自動軸 (population/mode) の
                上書きには必ず必要。

        Raises:
            ValueError: 長さが N と一致しない / 既存軸を overwrite なしで上書きしようとした /
                dtype=object を具体的な dtype に解決できなかった。
        """
        arr = self._normalize(name, values)
        if arr.ndim != 1 or len(arr) != self._total:
            raise ValueError(
                f"軸 {name!r} の値は長さ {self._total} の 1次元配列である必要があります "
                f"(got shape {arr.shape})。"
            )
        if name in self._axes and not overwrite:
            kind = "自動軸" if name in _AUTO_AXES else "既存軸"
            raise ValueError(
                f"{kind} {name!r} は既に存在します。上書きするなら overwrite=True を指定してください。"
            )
        self._axes[name] = arr
        self._ids_by_cache.pop(name, None)

    @staticmethod
    def _normalize(name: str, values: Sequence[Any]) -> np.ndarray:
        """ラベル配列を具体的な dtype ('<U*' / 数値) に正規化する。

        pandas の文字列列 (`df["Layer"].to_numpy()`) などは dtype=object で渡ってくるが、
        object 配列は npz に `allow_pickle=False` で保存できず (= 永続化した軸を
        保存した軸を読み戻せない)、`np.unique` / `lexsort` も遅い。要素から dtype を
        再推論して解決する。
        """
        arr = np.asarray(values)
        if arr.dtype != object:
            return arr
        resolved = np.asarray(arr.tolist())
        if resolved.dtype == object:
            raise ValueError(
                f"軸 {name!r} の dtype を解決できませんでした (要素の型が混在しています)。"
                " 軸の値は文字列か数値で揃えてください。"
            )
        return resolved

    def drop_axis(self, name: str) -> None:
        """軸を削除する。自動軸は削除できない。"""
        if name in _AUTO_AXES:
            raise ValueError(f"自動軸 {name!r} は削除できません。")
        if name not in self._axes:
            raise KeyError(f"未知の軸: {name!r} (利用可能: {list(self._axes)})")
        del self._axes[name]
        self._ids_by_cache.pop(name, None)

    def axes(self) -> List[str]:
        """定義済みの軸名一覧。"""
        return list(self._axes)

    def has_axis(self, name: str) -> bool:
        return name in self._axes

    def _require_axis(self, name: str) -> np.ndarray:
        if name not in self._axes:
            raise KeyError(f"未知の軸: {name!r} (利用可能: {list(self._axes)})")
        return self._axes[name]

    # ==================================================================
    # カテゴライズ
    # ==================================================================
    def labels(self, axis: str) -> np.ndarray:
        """指定軸の長さ N のラベル配列(コピー)を返す。"""
        return self._require_axis(axis).copy()

    def ids_by(self, axis: str) -> Dict[Any, np.ndarray]:
        """{カテゴリ値: グローバルID配列(昇順)} を返す。

        キーの順序は**グローバルID空間での初出順**(層順に整列したデータなら層順になる)。

        戻り値は dict も ID 配列も**呼び出し側専用のコピー**。キーを足しても消しても、
        `ids += offset` のように配列を in-place で書き換えても、レイアウトには影響しない
        (内部の実体を渡すと、`population` 軸の配列はローカルID↔グローバルID変換表そのもの
        なので、何気ない加工が変換表と昇順不変条件を静かに壊す)。

        `np.nonzero` によるグループ分け自体はキャッシュされるので、繰り返し呼んでも
        再計算はされない。かかるのはコピーの分だけ。
        """
        return {value: ids.copy() for value, ids in self._grouped(axis).items()}

    def _grouped(self, axis: str) -> Dict[Any, np.ndarray]:
        """`ids_by` の内部キャッシュ(実体)を返す。

        **絶対に外へ漏らさないこと**。返した配列を加工せず、そこから新しい配列を作る
        (fancy-index 等) 内部利用に限る。
        """
        if axis in self._ids_by_cache:
            return self._ids_by_cache[axis]
        arr = self._require_axis(axis)
        groups: Dict[Any, np.ndarray] = {}
        for value in self._values_appearance(arr):
            groups[value] = np.nonzero(arr == value)[0]
        self._ids_by_cache[axis] = groups
        return groups

    def ids_where(self, **filters: Any) -> np.ndarray:
        """指定軸の値がすべて一致するグローバルIDを昇順で返す。

        例: ids_where(layer="IN2", mode="excitatory")
        """
        mask = np.ones(self._total, dtype=bool)
        for name, value in filters.items():
            mask &= (self._require_axis(name) == value)
        return np.nonzero(mask)[0]

    def values(self, axis: str, order: str = "appearance") -> List[Any]:
        """軸に現れる値の一覧。

        Args:
            order: "appearance" (グローバルID順の初出順) または "sorted" (値でソート)。
        """
        arr = self._require_axis(axis)
        if order == "appearance":
            return self._values_appearance(arr)
        if order == "sorted":
            return list(np.unique(arr))
        raise ValueError(f"未知の order: {order!r} (appearance | sorted)")

    @staticmethod
    def _values_appearance(arr: np.ndarray) -> List[Any]:
        """配列に現れる値を初出順で返す。"""
        _, first_idx = np.unique(arr, return_index=True)
        return list(arr[np.sort(first_idx)])

    # ==================================================================
    # ソート
    # ==================================================================
    def order_by(self, *axes: str, ascending: bool = True) -> np.ndarray:
        """指定軸を優先順に用いて全グローバルIDを並べ替える permutation を返す。

        先に指定した軸ほど優先度が高い。同値はグローバルID昇順で安定に決まる。
        文字列軸は整数コード化してから `np.lexsort` にかけるので、文字列軸と数値軸を
        混在させてよい。

        例: `order = layout.order_by("layer", "mode")` → `W[np.ix_(order, order)]` で
        層→種別の順にブロック化した行列が得られる。
        """
        if not axes:
            raise ValueError("order_by には少なくとも1つの軸名が必要です。")
        return self._lexsort(np.arange(self._total), axes, ascending)

    def rank_by(self, *axes: str, ascending: bool = True) -> np.ndarray:
        """各グローバルIDが `order_by(*axes)` の何番目に来るかを返す (0 始まり)。

        `order_by` の逆写像。`order_by` が「表示順に並べたID列」を返すのに対し、こちらは
        「そのIDの表示位置」を引ける長さ N の配列を返す。ラスターの y 座標のように
        **グローバルIDを表示位置へ写したい**場合に使う (dict と Python ループを使わずに
        `rank[ids]` の fancy-index で一括変換できる)。
        """
        order = self.order_by(*axes, ascending=ascending)
        rank = np.empty(self._total, dtype=np.int64)
        rank[order] = np.arange(self._total, dtype=np.int64)
        return rank

    def sort_ids(self, ids: Sequence[int], *axes: str, ascending: bool = True) -> np.ndarray:
        """グローバルIDの部分集合を指定軸で並べ替えて返す。

        `order_by` が全体の permutation を返すのに対し、こちらは与えたID集合そのものを
        並べ替えた配列を返す。
        """
        if not axes:
            raise ValueError("sort_ids には少なくとも1つの軸名が必要です。")
        ids_arr = np.asarray(ids, dtype=np.int64)
        return ids_arr[self._lexsort(ids_arr, axes, ascending)]

    def _lexsort(self, ids: np.ndarray, axes: Tuple[str, ...], ascending: bool) -> np.ndarray:
        """ids に対する、axes 優先順の並べ替えインデックスを返す。"""
        codes = []
        for name in axes:
            arr = self._require_axis(name)[ids]
            if np.issubdtype(arr.dtype, np.number):
                key = arr
            else:
                # 文字列などは整数コード化する (lexsort は object/str を扱えないため)。
                _, key = np.unique(arr, return_inverse=True)
            if not ascending:
                # bool/符号なし整数は単項マイナスで折り返すため符号付きに寄せる。
                if key.dtype == bool or np.issubdtype(key.dtype, np.unsignedinteger):
                    key = key.astype(np.int64)
                key = -key
            codes.append(key)
        # np.lexsort は **最後のキーが最優先**。最低優先度に位置(=グローバルID昇順)を
        # 置き、以降 axes を逆順に積むことで axes[0] が最優先になる。
        keys = [np.arange(len(ids))] + [codes[i] for i in reversed(range(len(codes)))]
        return np.lexsort(tuple(keys))

    # ==================================================================
    # population 軸 (ローカルID↔グローバルID変換表)
    # ==================================================================
    @property
    def total_neurons(self) -> int:
        return self._total

    def global_indices(self, name: str) -> np.ndarray:
        """population のグローバルID集合。**常に昇順**(連番/散在いずれも)。

        population ローカル index i がこの配列の i 番目に対応する。これがローカルID↔
        グローバルID変換表の実体だが、返るのは**コピー**なので自由に書き換えてよい。
        """
        return self._indices_of(name).copy()

    def _indices_of(self, name: str) -> np.ndarray:
        """`global_indices` の内部キャッシュ(実体)。`_grouped` 同様、外へ漏らさないこと。"""
        try:
            return self._grouped("population")[name]
        except KeyError:
            raise KeyError(f"未知の population: {name!r} (利用可能: {self._pop_order})") from None

    # ==================================================================
    # global↔local 変換 (散在集合対応の fancy-index。simulator/base_loader が委譲)
    # ==================================================================
    # **正準 population 順 (canonical order)**
    #
    #   `_pop_order` が唯一の定義であり、`from_config` 経由なら config.neurons の宣言順に
    #   一致する。`split_global_to_local` が返す dict のキーは必ずこの順に並ぶので、順序が
    #   必要な呼び出し側は `list(split_global_to_local(...))` で取得できる。
    #
    #   population 名の一覧そのものが欲しいだけなら config (`config.neurons`) を直接読むこと。
    #   layout が config の写しを再配布すると、同じ情報の入手経路が2本になって
    #   「どちらが正か」が曖昧になる。layout が答えるのは *ID空間の話* だけでよい。
    #   `ids_by("population")` のキー順は初出順であって正準順ではない点に注意。
    def split_global_to_local(self, global_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """(total_neurons,) のグローバル配列を population 毎の配列に分割する。

        返る dict のキーは**正準 population 順**(上記参照)。
        """
        if len(global_arr) != self._total:
            raise ValueError(
                f"Input data size {len(global_arr)} does not match total neurons {self._total}"
            )
        # fancy-index の結果は新しい配列なので、実体 (_indices_of) をそのまま使ってよい。
        return {name: global_arr[self._indices_of(name)] for name in self._pop_order}

    def merge_local_to_global(
        self, local_dict: Dict[str, np.ndarray], dtype=np.float32
    ) -> np.ndarray:
        """population 毎の配列を (total_neurons,) のグローバル配列に結合する。"""
        global_data = np.zeros(self._total, dtype=dtype)
        for name, local_data in local_dict.items():
            indices = self._indices_of(name)   # 代入先の添字にしか使わないので実体で良い
            local_data = np.asarray(local_data)
            if len(local_data) != len(indices):
                raise ValueError(
                    f"population {name!r} のデータ長 {len(local_data)} が "
                    f"ニューロン数 {len(indices)} と一致しません。"
                )
            global_data[indices] = local_data
        return global_data

    def local_to_global(self, name: str, local_ids: np.ndarray) -> np.ndarray:
        """population ローカルインデックスをグローバルIDに変換する。

        local i は global_indices[i](常に昇順)に対応する。
        """
        # fancy-index の結果は新しい配列なので、実体をそのまま引いてよい。
        return self._indices_of(name)[np.asarray(local_ids)]

    # ==================================================================
    # 永続化 (外部軸はビルド時にしか存在しないため、解析側のために保存する)
    # ==================================================================
    def axes_to_dict(self) -> Dict[str, np.ndarray]:
        """永続化すべき軸(=外部から与えられた軸)を返す。

        `population` / `mode` は config.yaml から決定論的に再導出できるため含めない。
        """
        return {
            name: arr.copy()
            for name, arr in self._axes.items()
            if name not in _AUTO_AXES
        }

    def load_axes(self, axes: Dict[str, np.ndarray]) -> None:
        """`axes_to_dict()` で保存した軸を復元する(既存の同名軸は上書き)。"""
        for name, values in axes.items():
            self.add_axis(name, values, overwrite=True)

    def save_axes(self, path: Path | str) -> Optional[Path]:
        """外部軸を npz として `path` に書き出す。保存すべき軸が無ければ何もせず None。

        **置き場所 (どの run ディレクトリか) は呼び出し側が決める。** このクラスが知って
        いるのは「軸をどう npz に直列化するか」と、その名前 (`AXES_NAME`) までで、
        `outputs/` のディレクトリ規約は持たない。

        `add_axis` が dtype=object を具体 dtype へ正規化しているのは、この npz を
        `allow_pickle=False` で読み書きできるようにするためである。
        """
        axes = self.axes_to_dict()
        if not axes:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **axes)
        return path

    def load_axes_file(self, path: Path | str) -> None:
        """`save_axes()` が書いた npz から外部軸を復元する(既存の同名軸は上書き)。"""
        with np.load(Path(path), allow_pickle=False) as data:
            self.load_axes({name: data[name] for name in data.files})
