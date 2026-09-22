import os
import inspect
import itertools
import warnings
import numpy as np
import pygenn
from dataclasses import dataclass
from typing import Dict, Any, NamedTuple, Tuple
from pathlib import Path
import sys

# CUDAバックエンドを有効にするため、未設定の場合はデフォルトパスを補完する
if not os.environ.get("CUDA_PATH"):
    for _candidate in ["/usr/local/cuda", "/usr/cuda"]:
        if Path(_candidate).exists():
            os.environ["CUDA_PATH"] = _candidate
            break

# config の backend 値 -> GeNN のバックエンド名。config 側を "cpu" という短い名前にしてあるのは
# メイン config に人が書く値だから。GeNN の実装名 (single_threaded_cpu) はここで吸収する。
_GENN_BACKEND = {"cuda": "cuda", "cpu": "single_threaded_cpu"}

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pydanticの設定モデルと、コンポーネントを動的ロードするレジストリをインポート
from src.core.config_manager import AppConfig
from src.core.layout import NetworkLayout
from src.core.registry import AREA_MODELS, SPATIAL_MODELS, CONNECTION_MODELS, WEIGHT_MODELS, DELAY_MODELS, NEURON_MODELS, SYNAPSE_MODELS, PLASTICITY_MODELS

# GeNN が生成する `<モデル名>_CODE` を集める親ディレクトリ。
#
# **実行環境の都合であって実験条件ではない**ので config には持たせない。
# `NetworkBuilder(code_gen_dir=...)` を省略するとここへ出る。
GENN_CODE_DIR = "genn_code"


class GlobalCOO(NamedTuple):
    """グローバルID空間での結合を COO で表したもの。**ビルド以降の唯一の受け渡し形式**。

    疎生成経路と密生成経路の違いを吸収する場所はここ 1 箇所だけで、
    `NetworkBuilder.global_coo()` を通った後のコード (utils / scripts / test) は
    もう `is_sparse` を見ない。密な (N, N) が本当に要るのは行列画像を描くときだけなので、
    その復元は描画関数が自分の中でやる (`src/utils/plotting/matrices.py`)。

    **不変条件: 行優先ソート済み** (row 昇順、同一 row 内で col 昇順)。
    密経路は `np.nonzero(mask)` が、疎経路は各コネクタの `generate_sparse()` が保証する。
    GeNN はシナプスを送信元ニューロン順に格納するので、`_pair_coo()` の集団ローカル COO も
    この順序に乗っている。
    """
    row: np.ndarray      # int32, グローバル pre ID
    col: np.ndarray      # int32, グローバル post ID
    weights: np.ndarray  # float64
    delays: np.ndarray   # float64 [ms]
    shape: Tuple[int, int]

    @property
    def num_synapses(self) -> int:
        return int(self.row.size)


@dataclass(frozen=True)
class SynapseIndex:
    """1つのシナプス集団 (src_pop -> tgt_pop) の接続インデックスを保持する。

    GeNN のシナプス変数は「集団ローカルの (pre, post) 行優先ソート順」で読み書きされる
    (pygenn が set_sparse_connections で lexsort し、値の getter は行優先で返す)。
    ここに保持する配列はその順序と一致しており、`simulator.pull_synapse` 系はこれを使って
    結合マスクを再走査せずに値をグローバルIDへ散布できる。

    `global_positions` は **`global_coo()` の中での位置**。集団は `global_coo()` からの
    ブール選択で切り出されるので (`_pair_coo`)、GeNN から取り出した値はこの位置へ
    そのまま散布すれば `global_coo()` と同じ並びになる —— 並べ替えは要らない。
    """
    src_name: str
    tgt_name: str
    local_src: np.ndarray        # int32, 集団ローカル pre index
    local_tgt: np.ndarray        # int32, 集団ローカル post index
    global_src: np.ndarray       # int32, グローバル pre ID
    global_tgt: np.ndarray       # int32, グローバル post ID
    global_positions: np.ndarray # int64, global_coo() の中での位置

    @property
    def num_synapses(self) -> int:
        return int(self.local_src.size)


class NetworkBuilder:
    def __init__(self, config: AppConfig, model_name: str = "SNN_Model", code_gen_dir: str | None = None):
        self.config = config
        self.rng = np.random.RandomState(config.simulation.seed)

        # GeNN が生成する <model_name>_CODE を置く親ディレクトリ。`Simulator.setup()` が使う。
        # **省略するとプロジェクトの既定 (`genn_code/`)。** 生成コードがリポジトリ直下へ
        # 散らばらないよう、行き先は 1 か所に集める。
        self.code_gen_dir = code_gen_dir or GENN_CODE_DIR

        # config のニューロン宣言順に連番でグローバルインデックスを割り当てる決定論的レイアウト。
        # RandomState を消費しないため、GeNN ビルドなしでも from_config だけで再現できる。
        self.layout = NetworkLayout.from_config(config)
        self.total_neurons = self.layout.total_neurons

        # backend は config が決める。ここで既定値を持たないのは seed / layout.assignment と
        # 同じ理由で、無言で埋めるとその値が記録に残らないまま結果が決まってしまうから
        # (GeNN のデバイス RNG はバックエンドごとに別の乱数列を出すので、backend が違えば
        #  同じ seed でも別のスパイク列になる)。既定の適用と "auto" の解決は
        # `ConfigManager._materialize_backend()` の仕事で、そこを通れば実値になっている。
        backend = getattr(self.config.simulation, "backend", None)
        if backend is None:
            raise ValueError(
                "config.simulation.backend がありません。ConfigManager.resolve() を通せば"
                " 実値が入ります (backend を記録していない古い config.yaml を再実行する"
                " 場合は simulation.backend に 'cuda' か 'cpu' を明示してください)。"
            )
        if backend not in _GENN_BACKEND:
            raise ValueError(
                f"未知の simulation.backend: {backend!r} (cuda | cpu)。"
                " 'auto' は ConfigManager が実値へ解決するので、ここには現れないはずです。"
            )
        _backend = _GENN_BACKEND[backend]
        print(f"[NetworkBuilder] backend = {_backend}  [config.simulation.backend={backend!r}]")
        # optimize_code=True は生成コードを -O3 -ffast-math でコンパイルさせる
        # (GeNN の既定は最適化フラグ無し = -O0。この規模で約 7 倍の差になる)。
        # **コンパイルフラグは model.sha に入らない。** ここを変えても再生成は走らないので、
        # 辻褄合わせは `GeNNSimulator.setup()` の always_rebuild=True が担う。
        self.genn_model = pygenn.GeNNModel("double", model_name, time_precision="double",
                                           backend=_backend, optimize_code=True)
        self.genn_model.dt = self.config.simulation.dt
        # self.genn_model.batch_size = self.config.task.batch_size

        # GeNN のデバイス RNG シード。config.simulation.seed は NetworkBuilder の
        # np.random.RandomState (= ネットワーク構造の生成) にしか使われておらず、
        # GeNN 側は既定値 0 (= 実行ごとにランダムな種) のままだった。そのため
        # escape noise (gennrand_uniform) を使うモデルは同一 seed でも実行ごとに
        # 別のスパイク列になっていた (CPU で 3422/4437/4814 spikes と実測)。
        # ここで明示的に渡すことでシミュレーション全体が再現可能になる。
        # ※ GeNN では seed=0 が「ランダムな種」を意味するため 0 は避ける。
        # seed: null (= 明示的にランダム、configs/test.yaml など) は np.random.RandomState(None)
        # と揃えて GeNN 側もランダム (=0) のままにする。
        _seed = self.config.simulation.seed
        if _seed is None:
            self.genn_model.seed = 0
        else:
            _genn_seed = int(_seed)
            self.genn_model.seed = _genn_seed if _genn_seed != 0 else 1

        self._component_lifeline = []
        self.global_coords = None
        # 密生成経路の作業領域。**生成経路の内部都合なので private**。
        # ビルド以降にグローバルな結合が要るコードは `global_coo()` を使うこと
        # (疎/密の分岐がそこ 1 箇所に閉じる)。
        self._global_mask = None
        self._global_weights = None
        self._global_delays = None
        # ニューロンを配置し軸索を閉じ込める 2D 領域 (_generate_global_matrices で構築)
        self.area = None
        # 結合コンポーネントの実体。生成後も残すのは、行列に落ちない副産物 (axon_growth の
        # 軸索折れ線など) を可視化・解析から読めるようにするため。area と同じ扱い。
        self.connection = None
        # 疎生成経路で使う COO (すべて index 整合の 1D 配列)。密経路では None のまま。
        # これも生成経路の内部都合。外へ出すのは `global_coo()` に正規化した後だけ。
        self._sparse_rows = None
        self._sparse_cols = None
        self._sparse_weights = None
        self._sparse_delays = None
        # 疎/密どちらで生成しても同じ形になる正規化済み COO (`global_coo()` のキャッシュ)
        self._global_coo: GlobalCOO | None = None
        # 疎経路での グローバルID -> (集団コード, ローカルindex) 索引表 (遅延構築)
        self._index_table = None
        # "src_to_tgt" -> SynapseIndex。_build_synapses が GeNN へ登録した接続順を記録する。
        self.synapse_index: Dict[str, SynapseIndex] = {}
        # "src_to_tgt" -> fan-in 正規化に使うシナプス数。None なら実際の本数を使う (通常)。
        # 損傷実験で「切断してもゲインは変えない」を実現するために使う。
        self._fan_in_reference: Dict[str, int] | None = None

    def build(self, rec_spike: bool = True,
              transform_coo=None,
              preserve_fan_in_scale: bool = False) -> Tuple[pygenn.GeNNModel, NetworkLayout]:
        """GeNN モデルを構築する。

        Args:
            transform_coo: 生成直後のグローバル COO を差し替えるフック
                (`GlobalCOO -> GlobalCOO`)。**乱数の消費が終わった後、GeNN へ登録する前**に
                1 度だけ呼ばれる。損傷実験 (シナプスの構造的除去) と重み復元の入口。
                None なら従来と完全に同一の経路 (分岐も乱数の消費も一切変わらない)。
                詳細は `replace_global_coo()` の docstring。
            preserve_fan_in_scale: `transform_coo` でシナプスを削っても、可塑性モデルの
                fan-in 正規化 (`normalize_gmax_by_fan_in`) の分母を**変換前の本数**に
                固定する。損傷が実効ゲインを動かす交絡を止めるためのもの。
        """
        print("=== ネットワーク構築 (Network Building) ===")
        # 1. 全ニューロン一括での座標・グローバル行列生成 (レイアウトは __init__ で確定済み)
        self._generate_global_matrices()
        if transform_coo is not None:
            self.replace_global_coo(transform_coo(self.global_coo()),
                                    preserve_fan_in_scale=preserve_fan_in_scale)
        elif preserve_fan_in_scale:
            raise ValueError(
                "preserve_fan_in_scale は transform_coo と一緒にしか意味がありません"
                " (変換しなければ fan-in は元から変わらない)。"
            )
        # 2. GeNNへのニューロン・シナプスの登録
        self._build_neuron_populations()
        self._build_synapses()
        # 3. 入出力ポートのハードウェア的構築とメタデータ生成
        self._build_input_ports()
        self._build_output_ports(rec_spike)

        return self.genn_model, self.layout

    def _component_classes(self):
        network = self.config.network
        return (
            AREA_MODELS.get(network.area.profile_name),
            SPATIAL_MODELS.get(network.space.profile_name),
            CONNECTION_MODELS.get(network.connection.profile_name),
            WEIGHT_MODELS.get(network.weight.profile_name),
            DELAY_MODELS.get(network.delay.profile_name),
        )

    def _use_sparse(self) -> bool:
        """疎生成経路を使うかどうかを config と各コンポーネントの対応状況から決める。

        入力値:
          "auto"(既定): 結合/重み/遅延の3段すべてが疎対応なら疎。1段でも非対応なら密。
          "force"     : 疎を必須とし、非対応クラス名を挙げて即エラー(大規模実行で
                        20分走ってから OOM kill されるのを防ぐ)。
          "off"       : 常に密(過去のネットワーク実現を再現したいとき)。

        記録値 ("on" / "off"): 保存済み config.yaml をそのまま再実行するための入口。
        `_generate_global_matrices()` が決定結果をここへ焼き込むため、記録された config は
        必ずこの形になっている。"on" は「疎で走った」の意なので `force` と同じ扱い
        (非対応ならエラー)にし、黙って密へ落ちて別の実現になるのを防ぐ。
        """
        mode = getattr(self.config.network, "sparse", "auto")
        if mode not in ("auto", "force", "off", "on"):
            raise ValueError(
                f"network.sparse は 'auto' / 'force' / 'off' / 'on' のいずれかです (got {mode!r})。"
            )
        if mode == "off":
            return False
        if mode == "on":
            mode = "force"

        # area と space は行列を作らないので疎/密の判定対象外。
        _, _, connect_cls, weight_cls, delay_cls = self._component_classes()
        unsupported = [
            cls.__name__
            for cls in (connect_cls, weight_cls, delay_cls)
            if not getattr(cls, "supports_sparse", False)
        ]
        if not unsupported:
            return True
        if mode == "force":
            raise ValueError(
                "network.sparse='force' ですが、疎生成に対応していないコンポーネントがあります: "
                f"{', '.join(unsupported)}。'auto' にするか、疎対応のプロファイルを選んでください。"
            )

        # auto で密へ落ちたケース。密は N×N を 3 本 (mask int8 + weights/delays float32 =
        # 9N² バイト) 確保し、生成中は距離行列・確率行列 (float64) でさらに数倍になる。
        # 全結合のように「密のほうが軽い」構成なら意図どおりだが、その場合は off を明示して
        # 記録に残すべきなので、黙って落ちずに知らせる。
        dense_bytes = self.total_neurons ** 2 * 9
        warnings.warn(
            f"network.sparse='auto' ですが疎生成に非対応のコンポーネントがあるため密経路を使います: "
            f"{', '.join(unsupported)}。"
            f" 密な N×N 行列に約 {dense_bytes / 2**30:.2f} GiB (生成中のピークはこの数倍) を確保します。"
            " 意図的に密を選んでいる場合 (全結合など、疎より軽くなる構成) は"
            " network.sparse='off' を明示してください。",
            stacklevel=2,
        )
        return False

    @property
    def is_sparse(self) -> bool:
        """疎生成経路で構築されたか。真実の在り処は `config.network.sparse`。

        `_generate_global_matrices()` が決定結果を config へ焼き込むので、ビルド後は
        ここを見れば分岐が分かる。`_sparse_rows is not None` のような副作用からの推測は
        しないこと (真実が 2 箇所になる)。
        """
        return self.config.network.sparse == "on"

    def _generate_global_matrices(self):
        """全ニューロンの座標と、グローバルな結合情報を生成する"""
        print("  Generating Global Coordinates and Matrices...")
        use_sparse = self._use_sparse()

        # 決定結果を config へ焼き込む。以後この分岐を見たい人は config.network.sparse を
        # 読む (seed / backend / layout.assignment と同じく、実際にどちらで走ったかを
        # 記録に残すため)。`_sparse_rows is not None` のような副作用からの推測はしない。
        self.config.network.sparse = "on" if use_sparse else "off"

        if use_sparse:
            self._generate_global_sparse()
        else:
            self._generate_global_dense()

        # ここで疎/密を 1 つの COO に正規化する。以降 (GeNN 登録・可視化・解析) は
        # この形しか見ないので、生成経路の違いはこの行より先へは漏れない。
        self._global_coo = self._normalize_to_coo()

    def _normalize_to_coo(self) -> GlobalCOO:
        """生成直後のグローバル結合を `GlobalCOO` へ正規化する (`global_coo()` の実装)。

        密経路は「マスクが非零のところ」を実結合とする (重みが 0 でも結合はある)。
        `np.nonzero` の返す順序が行優先なので、`GlobalCOO` の不変条件はそのまま満たされる。
        """
        shape = (self.total_neurons, self.total_neurons)
        if self.is_sparse:
            return GlobalCOO(
                row=np.asarray(self._sparse_rows, dtype=np.int32),
                col=np.asarray(self._sparse_cols, dtype=np.int32),
                weights=np.asarray(self._sparse_weights, dtype=np.float64),
                delays=np.asarray(self._sparse_delays, dtype=np.float64),
                shape=shape,
            )
        row, col = np.nonzero(np.asarray(self._global_mask))
        return GlobalCOO(
            row=row.astype(np.int32),
            col=col.astype(np.int32),
            weights=np.asarray(self._global_weights)[row, col].astype(np.float64),
            delays=np.asarray(self._global_delays)[row, col].astype(np.float64),
            shape=shape,
        )

    def global_coo(self) -> GlobalCOO:
        """グローバルな結合を COO で返す。**ビルド以降はこれが唯一の入口**。

        疎経路で生成されたか密経路で生成されたかに関わらず同じ形・同じ順序で返るので、
        呼び出し側が `is_sparse` を見る必要はない。行列生成前に呼ぶとエラー。
        """
        if self._global_coo is None:
            raise RuntimeError(
                "global_coo() は結合の生成後にしか呼べません。"
                " build() か _generate_global_matrices() を先に実行してください。"
            )
        return self._global_coo

    def _pair_synapse_counts(self, coo: GlobalCOO) -> Dict[str, int]:
        """COO をシナプス集団ごとに数える (`_pair_coo()` と同じ切り分け)。"""
        if self._index_table is None:
            self._index_table = self._local_index_table()
        pop_code, _local_of, code_of_name = self._index_table
        counts: Dict[str, int] = {}
        for syn_cfg in self.config.synapses.values():
            for src_name, tgt_name in itertools.product(syn_cfg.source, self.config.neurons):
                sel = (
                    (pop_code[coo.row] == code_of_name[src_name])
                    & (pop_code[coo.col] == code_of_name[tgt_name])
                )
                counts[f"{src_name}_to_{tgt_name}"] = int(np.count_nonzero(sel))
        return counts

    def replace_global_coo(self, coo: GlobalCOO, preserve_fan_in_scale: bool = False) -> None:
        """生成済みのグローバル COO を差し替える。**GeNN へ登録する前にだけ**呼べる。

        ここが唯一の差し替え地点である理由は 2 つ:

        - **乱数ストリームより後**。座標・結合・初期重み・遅延の生成 (`self.rng` の消費) は
          `_generate_global_matrices()` で完了しているので、ここで何をしても同一 seed の
          ネットワーク実現は変わらない。「シナプスを削ったせいで別のネットワークになった」
          が原理的に起きない。
        - **GeNN 登録より前**。`_build_synapses()` は `global_coo()` しか読まず、重みを
          `init_weight_update(vars={"w": ...})` へ渡す。よって構造的除去 (行の削除) と
          重み復元 (weights の差し替え) の両方が「GeNN 登録時の初期値」として一度に入り、
          setup 後の重み注入 (pygenn の SPARSE 行ごとの Python ループ、`setup()` の
          `backup_initial_states` が注入前の値を覚えてしまう罠) を避けられる。

        検証する不変条件。1 つでも破れば `ValueError` を投げる —— 黙って通すと、GeNN が
        期待する「集団ローカルの (pre, post) 行優先ソート順」が崩れたまま学習が走り、
        別のシナプスに重みが乗ったことに誰も気付けない:

        1. row / col / weights / delays の長さが等しい
        2. shape が `(total_neurons, total_neurons)` のまま
        3. `0 <= row, col < total_neurons`
        4. `row * N + col` が**狭義単調増加** (= 行優先ソート済み、かつ重複ペアなし)
        5. weights / delays が有限
        6. シナプスが 1 本以上ある (0 本の GeNN モデルは下流が NaN だらけになるだけで、
           原因が追えない)

        Note:
            `self.connection.axon_geometry()` は差し替えの影響を受けない。幾何の `pre`/`post`
            は**差し替え前の COO** と位置一致しているので、差し替え後にそのまま
            `axon_network()` へ渡すと切断したシナプスまで描かれる。部分集合を取った
            コピーを渡すこと。
        """
        if self.synapse_index:
            raise RuntimeError(
                "replace_global_coo() は GeNN へシナプスを登録する前にしか呼べません"
                f" (すでに {len(self.synapse_index)} 集団が登録済み)。"
                " build(transform_coo=...) を使ってください。"
            )
        if self._global_coo is None:
            raise RuntimeError(
                "replace_global_coo() は結合の生成後にしか呼べません。"
                " _generate_global_matrices() を先に実行してください。"
            )

        row = np.asarray(coo.row, dtype=np.int64)
        col = np.asarray(coo.col, dtype=np.int64)
        weights = np.asarray(coo.weights, dtype=np.float64)
        delays = np.asarray(coo.delays, dtype=np.float64)

        n = int(self.total_neurons)
        if tuple(coo.shape) != (n, n):
            raise ValueError(
                f"差し替え後の shape {tuple(coo.shape)} が (total_neurons, total_neurons)"
                f" = ({n}, {n}) と一致しません。"
            )
        sizes = {row.size, col.size, weights.size, delays.size}
        if len(sizes) != 1:
            raise ValueError(
                "row / col / weights / delays の長さが一致しません: "
                f"row={row.size}, col={col.size}, weights={weights.size}, delays={delays.size}"
            )
        if row.size == 0:
            raise ValueError(
                "差し替え後のシナプスが 0 本です。シナプスの無い GeNN モデルは下流の解析が"
                " すべて NaN になるだけで原因が追えないので、ここで止めます。"
            )
        if row.min() < 0 or row.max() >= n or col.min() < 0 or col.max() >= n:
            raise ValueError(
                f"グローバルIDが範囲外です (row: {row.min()}..{row.max()},"
                f" col: {col.min()}..{col.max()}, N={n})。"
            )
        # int64 で計算すること。int32 のままだと N が大きいときに row*N+col が折り返り、
        # 別のペアが同じキーになって「ソート済み」の判定をすり抜ける。
        keys = row * n + col
        if row.size > 1:
            steps = np.diff(keys)
            if np.any(steps <= 0):
                bad = int(np.count_nonzero(steps <= 0))
                raise ValueError(
                    f"差し替え後の COO が行優先ソート済みではありません ({bad} 箇所で"
                    " 昇順が崩れているか、同じ (pre, post) が重複しています)。"
                    " GeNN は送信元ニューロン順にシナプスを格納するので、この順序は"
                    " `_pair_coo()` の前提です。"
                )
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(delays)):
            raise ValueError("差し替え後の weights / delays に非有限値が含まれています。")

        if preserve_fan_in_scale:
            # **差し替える前の** COO で数える。可塑性モデルの g_scale = num_post/num_synapses
            # をここに固定することで、シナプスを削っても実効ゲインが動かなくなる。
            self._fan_in_reference = self._pair_synapse_counts(self._global_coo)

        self._global_coo = GlobalCOO(
            row=row.astype(np.int32),
            col=col.astype(np.int32),
            weights=weights,
            delays=delays,
            shape=(n, n),
        )
        # 生成経路の作業領域は差し替え後の結合を指していない。読めてしまうより落ちるほうが
        # 安全なので落とす (どのみち `_pair_coo()` は global_coo() しか見ない)。
        self._global_mask = None
        self._global_weights = None
        self._global_delays = None
        self._sparse_rows = None
        self._sparse_cols = None
        self._sparse_weights = None
        self._sparse_delays = None
        self._index_table = None

    def _inject_axes(self, component):
        """コンポーネントが宣言するカテゴリ/ソート軸を NetworkLayout に注入する。

        `generate()` / `generate_sparse()` の**直後**に呼ぶこと。コンポーネントは自身が
        計算した座標・マスク等から軸を導出できる。後段のコンポーネント(結合→重み→遅延)は
        先行コンポーネントが宣言した軸を `layout.ids_by(...)` などで参照できる。
        """
        axes = component.describe_axes()
        if not axes:
            return
        for name, values in axes.items():
            self.layout.add_axis(name, values)
            print(f"    Axis '{name}' declared by {type(component).__name__} "
                  f"({len(self.layout.ids_by(name))} categories)")

    def _generate_global_dense(self):
        """密な (N, N) 行列として結合・重み・遅延を生成する(従来経路)"""
        network = self.config.network

        # Pydanticモデルから辞書を取得
        area_cfg = network.area
        space_cfg = network.space
        conn_cfg = network.connection
        weight_cfg = network.weight
        delay_cfg = network.delay

        areaClass, spaceClass, connectClass, weightClass, delayClass = self._component_classes()

        # 1. 領域の構築 (soma の配置範囲と軸索の伸長範囲)。
        #    RandomState を渡さないのは、area が rng を消費すると空間→結合→重み→遅延の
        #    単一ストリームが全部ずれ、同じ seed の既存ネットワークが別物になるから。
        #    乱数は area.sample(n, rng) の引数としてのみ空間モデルから渡る。
        self.area = areaClass(area_cfg, self.total_neurons, layout=self.layout)

        # 2. 空間座標の生成
        space = spaceClass(space_cfg, self.total_neurons, self.rng, layout=self.layout, area=self.area)
        self.global_coords = space.generate()
        self._inject_axes(space)

        # 3. 結合マスクの生成
        connection = connectClass(conn_cfg, self.total_neurons, self.global_coords, self.rng, layout=self.layout, area=self.area)
        self.connection = connection
        self._global_mask = connection.generate()
        self._inject_axes(connection)

        # 4. 重み行列の生成
        weight = weightClass(weight_cfg, self.total_neurons, self.global_coords, self._global_mask, self.rng, layout=self.layout)
        self._global_weights = weight.generate()
        self._inject_axes(weight)

        # 5. 遅延行列の生成
        delay = delayClass(delay_cfg, self.total_neurons, self.global_coords, self._global_mask, self.rng, layout=self.layout)
        self._global_delays = delay.generate()
        self._inject_axes(delay)

    def _generate_global_sparse(self):
        """COO (rows, cols, weights, delays) として結合情報を生成する。

        密行列を一切作らないため、N が数万規模でもメモリに収まる。乱数の消費順は
        密経路と一致するよう各コンポーネント側で保証している
        (connectors.GaussianDistanceTypeTopology.generate_sparse の docstring 参照)。
        """
        network = self.config.network
        areaClass, spaceClass, connectClass, weightClass, delayClass = self._component_classes()

        # 密経路と同じく area は rng を受け取らない (_generate_global_dense のコメント参照)。
        self.area = areaClass(network.area, self.total_neurons, layout=self.layout)

        space = spaceClass(
            network.space, self.total_neurons, self.rng, layout=self.layout, area=self.area
        )
        self.global_coords = space.generate()
        self._inject_axes(space)

        connection = connectClass(
            network.connection, self.total_neurons, self.global_coords, self.rng,
            layout=self.layout, area=self.area,
        )
        self.connection = connection
        rows, cols = connection.generate_sparse()
        self._inject_axes(connection)
        self._sparse_rows = np.asarray(rows, dtype=np.int32)
        self._sparse_cols = np.asarray(cols, dtype=np.int32)

        weight = weightClass(
            network.weight, self.total_neurons, self.global_coords,
            mask=None, rng=self.rng, layout=self.layout,
        )
        self._sparse_weights = weight.generate_sparse(self._sparse_rows, self._sparse_cols)
        self._inject_axes(weight)

        delay = delayClass(
            network.delay, self.total_neurons, self.global_coords,
            mask=None, rng=self.rng, layout=self.layout,
        )
        self._sparse_delays = delay.generate_sparse(self._sparse_rows, self._sparse_cols)
        self._inject_axes(delay)

        print(f"    Sparse connectivity: {self._sparse_rows.size} synapses "
              f"({self._sparse_rows.size / max(self.total_neurons, 1):.1f} per neuron)")

    def _build_neuron_populations(self):
        """GeNN上にニューロンポピュレーションを定義

        population 名・個数・モデルパラメータはすべて config が源なので、ここでは
        layout を参照しない(グローバルIDを使わないため変換表が不要)。
        """
        for group_name, params in self.config.neurons.items():
            num_neurons = params.num

            NeuronClass = NEURON_MODELS.get(params.type)
            neuron_instance = NeuronClass(params, self.config.simulation.dt)
            
            self.genn_model.add_neuron_population(
                pop_name = group_name, 
                num_neurons = num_neurons,
                neuron = neuron_instance.model_class, 
                params = neuron_instance.params, 
                vars = neuron_instance.vars
            )
            print(f"  Added NeuronGroup: {group_name} ({num_neurons} neurons)")

    def _quantize_delays(self, delays_ms: np.ndarray, pop_label: str) -> np.ndarray:
        """遅延[ms]をシミュレーションステップ数(uint8)へ量子化する。

        GeNN スニペット側で遅延変数 `d` は uint8_t 宣言 (かつ per-synapse 到着機構の
        arrival_delay_var) のため 255 ステップが上限。従来はここで無言に折り返していたので、
        明示的なエラーにする。
        """
        dt = self.config.simulation.dt
        steps = np.rint(np.asarray(delays_ms, dtype=np.float64) / dt)
        if steps.size and steps.max() > 255:
            max_ms = float(steps.max()) * dt
            raise ValueError(
                f"{pop_label}: 遅延 {max_ms:.2f} ms = {int(steps.max())} ステップ が uint8 の上限 "
                f"(255 ステップ = {255 * dt:.1f} ms) を超えています。"
                f" network.delay の max_delay を {255 * dt:.1f} ms 以下にするか、"
                f" 伝導速度を上げてください。"
            )
        return steps.astype(np.uint8)

    def _local_index_table(self):
        """グローバルID -> (集団コード, 集団ローカルindex) の索引表を作る。

        layout の集団は 0..N-1 の分割なので、全ID に対して一意に定まる。
        """
        n = self.total_neurons
        pop_code = np.full(n, -1, dtype=np.int16)
        local_of = np.zeros(n, dtype=np.int32)
        code_of_name = {}
        for code, name in enumerate(self.config.neurons):
            ids = np.asarray(self.layout.global_indices(name), dtype=np.int64)
            pop_code[ids] = code
            local_of[ids] = np.arange(ids.size, dtype=np.int32)
            code_of_name[name] = code
        return pop_code, local_of, code_of_name

    def _pair_coo(self, src_name: str, tgt_name: str):
        """(src_pop -> tgt_pop) の接続を集団ローカルの COO として取り出す。

        入力はグローバル COO 1 本 (`global_coo()`)。疎で生成したか密で生成したかで
        分岐しないのは、正規化の時点で違いが消えているから。

        `layout.global_indices` は昇順なので `local_of` は各集団上で単調増加であり、
        行優先ソート済みキーへの単調写像はソート順を保つ。よってここで得られる並びは、
        密行列を `np.where(sub_mask)` で走査したのと同一 = GeNN が期待する
        「集団ローカルの (pre, post) 行優先ソート順」になる。

        Returns:
            (local_src, local_tgt, weights_flat, delays_ms, positions) いずれも行優先
            ソート済みで index が整合した 1D 配列。接続が無ければすべて空配列。
            `positions` は `global_coo()` の中での位置 (昇順)。
        """
        coo = self.global_coo()
        if self._index_table is None:
            self._index_table = self._local_index_table()
        pop_code, local_of, code_of_name = self._index_table

        sel = (
            (pop_code[coo.row] == code_of_name[src_name])
            & (pop_code[coo.col] == code_of_name[tgt_name])
        )
        positions = np.flatnonzero(sel)
        return (
            local_of[coo.row[positions]],
            local_of[coo.col[positions]],
            coo.weights[positions],
            coo.delays[positions],
            positions,
        )

    def _build_synapses(self):
        """グローバルな結合情報からシナプス集団ごとに切り出し、GeNN へ登録する"""
        print("  Building Synapse Populations...")
        self._index_table = None
        # source は population 名のリスト (単一名も ConfigManager が 1 要素に正規化済み)。
        # 同じ極性のニューロンを動特性 mode 別に複数 population へ分けても、シナプス種別
        # (可塑性・コンダクタンス) の定義は 1 箇所で済む。
        for syn_cfg in self.config.synapses.values():
            for src_name, tgt_name in itertools.product(syn_cfg.source, self.config.neurons):
                print(f"src:{src_name}, tgt:{tgt_name}")

                src_indices = self.layout.global_indices(src_name)
                tgt_indices = self.layout.global_indices(tgt_name)

                local_src_idx, local_tgt_idx, weights_flat, delays_flat_ms, positions = \
                    self._pair_coo(src_name, tgt_name)

                delay_by_target = getattr(syn_cfg, "delay_by_target", None)
                # delay_by_target 指定は集団内で単一定数 = 均一遅延。この場合のみ GeNN の
                # 軸索遅延(axonal_delay_steps)を使い、pre_spike_syn_code を到着時刻に
                # イベント駆動化して毎ステップの syn_dynamics_code を撤廃する(高速化)。
                use_axonal = delay_by_target is not None and tgt_name in delay_by_target
                if use_axonal:
                    delays_flat_ms = np.full_like(delays_flat_ms, float(delay_by_target[tgt_name]))
                elif hasattr(syn_cfg, "delay"):
                    delays_flat_ms = np.full_like(delays_flat_ms, float(syn_cfg.delay))

                if len(local_src_idx) == 0:
                    print(f"    Skipping SynapseGroup: {src_name}_to_{tgt_name} (No connections found)")
                    continue
                delays_flat = self._quantize_delays(delays_flat_ms, f"{src_name}_to_{tgt_name}")

                src_pop = self.genn_model.neuron_populations[src_name]
                tgt_pop = self.genn_model.neuron_populations[tgt_name]

                PlasClass = PLASTICITY_MODELS.get(syn_cfg.plasticity.type)
                # 可塑性モデルが axonal_delay_steps を受け取れる場合のみ軸索遅延経路を適用する
                # (= opt-in)。標準モデルは受け取らないため従来の dendritic delay 経路のまま。
                supports_axonal = "axonal_delay_steps" in inspect.signature(PlasClass.__init__).parameters
                axonal_steps = None
                if use_axonal and supports_axonal:
                    uniform_delay_ms = float(delay_by_target[tgt_name])
                    # STDP の実効到着時刻は emission + axonal_delay_steps*dt になる(learn-post /
                    # pre_spike_syn の処理ステップ基準。実測で確認済み)。よって STDP タイミングを
                    # delay_corrected と一致させるには steps = round(D/dt)。
                    # ※電流(addToPost)の到着は (steps+1)*dt となり D より 1dt(=dt)遅いが、
                    #   PSC は指数減衰でならされるため重み学習への影響は無視できる。
                    axonal_steps = max(0, int(round(uniform_delay_ms / self.config.simulation.dt)))
                else:
                    use_axonal = False  # 非対応モデルには適用しない

                plas_kwargs = dict(
                    config=syn_cfg.plasticity,
                    dt=self.config.simulation.dt,
                    weight=weights_flat,
                    delay=delays_flat,
                    num_pre=src_pop.num_neurons,
                    num_post=tgt_pop.num_neurons,
                )
                if use_axonal:
                    plas_kwargs["axonal_delay_steps"] = axonal_steps
                # fan-in 正規化の分母の固定も axonal と同じ opt-in 方式。受け取れない可塑性
                # モデルには渡さない (標準モデルはそもそも fan-in 正規化を持たない)。
                if self._fan_in_reference is not None:
                    pair_name = f"{src_name}_to_{tgt_name}"
                    if "fan_in_num_synapses" in inspect.signature(PlasClass.__init__).parameters:
                        plas_kwargs["fan_in_num_synapses"] = self._fan_in_reference[pair_name]
                plas_instance = PlasClass(**plas_kwargs)
                self._component_lifeline.append(plas_instance)
                weight_init = pygenn.genn_model.init_weight_update(
                    snippet=plas_instance.snippet, 
                    params=plas_instance.params, 
                    vars=plas_instance.vars,
                    pre_vars=plas_instance.pre_vars,
                    post_vars=plas_instance.post_vars,
                    pre_var_refs=plas_instance.pre_var_refs,
                    post_var_refs=plas_instance.post_var_refs,
                    psm_var_refs=plas_instance.psm_var_refs   
                )
                
                SynClass = SYNAPSE_MODELS.get(syn_cfg.synapse.type)
                syn_instance = SynClass(
                    config=syn_cfg.synapse, 
                    dt=self.config.simulation.dt,
                    pop=tgt_pop
                )
                self._component_lifeline.append(syn_instance)
                post_init = pygenn.genn_model.init_postsynaptic(
                    snippet=syn_instance.snippet, 
                    params=syn_instance.params,
                    vars=syn_instance.vars,
                    var_refs=syn_instance.var_refs
                )

                sg = self.genn_model.add_synapse_population(
                    pop_name=f"{src_name}_to_{tgt_name}", 
                    matrix_type="SPARSE", 
                    source=src_pop, 
                    target=tgt_pop, 
                    weight_update_init=weight_init, 
                    postsynaptic_init=post_init
                )
                # GeNN へ渡した接続順をそのまま記録する。以降 simulator 側は結合マスクを
                # 再走査せずにシナプス変数をグローバルIDへ散布できる。
                self.synapse_index[f"{src_name}_to_{tgt_name}"] = SynapseIndex(
                    src_name=src_name,
                    tgt_name=tgt_name,
                    local_src=local_src_idx.astype(np.int32, copy=False),
                    local_tgt=local_tgt_idx.astype(np.int32, copy=False),
                    global_src=np.asarray(src_indices, dtype=np.int32)[local_src_idx],
                    global_tgt=np.asarray(tgt_indices, dtype=np.int32)[local_tgt_idx],
                    global_positions=positions,
                )

                sg.set_sparse_connections(local_src_idx, local_tgt_idx)
                if use_axonal:
                    # 軸索遅延: ソーススパイクを axonal_steps 分遅延スロットから読み、到着時刻に
                    # pre_spike_syn_code を発火。addToPost 即時投与なので dendritic バッファは不要。
                    sg.axonal_delay_steps = axonal_steps
                else:
                    max_delay_steps = int(np.max(delays_flat)) + 1
                    sg.max_dendritic_delay_timesteps = max_delay_steps
                    sg.num_threads_per_spike = self._arrival_threads_per_spike(
                        local_src_idx, max_delay_steps, f"{src_name}_to_{tgt_name}")

    @staticmethod
    def _arrival_threads_per_spike(local_src_idx, max_delay_steps, label):
        """per-synapse 到着 kernel の 1 スパイクあたりスレッド数を決める。

        到着 kernel は既定 (=1) だと「並列度=キュー内スパイク数(数十)、直列深さ=行長
        (数百〜千)」という GPU の苦手な形になり、依存するランダムアクセスのレイテンシが
        素通しで積み上がる。1 スパイクを T スレッドで分担すると presynaptic kernel と同じ
        「広くて浅い」形に転置でき、遅延バケツ内は連続アクセス (coalesced) になる。

        T は「1 スパイク・1 遅延ステップあたり平均何本のシナプスが届くか」= 行長/遅延段数
        を目安に、warp 幅 32 を上限として 2 の冪へ丸める。バケツより T が大きいと余ったスレッド
        が遊ぶだけなので、上振れを避けてこの規模に合わせる。
        """
        if len(local_src_idx) == 0:
            return 1
        _, counts = np.unique(np.asarray(local_src_idx), return_counts=True)
        mean_bucket = float(counts.mean()) / max(1, int(max_delay_steps))
        t = 1
        while t < 32 and t < mean_bucket:
            t *= 2
        print(f"    [arrival] {label}: 行長 {counts.mean():.0f} / 遅延 {max_delay_steps} 段"
              f" -> num_threads_per_spike = {t}")
        return t

    def _build_input_ports(self):
        """GeNN上に入力専用のglobal_popを定義し、本体へ1対1で接続する"""

        if self.config.inputs.GaussianNoise.enable:
            for pop_name in self.config.neurons:
                cs_name = f"GaussianNoise_CS_to_{pop_name}"

                cs = self.genn_model.add_current_source(
                    cs_name=cs_name,
                    current_source_model="GaussianNoise", 
                    pop=self.genn_model.neuron_populations[pop_name],
                    params={
                        "mean": self.config.inputs.GaussianNoise.mean,
                        "sd": self.config.inputs.GaussianNoise.sd
                    },
                    vars={}
                )
                print(f"    Added Gaussian Noise Current Source '{cs_name}' to '{pop_name}'")
            
            print(f"  Added Input Groups for {self.total_neurons} global neurons.")
            

    def _build_output_ports(self, rec_spike: bool):
        """スパイク記録の準備"""
        for group_name in self.config.neurons.keys():
            pop_name = group_name
            pop = self.genn_model.neuron_populations[pop_name]
    
            # スパイクの記録設定
            pop.spike_recording_enabled = rec_spike
                





if __name__ == "__main__":
    import traceback
    from src.core.config_manager import ConfigManager
    import src.models.neurons.pqn_float
    import src.models.neurons.pqn_int
    import src.models.neurons.lif
    import src.models.network.space
    import src.models.network.connectors
    import src.models.network.weights
    import src.models.network.delays
    import src.models.plasticity.standard_models
    import src.models.plasticity.custom_Akita
    import src.models.synapses.standard_models
    import src.models.synapses.custom
    import src.data.test_data

    print("=== NetworkBuilder 動作検証テストを開始します ===")

    try:
        # 1. Configのロード
        config_src = "test.yaml" # 実際のファイルパスに合わせてください
        print(f"Loading config from {config_src}...")
        manager = ConfigManager(config_src, "pqn_test") 
        config = manager.resolve()

        # 2. ビルダーの初期化
        print("\n[TEST] Initializing NetworkBuilder...")
        builder = NetworkBuilder(config)
        print(f"  ✓ Initialization successful. Total neurons: {builder.total_neurons}")

        # 3. ビルドプロセスの実行
        print("\n[TEST] Executing build()...")
        genn_model, layout = builder.build(rec_spike=True)

        # 4. アサーションと検証
        print("\n[TEST] Validating generated objects...")
        Npop_names = list(genn_model.neuron_populations.keys())
        print(f"  - Registered Populations in GeNN: {Npop_names}")
        Spop_names = list(genn_model.synapse_populations.keys())
        print(f"  - Registered Synapse Populations in GeNN: {Spop_names}")
        CSpop_names = list(genn_model.current_sources.keys())
        print(f"  - Registered Current Sources in GeNN: {CSpop_names}")

        # --- 本体層の存在確認 ---
        for body_name in config.neurons.keys():
            assert body_name in Npop_names, f"Body Pop '{body_name}' not found in GeNN model."
            # config の写しではなく population 軸そのものを見る (config と突き合わせても
            # 恒真になるだけで、実際にグローバルIDが割り当たったかを検査できない)。
            assert body_name in layout.values("population"), f"'{body_name}' missing in layout."
        print("  ✓ Body populations successfully validated.")

        # --- Current Source の確認 ---
        if config.inputs.GaussianNoise.enable:
            cs_names = list(genn_model.current_sources.keys())
            print(f"  - Registered Current Sources: {cs_names}")
            # 各body_popに対してDCソースが作られているか確認
            for body_name in config.neurons.keys():
                expected_cs = f"DC_CS_to_{body_name}"
                assert expected_cs in cs_names, f"Current Source '{expected_cs}' missing."
            print("  ✓ Current sources successfully validated.")

        print("\n🎉 === 全てのテストをクリアしました (All Tests Passed) === 🎉")

    except Exception as e:
        print(f"\n❌ [Error] 検証中にエラーが発生しました:")
        traceback.print_exc()
