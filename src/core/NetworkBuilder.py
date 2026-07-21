import os
import inspect
import numpy as np
import pygenn
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from pathlib import Path
import sys

# CUDAバックエンドを有効にするため、未設定の場合はデフォルトパスを補完する
if not os.environ.get("CUDA_PATH"):
    for _candidate in ["/usr/local/cuda", "/usr/cuda"]:
        if Path(_candidate).exists():
            os.environ["CUDA_PATH"] = _candidate
            break

# CPU/GPU バックエンド選択:
#   None  = ニューロン数で自動選択 (total_neurons <= GPU_NEURON_THRESHOLD なら CPU、超えたら GPU)
#   True  = 常に GPU (cuda) を強制
#   False = 常に CPU (single_threaded_cpu) を強制
# 自動選択の根拠: per-synapse 到着イベント駆動化 (pre_arrival_syn_code) 後の実測で、小規模では
# CPU が最速 (100n=11.4µs/step, GPU 40.8µs は per-step 起動 floor 律速で勝てない)、~400n 付近が
# crossover で大規模は GPU が平坦有利。詳細: docs/gpu_vs_cpu.md「実装後の実測」。
USE_GPU = True
GPU_NEURON_THRESHOLD = 400

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pydanticの設定モデルと、コンポーネントを動的ロードするレジストリをインポート
from src.core.config_manager import AppConfig
from src.core.layout import NetworkLayout
from src.core.registry import SPATIAL_MODELS, CONNECTION_MODELS, WEIGHT_MODELS, DELAY_MODELS, NEURON_MODELS, SYNAPSE_MODELS, PLASTICITY_MODELS

@dataclass(frozen=True)
class SynapseIndex:
    """1つのシナプス集団 (src_pop -> tgt_pop) の接続インデックスを保持する。

    GeNN のシナプス変数は「集団ローカルの (pre, post) 行優先ソート順」で読み書きされる
    (pygenn が set_sparse_connections で lexsort し、値の getter は行優先で返す)。
    ここに保持する配列はその順序と一致しており、`simulator.pull_synapse` 系はこれを使って
    結合マスクを再走査せずに値をグローバルIDへ散布できる。
    """
    src_name: str
    tgt_name: str
    local_src: np.ndarray   # int32, 集団ローカル pre index
    local_tgt: np.ndarray   # int32, 集団ローカル post index
    global_src: np.ndarray  # int32, グローバル pre ID
    global_tgt: np.ndarray  # int32, グローバル post ID

    @property
    def num_synapses(self) -> int:
        return int(self.local_src.size)


class NetworkBuilder:
    def __init__(self, config: AppConfig, model_name: str = "SNN_Model", code_gen_dir: str | None = None):
        self.config = config
        self.rng = np.random.RandomState(config.simulation.seed)

        # GeNN が生成する <model_name>_CODE を置く親ディレクトリ。
        # None の場合はカレントディレクトリ (従来挙動)。Simulator.setup() の build() で使う。
        self.code_gen_dir = code_gen_dir

        # config のニューロン宣言順に連番でグローバルインデックスを割り当てる決定論的レイアウト。
        # RandomState を消費しないため、GeNN ビルドなしでも from_config だけで再現できる。
        # backend 自動選択 (USE_GPU is None) がニューロン数を参照するため、モデル生成より先に確定させる。
        self.layout = NetworkLayout.from_config(config)
        self.total_neurons = self.layout.total_neurons

        # backend 選択: USE_GPU が None ならニューロン数で自動、True/False なら強制。
        if USE_GPU is None:
            use_gpu = self.total_neurons > GPU_NEURON_THRESHOLD
            _reason = f"auto (total_neurons={self.total_neurons} {'>' if use_gpu else '<='} {GPU_NEURON_THRESHOLD})"
        else:
            use_gpu = bool(USE_GPU)
            _reason = "forced (USE_GPU)"
        _backend = "cuda" if use_gpu else "single_threaded_cpu"
        print(f"[NetworkBuilder] backend = {_backend}  [{_reason}]")
        self.genn_model = pygenn.GeNNModel("double", model_name, time_precision="double", backend=_backend)
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
        self.global_mask = None
        self.global_weights = None
        self.global_delays = None
        # 疎生成経路で使う COO (すべて index 整合の 1D 配列)。密経路では None のまま。
        self.sparse_rows = None
        self.sparse_cols = None
        self.sparse_weights = None
        self.sparse_delays = None
        # 疎経路での グローバルID -> (集団コード, ローカルindex) 索引表 (遅延構築)
        self._index_table = None
        # "src_to_tgt" -> SynapseIndex。_build_synapses が GeNN へ登録した接続順を記録する。
        self.synapse_index: Dict[str, SynapseIndex] = {}

    def build(self, rec_spike: bool = True) -> Tuple[pygenn.GeNNModel, NetworkLayout]:
        print("=== ネットワーク構築 (Network Building) ===")
        # 1. 全ニューロン一括での座標・グローバル行列生成 (レイアウトは __init__ で確定済み)
        self._generate_global_matrices()
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
            SPATIAL_MODELS.get(network.space.profile_name),
            CONNECTION_MODELS.get(network.connection.profile_name),
            WEIGHT_MODELS.get(network.weight.profile_name),
            DELAY_MODELS.get(network.delay.profile_name),
        )

    def _use_sparse(self) -> bool:
        """疎生成経路を使うかどうかを config と各コンポーネントの対応状況から決める。

        "auto"(既定): 結合/重み/遅延の3段すべてが疎対応なら疎。1段でも非対応なら密。
        "force"     : 疎を必須とし、非対応クラス名を挙げて即エラー(大規模実行で
                      20分走ってから OOM kill されるのを防ぐ)。
        "off"       : 常に密(過去のネットワーク実現を再現したいとき)。
        """
        mode = getattr(self.config.network, "sparse", "auto")
        if mode not in ("auto", "force", "off"):
            raise ValueError(
                f"network.sparse は 'auto' / 'force' / 'off' のいずれかです (got {mode!r})。"
            )
        if mode == "off":
            return False

        _, connect_cls, weight_cls, delay_cls = self._component_classes()
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
        return False

    def _generate_global_matrices(self):
        """全ニューロンの座標と、グローバルな結合情報を生成する"""
        print("  Generating Global Coordinates and Matrices...")
        if self._use_sparse():
            self._generate_global_sparse()
        else:
            self._generate_global_dense()

    def _generate_global_dense(self):
        """密な (N, N) 行列として結合・重み・遅延を生成する(従来経路)"""
        network = self.config.network

        # Pydanticモデルから辞書を取得
        space_cfg = network.space
        conn_cfg = network.connection
        weight_cfg = network.weight
        delay_cfg = network.delay

        spaceClass, connectClass, weightClass, delayClass = self._component_classes()

        # 1. 空間座標の生成
        self.global_coords = spaceClass(space_cfg, self.total_neurons, self.rng, layout=self.layout).generate()

        # 2. 結合マスクの生成
        self.global_mask = connectClass(conn_cfg, self.total_neurons, self.global_coords, self.rng, layout=self.layout).generate()

        # 3. 重み行列の生成
        self.global_weights = weightClass(weight_cfg, self.total_neurons, self.global_coords, self.global_mask, self.rng, layout=self.layout).generate()

        # 4. 遅延行列の生成
        self.global_delays = delayClass(delay_cfg, self.total_neurons, self.global_coords, self.global_mask, self.rng, layout=self.layout).generate()

    def _generate_global_sparse(self):
        """COO (rows, cols, weights, delays) として結合情報を生成する。

        密行列を一切作らないため、N が数万規模でもメモリに収まる。乱数の消費順は
        密経路と一致するよう各コンポーネント側で保証している
        (connectors.GaussianDistanceTypeTopology.generate_sparse の docstring 参照)。
        """
        network = self.config.network
        spaceClass, connectClass, weightClass, delayClass = self._component_classes()

        self.global_coords = spaceClass(
            network.space, self.total_neurons, self.rng, layout=self.layout
        ).generate()

        rows, cols = connectClass(
            network.connection, self.total_neurons, self.global_coords, self.rng, layout=self.layout
        ).generate_sparse()
        self.sparse_rows = np.asarray(rows, dtype=np.int32)
        self.sparse_cols = np.asarray(cols, dtype=np.int32)

        self.sparse_weights = weightClass(
            network.weight, self.total_neurons, self.global_coords,
            mask=None, rng=self.rng, layout=self.layout,
        ).generate_sparse(self.sparse_rows, self.sparse_cols)

        self.sparse_delays = delayClass(
            network.delay, self.total_neurons, self.global_coords,
            mask=None, rng=self.rng, layout=self.layout,
        ).generate_sparse(self.sparse_rows, self.sparse_cols)

        print(f"    Sparse connectivity: {self.sparse_rows.size} synapses "
              f"({self.sparse_rows.size / max(self.total_neurons, 1):.1f} per neuron)")

    def _build_neuron_populations(self):
        """GeNN上にニューロンポピュレーションを定義"""
        for group_name, grp in self.layout.items():
            params = grp.params
            num_neurons = grp.num

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
        for code, (name, spec) in enumerate(self.layout.items()):
            ids = np.asarray(self.layout.global_indices(name), dtype=np.int64)
            pop_code[ids] = code
            local_of[ids] = np.arange(ids.size, dtype=np.int32)
            code_of_name[name] = code
        return pop_code, local_of, code_of_name

    def _pair_coo(self, src_name: str, tgt_name: str):
        """(src_pop -> tgt_pop) の接続を集団ローカルの COO として取り出す。

        Returns:
            (local_src, local_tgt, weights_flat, delays_ms) いずれも行優先ソート済みで
            index が整合した 1D 配列。接続が無ければすべて空配列。
        """
        if self.sparse_rows is None:
            # 密経路: 従来どおりグローバル行列から np.ix_ で切り出す。
            src_indices = self.layout.global_indices(src_name)
            tgt_indices = self.layout.global_indices(tgt_name)
            sub_mask = self.global_mask[np.ix_(src_indices, tgt_indices)]
            local_src, local_tgt = np.where(sub_mask != 0)
            sub_weights = self.global_weights[np.ix_(src_indices, tgt_indices)]
            sub_delays = self.global_delays[np.ix_(src_indices, tgt_indices)]
            return (
                local_src,
                local_tgt,
                sub_weights[local_src, local_tgt],
                sub_delays[local_src, local_tgt].astype(np.float64),
            )

        # 疎経路: グローバル COO をブールフィルタする。
        # layout.global_indices は昇順なので local_of は各集団上で単調増加であり、
        # 行優先ソート済みキーへの単調写像はソート順を保つ。よって密経路の
        # np.where(sub_mask) と同一の順序が得られる。
        if self._index_table is None:
            self._index_table = self._local_index_table()
        pop_code, local_of, code_of_name = self._index_table

        sel = (
            (pop_code[self.sparse_rows] == code_of_name[src_name])
            & (pop_code[self.sparse_cols] == code_of_name[tgt_name])
        )
        rows = self.sparse_rows[sel]
        cols = self.sparse_cols[sel]
        return (
            local_of[rows],
            local_of[cols],
            self.sparse_weights[sel],
            self.sparse_delays[sel].astype(np.float64),
        )

    def _build_synapses(self):
        """グローバルな結合情報からシナプス集団ごとに切り出し、GeNN へ登録する"""
        print("  Building Synapse Populations...")
        self._index_table = None
        for syn_group_name, syn_cfg in self.config.synapses.items():
            for tgt_name in self.layout.names():
                src_name = syn_cfg.source
                # tgt_name = syn_cfg.target
                print(f"src:{src_name}, tgt:{tgt_name}")

                src_indices = self.layout.global_indices(src_name)
                tgt_indices = self.layout.global_indices(tgt_name)

                local_src_idx, local_tgt_idx, weights_flat, delays_flat_ms = self._pair_coo(
                    src_name, tgt_name
                )

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
            for pop_name in self.layout.names():
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
            assert body_name in layout.names(), f"'{body_name}' missing in layout."
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
