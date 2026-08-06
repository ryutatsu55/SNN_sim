import shutil
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, Any, Literal, Optional, List
from pydantic import BaseModel, Field, ConfigDict


# メイン config が layout.assignment を書かなかったときに適用する値。
# `_materialize_layout()` が resolve 時に実値として焼き込むので、ここを変えても
# **既に保存された config.yaml の解釈は変わらない** (将来の run にだけ効く)。
DEFAULT_ASSIGNMENT = "random"

# メイン config が simulation.backend を書かなかったときに適用する値。
# 既定を "auto" にしないのは、それが**過去の run と結果を変える**から。GeNN の
# デバイス RNG (escape noise 等) はバックエンドごとに別の乱数列を出すため、同じ seed でも
# CPU と GPU ではスパイク列が違う (実測: N=100/2000ms/seed=777 で CPU 73 spikes,
# GPU 81 spikes)。
DEFAULT_BACKEND = "cpu"

# backend: "auto" を実値へ解決するときのニューロン数しきい値 (これ以下なら cpu)。
# 根拠: per-synapse 到着イベント駆動化 (pre_arrival_syn_code) 後の実測で、小規模では CPU が
# 最速 (100n=11.4µs/step。GPU の 40.8µs は per-step 起動 floor 律速で勝てない)、~400n 付近が
# crossover で大規模は GPU が平坦有利。詳細: docs/technical/gpu_vs_cpu.md「実装後の実測」。
# しきい値がここにあるのは、"auto" を**記録に残る実値へ焼き込むのが resolve() の役目**だから
# (NetworkBuilder に置くと解決が実行時になり、config.yaml に "auto" のまま残ってしまう)。
AUTO_BACKEND_NEURON_THRESHOLD = 400


def _to_python_native(obj):
    """numpy スカラ等を Python ネイティブに落とす (yaml.safe_dump 用)。"""
    if isinstance(obj, dict):
        return {k: _to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

# ==========================================
# 1. Pydanticによるスキーマ定義 (データコントラクト)
# ==========================================

class SimulationConfig(BaseModel):
    """シミュレーションの基本設定"""
    dt: float = Field(..., description="シミュレーションのタイムステップ(ms)")
    N: int = Field(..., description="総ニューロン数")
    # None = 「未確定」。ConfigManager.resolve() が実値に確定させるので、resolve() を
    # 通った AppConfig の seed は必ず int になる。ここで既定値を乱数にしてはいけない
    # (Field の default は import 時に 1 回しか評価されず、プロセス内で共有される定数に
    #  なってしまう)。
    seed: Optional[int] = Field(default=None, description="乱数シード (未指定なら resolve() 時にランダム生成)")
    # GeNN の計算バックエンド。seed と同じく「実行の再現に必要な情報」なので config に持たせ、
    # `resolve()` が実値 ("cuda" | "cpu") へ確定させて記録に焼き込む。入力 YAML には "auto"
    # (ニューロン数で自動選択) も書けるが、"auto" が検証済み AppConfig に残ることはない。
    #
    # None を許すのは **backend フィールドが無い時代に保存された config.yaml を
    # `load_resolved()` で読めるようにするため**。seed や layout.assignment と違い backend は
    # レイアウト復元に要らない (= 解析には影響しない) ので、記録に無くても解析は通してよい。
    # ただし None のまま再実行はできず、NetworkBuilder が明示的に弾く。
    backend: Optional[Literal["cuda", "cpu"]] = Field(
        default=None, description='GeNN バックエンド。入力では "auto" も可 (resolve() が実値化)'
    )
    # duration: Optional[float] = Field(default=None, description="合計シミュレーション時間(ms)")
    # backend: str = Field(default="CUDA", description="GeNNのバックエンド")
    # model_config = ConfigDict(extra='allow')

class NeuronConfig(BaseModel):
    """ニューロングループの設定"""
    type: str = Field(..., description="ニューロンモデルの種類 (例: PQN_int)")
    mode: Optional[str] = Field(default=None, description="モデルの動特性モード (例: RSexci, LTS)")
    # E/I は mode とは独立の軸。mode は動特性 (RSexci/LTS/FS…) を選ぶためのキーであり、
    # 同じ極性の population を複数の mode で持てるようにするため分離してある。
    # NetworkLayout.from_config が polarity 軸を貼り、E/I 依存の結合・遅延・可視化は
    # すべて layout.ids_by("polarity") を見る。
    polarity: Literal["excitatory", "inhibitory"] = Field(
        ..., description="シナプス極性。E/I 依存処理はすべてこれを見る"
    )
    num: int = Field(..., description="このグループのニューロン数")
    model_config = ConfigDict(extra='allow') # components/neurons.yamlから読み出される追加パラメータ(tau, v_rest等)を許可

class SynapseParamsConfig(BaseModel):
    """シナプスダイナミクス(伝達物質の放出など)の設定"""
    type: str = Field(..., description="シナプスモデルの種類 (例: tsodyks_markram)")
    mode: Optional[str] = Field(default=None, description="シナプスのモード (例: facilitate)")
    model_config = ConfigDict(extra='allow')

class PlasticityConfig(BaseModel):
    """STDPなどの学習・可塑性の設定"""
    type: str = Field(..., description="可塑性モデルの種類 (例: static)")
    mode: Optional[str] = Field(default=None, description="可塑性のモード / パラメータブロックキー (例: e-stdp)")
    pairing: Optional[str] = Field(default=None, description="STDP のペアリング方式 (nearest | trace)。custom_Akita で使用")
    model_config = ConfigDict(extra='allow')

class SynapseGroupConfig(BaseModel):
    """シナプス結合グループの設定。

    `source` は「このシナプス種別で発信する population 群」。同じ極性のニューロンを
    複数の population に分けた場合 (動特性 mode 別など) にまとめて 1 つのシナプス種別へ
    束ねられる。

    **常にリストで書くこと**。1 個でも `[Layer_Exc]` とする。スカラも受けて内部で包むと、
    同じ意味に 2 つの書き方ができてしまい、かつ `save_config` が書く config.yaml では
    どちらもリストに揃うため入力と記録の形が食い違う。
    """
    source: List[str] = Field(..., min_length=1, description="シナプス前ニューロングループ名のリスト")
    # target: str = Field(..., description="シナプス後ニューロングループ名")
    # weight_scale: float = Field(default=1.0, description="初期重みのスケーリング係数")
    synapse: SynapseParamsConfig
    plasticity: PlasticityConfig
    model_config = ConfigDict(extra='allow')

class ComponentConfig(BaseModel):
    """ネットワーク生成(空間、結合確率、重み、遅延)のベーススキーマ"""
    profile_name: str = Field(..., description="componentsディレクトリのYAMLで定義されたプロファイル名")
    model_config = ConfigDict(extra='allow') 

class NetworkConfig(BaseModel):
    """ネットワークトポロジー生成の設定"""
    space: ComponentConfig
    connection: ComponentConfig
    weight: ComponentConfig
    delay: ComponentConfig
    sparse: str = Field(
        default="auto",
        description=(
            "結合行列の生成経路。"
            '"auto"=結合/重み/遅延の3段すべてが疎対応なら疎生成、'
            '"force"=疎生成を必須(非対応ならエラー)、'
            '"off"=常に密生成(過去の実現を再現したいとき)'
        ),
    )

class LayoutConfig(BaseModel):
    """NetworkLayout の構成: population のグローバルID割当方式。

    - assignment: "sequential"(config順に連番) / "random"(全体比率ランダム)。

    **既定値をここに書かない**のが要点。メイン config で未指定なら
    `ConfigManager._materialize_layout()` が実値へ確定させるので、検証済み AppConfig に
    到達した時点で assignment は必ず具体値になっている (seed と同じ扱い)。保存される
    config.yaml も実値を持つので、「この run がどちらだったか」は記録の中で完結し、
    既定値をあとから変えても過去の run の解釈は変わらない。

    構造層・空間モジュール等のカテゴリ軸はここでは定義しない。それらはネットワーク
    コンポーネントが `describe_axes()` で宣言し、NetworkBuilder が NetworkLayout へ
    注入する (`src/core/layout.py` 参照)。
    """
    assignment: Literal["sequential", "random"] = Field(
        ..., description='population のグローバルID割当方式'
    )
    model_config = ConfigDict(extra='allow')

class InputSourceConfig(BaseModel):
    """各入力ソースの汎用設定"""
    enable: bool = Field(..., description="この入力ソースを有効にするかどうか")
    model_config = ConfigDict(extra='allow')

class InputsConfig(BaseModel):
    GaussianNoise: Optional[InputSourceConfig] = None

class MetaConfig(BaseModel):
    timestamp: str

class AppConfig(BaseModel):
    """アプリケーション全体の設定を統括するルートスキーマ"""
    simulation: SimulationConfig
    inputs: InputsConfig
    neurons: Dict[str, NeuronConfig] = Field(default_factory=dict)
    synapses: Dict[str, SynapseGroupConfig] = Field(default_factory=dict)
    network: NetworkConfig
    # 任意セクションではない。resolve() / load_resolved() が必ず実値へ確定させてから
    # 検証するので、AppConfig を名乗る以上 assignment は決まっている。
    layout: LayoutConfig
    task: ComponentConfig
    meta: MetaConfig

# ==========================================
# 2. ConfigManager 実装
# ==========================================

class ConfigManager:
    def __init__(self):
        # 直近の resolve() に渡された入力 main YAML のパス。save_config() が
        # source_config.yaml として逐語コピーするために保持する。
        # load_resolved() 経由では入力元が無いので None のまま。
        self._source_path: Optional[Path] = None

    @staticmethod
    def _materialize_seed(sim_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """simulation.seed が未指定/null なら実値に確定させた copy を返す。

        seed は NetworkBuilder の RandomState (結合・重み・遅延) と GeNN のデバイス RNG
        (escape noise 等) と NetworkLayout の random 割当の 3 つを決める。None のままだと
        いずれもランダムかつ **記録されない** ため、同じ config から別のネットワークが
        生まれ、`runio.resolve_layout()` が実行時と異なる割当を返す。ここで確定させる
        ことで、resolve() の戻り値と保存される config.yaml が必ず実 seed を持つ。

        グローバル np.random の状態を読みも汚しもしないよう、OS エントロピーを直接引く。
        """
        sim_cfg = dict(sim_cfg)
        if sim_cfg.get("seed") is None:
            sim_cfg["seed"] = int(np.random.SeedSequence().entropy % (2 ** 32))
            print(f"[ConfigManager] simulation.seed が未指定のため自動生成しました: {sim_cfg['seed']}")
        return sim_cfg

    @staticmethod
    def _materialize_backend(
        sim_cfg: Dict[str, Any], neurons_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """simulation.backend を実値 ("cuda" | "cpu") に確定させた copy を返す。

        seed / layout.assignment と同じ理由でここで確定させる。GeNN のデバイス RNG は
        バックエンドごとに別の乱数列を出すので、**backend は seed と対で初めて run を一意に
        決める**。値が記録に残らないと、保存済み config.yaml を見ても「その結果が CPU 由来か
        GPU 由来か」が判別できない。

        "auto" は `AUTO_BACKEND_NEURON_THRESHOLD` との比較で実値へ解決する。解決を実行時
        (NetworkBuilder) ではなくここで行うのは、config.yaml に "auto" のまま残ると
        しきい値を変えた日に過去の run が別バックエンドの記録として読み直されるため。
        """
        sim_cfg = dict(sim_cfg)
        backend = sim_cfg.get("backend")
        if backend is None:
            backend = DEFAULT_BACKEND
            print(
                f"[ConfigManager] simulation.backend が未指定のため既定値を適用しました: "
                f"{DEFAULT_BACKEND}"
            )
        if backend == "auto":
            total = sum(int(n["num"]) for n in neurons_cfg.values())
            backend = "cpu" if total <= AUTO_BACKEND_NEURON_THRESHOLD else "cuda"
            print(
                f"[ConfigManager] simulation.backend='auto' を実値へ解決しました: {backend} "
                f"(total_neurons={total} vs threshold={AUTO_BACKEND_NEURON_THRESHOLD})"
            )
        elif backend not in ("cuda", "cpu"):
            raise ValueError(
                f"未知の simulation.backend: {backend!r} (cuda | cpu | auto)"
            )
        sim_cfg["backend"] = backend
        return sim_cfg

    @staticmethod
    def _materialize_layout(layout_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """layout.assignment が未指定なら既定値に確定させた copy を返す。

        seed と同じ理由でここで確定させる。assignment は「どのグローバルIDがどの
        population か」= E/I の配置を決めるので、値が記録に残らないと保存済み config から
        レイアウトを復元できない。既定値の所在をコードではなく記録側に置くことで、
        既定を変えても保存済み config の意味は動かない。
        """
        layout_cfg = dict(layout_cfg or {})
        if layout_cfg.get("assignment") is None:
            layout_cfg["assignment"] = DEFAULT_ASSIGNMENT
            print(
                f"[ConfigManager] layout.assignment が未指定のため既定値を適用しました: "
                f"{DEFAULT_ASSIGNMENT}"
            )
        return layout_cfg

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """指定されたPathのYAMLファイルを読み込む（ファイルがない場合は空辞書を返す）"""
        if not filepath.exists():
            print(f"Warning: Missing config file: {filepath}. Returning empty dict.")
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def resolve(self, main_path: str, active_task: str ) -> AppConfig:
        """
        全YAMLファイルを統合し、Pydanticで型検証された AppConfig を生成する。
        """
        config_dir = Path("configs")
        # サブファイルをまとめるディレクトリ
        components_dir = config_dir / "components" 
        main_config_path = Path(main_path)
        main_cfg = self._load_yaml(main_config_path)
        # save_config() が source_config.yaml を書き出せるように入力元を覚えておく。
        self._source_path = main_config_path

        # 統合用の辞書を構築
        resolved = {
            # seed / backend 未指定なら ここで実値に確定させる (以降 None も "auto" も現れない)
            "simulation": self._materialize_backend(
                self._materialize_seed(main_cfg["simulation"]), main_cfg["neurons"]
            ),
            "inputs": main_cfg["inputs"],
            "neurons": {},
            "synapses": {},
            "network": {},
            # layout も未指定なら ここで実値に確定させる (以降 None は現れない)
            "layout": self._materialize_layout(main_cfg.get("layout")),
            "task": {},
            "meta": {"timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}
        }

        # コンポーネントYAMLの事前読み込み
        neurons_data = self._load_yaml(components_dir / "neurons.yaml")
        synapses_data = self._load_yaml(components_dir / "synapses.yaml")
        plasticity_data = self._load_yaml(components_dir / "plasticity.yaml")

        # --- 1. ニューロン設定の解決 ---
        for group_name, n_cfg in main_cfg["neurons"].items():
            resolved["neurons"][group_name] = n_cfg.copy()
            n_type, n_mode = n_cfg["type"], n_cfg["mode"]
            # components/neurons.yaml から特定の type > mode のパラメータをマージ
            if n_type in neurons_data and n_mode in neurons_data[n_type]:
                resolved["neurons"][group_name].update(neurons_data[n_type][n_mode])

        # --- 2. シナプス＆可塑性設定の解決 ---
        for group_name, s_cfg in main_cfg["synapses"].items():
            resolved["synapses"][group_name] = s_cfg.copy()

            # plasticity (STDP等)
            plas_info = s_cfg["plasticity"]
            p_type, p_mode = plas_info["type"], plas_info["mode"]
            if p_type in plasticity_data and p_mode in plasticity_data[p_type]:
                resolved["synapses"][group_name]["plasticity"].update(plasticity_data[p_type][p_mode])
                
            # synapse (コンダクタンスや放出ダイナミクス)
            syn_info = s_cfg["synapse"]
            s_type, s_mode = syn_info["type"], syn_info["mode"]
            if s_type in synapses_data and s_mode in synapses_data[s_type]:
                resolved["synapses"][group_name]["synapse"].update(synapses_data[s_type][s_mode])

        # メイン設定ファイルの network ブロックを取得
        network = main_cfg["network"]

        # 読み込むべきコンポーネントYAMLと、test.yaml 内のキー名のマッピング
        network_map = {
            "space.yaml": ("space", network["space"]),
            "connections.yaml": ("connection", network["connection"]),
            "weights.yaml": ("weight", network["weight"]),
            "delays.yaml": ("delay", network["delay"]),
        }

        for yaml_file, (key_name, profile_name) in network_map.items():
            data = self._load_yaml(components_dir / yaml_file)
            profile_data = data[profile_name].copy()
            profile_data["profile_name"] = profile_name
            resolved["network"][key_name] = profile_data

        # 疎/密の生成経路の選択 (未指定なら "auto")
        resolved["network"]["sparse"] = network.get("sparse", "auto")

        # タスク設定の読み込み
        tasks_data = self._load_yaml(components_dir / "tasks.yaml")
        profile_data = tasks_data[active_task].copy()
        profile_data["profile_name"] = active_task
        resolved["task"] = profile_data
        # ★ 最後にPydanticモデルに流し込んで検証（Validation）を行う
        try:
            validated_config = AppConfig(**resolved)
        except Exception as e:
            raise ValueError(f"Config validation failed: {e}")
        self._check_synapse_sources(validated_config)
        return validated_config

    @staticmethod
    def _check_synapse_sources(config: AppConfig) -> None:
        """シナプス群の source と population の polarity の整合を確認する。

        Pydantic では表現できない「セクションを跨いだ」制約なので resolve() の最後で見る。
        `load_resolved()` からは呼ばない (保存済み config は実行時に検証済みの記録であり、
        解析のたびに同じ警告を出しても仕方がないため)。

        source の重複を弾くのは**データモデル上の矛盾**だから。`synapses` には target
        フィールドが無く、1 グループが「その source が出す全投射」を丸ごと定義する。
        よって同じ population が 2 グループの source になると、(src, tgt) ペアに対して
        どちらの可塑性/コンダクタンスが効くのかが定義できない。GeNN 側も集団名
        f"{src}_to_{tgt}" の重複で必ず落ちるが、そこで見えるのは症状 (名前の衝突) だけで、
        原因 (どの 2 グループが source を共有しているか) が分からない。

        ※ ターゲット別に可塑性を変えたい場合、集団名にグループ名を混ぜて衝突を回避するのは
          **誤り**。同じ (src, tgt) ペアに同一の結合を持つシナプス集団が 2 つできて電流が
          二重に入る。正しい対処は target フィールドの導入 (現状はコメントアウト)。

        残り 2 件は設計ミスの可能性が高いだけなので警告に留める。
        """
        owner: Dict[str, str] = {}
        for group_name, syn_cfg in config.synapses.items():
            for src_name in syn_cfg.source:
                if src_name not in config.neurons:
                    raise ValueError(
                        f"synapses.{group_name}.source の {src_name!r} は "
                        f"neurons に存在しません (定義済み: {list(config.neurons)})。"
                    )
                if src_name in owner:
                    raise ValueError(
                        f"population {src_name!r} が synapses.{owner[src_name]} と "
                        f"synapses.{group_name} の両方の source になっています。"
                        f" synapse group は source が出す全投射を定義する (target 指定が無い)"
                        f" ため、{src_name!r} の投射にどちらの可塑性/コンダクタンスが効くのかが"
                        f" 決まりません。1 つの population は 1 つの synapse group にだけ属せます。"
                    )
                owner[src_name] = group_name

                # polarity と シナプスの符号の食い違い。片方だけ直したときに起きる。
                syn_mode = syn_cfg.synapse.mode
                polarity = config.neurons[src_name].polarity
                if syn_mode in ("excitatory", "inhibitory") and syn_mode != polarity:
                    warnings.warn(
                        f"neurons.{src_name}.polarity={polarity!r} ですが、これを source と"
                        f" する synapses.{group_name}.synapse.mode={syn_mode!r} です。"
                        f" 結合生成 (layout.ids_by('polarity')) と GeNN のコンダクタンス符号が"
                        f" 食い違ったネットワークになります。",
                        stacklevel=2,
                    )

        silent = [name for name in config.neurons if name not in owner]
        if silent:
            warnings.warn(
                f"どの synapses.*.source にも現れない population があります: {silent}。"
                f" これらのニューロンは発火しても出力を持ちません。",
                stacklevel=2,
            )

    def load_resolved(self, resolved_config) -> AppConfig:
        """
        過去にresolveした.yamlファイルを読み込む

        保存済み config は**実験の記録**なので、seed が null でも実値を捏造しない。
        捏造すると「再現できないもの」が再現できるように見えてしまう。警告だけ出し、
        実害が出るケース (layout.assignment='random') は NetworkLayout 側が弾く。

        layout.assignment も同じ理由で補完しない。記録に無ければ検証エラーで落ちる。

        simulation.backend も補完しない。ただし assignment と違い**検証は通す** (None 可)。
        backend はレイアウト復元に不要なので、backend フィールドが無い時代の config.yaml でも
        解析はできるべきだから。再実行しようとした時点で NetworkBuilder が弾く。
        """
        config_path = Path(resolved_config)
        resolved = self._load_yaml(config_path)
        sim = resolved.get("simulation") or {}
        if sim.get("seed") is None:
            print(
                f"Warning: {config_path} には simulation.seed が記録されていません。"
                " この run のネットワーク実現は再現できません。"
            )
        if sim.get("backend") is None:
            print(
                f"Warning: {config_path} には simulation.backend が記録されていません。"
                " この run が CPU/GPU どちらの結果かは判別できません"
                " (解析には支障ありませんが、再実行するには backend の指定が必要です)。"
            )
        try:
            validated_config = AppConfig(**resolved)
            return validated_config
        except Exception as e:
            raise ValueError(f"Config validation failed: {e}")


    def save_config(self, resolved_config: AppConfig, save_dir: str | Path = "results") -> Path:
        """実験の証拠としてコンフィグを保存する。

        2 ファイルを書き出す:

        - ``config.yaml``        … 結合済み・検証済みの解決後 config。seed は実値。
                                   `runio.resolve_layout()` や解析スクリプトが読む記録で、
                                   `load_resolved()` に渡せばそのまま再実行できる。
        - ``source_config.yaml`` … `resolve()` に渡した入力 YAML の逐語コピー
                                   (コメントも seed: null もそのまま)。「何を書いて
                                   投げたか」の記録。再現に必要な seed は config.yaml 側。

        `load_resolved()` 経由で入力元が無い場合は config.yaml のみ書き出す。

        Returns:
            書き出した config.yaml のパス。
        """
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "config.yaml"

        # Pydanticモデルを辞書に変換して保存。safe_load で読み戻せるよう、
        # numpy スカラ等を Python ネイティブに落としてから safe_dump する。
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                _to_python_native(resolved_config.model_dump()),
                f, default_flow_style=False, sort_keys=False, allow_unicode=True,
            )

        if self._source_path is not None and self._source_path.exists():
            shutil.copy2(self._source_path, out_dir / "source_config.yaml")

        return out_path
    


if __name__ == "__main__":
    import json

    # 1. ConfigManagerの初期化
    # test.yamlを読み込み、active_taskとして 'pqn_test' を指定
    manager = ConfigManager()

    # 2. 設定の統合と検証を実行
    print("--- Resolving Config ---")
    config = manager.resolve("configs/test.yaml", "lif_test")

    # 3. 読み込み結果の確認
    print(f"Successfully loaded timestamp: {config.meta.timestamp}")
    print(f"Simulation DT: {config.simulation.dt} ms")
    # print(f"Backend: {config.backend if hasattr(config, 'backend') else config.simulation.backend}")

    # 4. ネストされたデータのアクセス確認
    # NetworkConfig 内の各コンポーネントが正しく展開されているか
    print(f"Weight Type: {config.network.weight.profile_name}")
    
    # 5. タスク設定が正しく読み込まれているか
    if config.task:
        print(f"Active Task Duration: {config.task.duration} ms")

    # 6. 保存機能のテスト
    save_path = manager.save_config(config, save_dir="results")
    print(f"--- Config saved to: {save_path} ---")

    # デバッグ用：全データの構造を表示（辞書形式）
    # print(json.dumps(config.model_dump(), indent=2, ensure_ascii=False))
