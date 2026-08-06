from abc import ABC, abstractmethod
from typing import Dict, Iterator, Tuple, Any, List
import numpy as np

from src.core.config_manager import AppConfig
from src.core.layout import NetworkLayout

# 入力データ用のシード派生オフセット。NetworkBuilder の RandomState(seed)(結合・重み・遅延)
# とは別ストリームにするために seed へ加算する定数。
#
# なぜ必要か: RandomState は決定論的なので、同じ seed で作った 2 つのインスタンスは
# **先頭から同じ数列**を吐く。オフセット無しだと「刺激の乱数」と「ネットワーク構造の乱数」が
# 同じ乱数を共有し、入力と構造の間に隠れた依存が生まれる (例: 教師ラベル列と最初の
# ニューロン群の座標が同じ生ビットから作られる)。NetworkLayout が E/I 割当で
# `_ASSIGN_SEED_OFFSET` を使って避けているのと同じリスク。
#
# 値は layout 側と別の素数。オフセット同士が近いと意味がないわけではない (MT19937 の
# 初期化は seed を撹拌するので隣接 seed でもストリームは独立) が、由来の違う定数にしておく。
_LOADER_SEED_OFFSET = 15485863


class BaseDataLoader(ABC):
    def __init__(self, config: 'AppConfig', layout: NetworkLayout):
        self.config = config
        self.layout = layout

        # ネットワーク全体のニューロン数を取得
        self.total_neurons = self.layout.total_neurons

        self.dt = self.config.simulation.dt
        self.duration = config.task.duration
        self.total_steps = int(self.duration / self.dt)
        
        # seed が None (= 記録の無い保存済み config を読んだ場合) のときは OS エントロピー
        # 初期化に任せる。ここで実値を捏造すると「再現できない run」が再現できるように
        # 見えてしまうため (ConfigManager.load_resolved と同じ立場)。resolve() を通っていれば
        # seed は必ず実値なので、この分岐に落ちるのは記録が欠けている場合だけ。
        seed = getattr(self.config.simulation, 'seed', None)
        if seed is None:
            self.rng = np.random.RandomState(None)
        else:
            self.rng = np.random.RandomState((int(seed) + _LOADER_SEED_OFFSET) % (2 ** 32))

    @abstractmethod
    def generate(self) -> Iterator[Tuple[List[Tuple[Dict[str, np.ndarray], int]], Dict[str, Any]]]:
        """
        【候補1のインターフェース】
        データを、意味のある単位（1トライアル等）で逐次生成する。
        """
        pass

    def load_all(self) -> Tuple[List[List[Tuple[Dict[str, np.ndarray], int]]], List[Dict[str, Any]]]:
        """
        【候補2のインターフェース】（基底クラスで共通実装）
        generate() を最後まで回し、すべての結果をリスト化して一括で返す。
        """
        all_inputs = []
        all_metadata = []
        
        for inputs, metadata in self.generate():
            all_inputs.append(inputs)
            all_metadata.append(metadata)
            
        return all_inputs, all_metadata

    def format_global_to_group(self, global_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        [共通インターフェース]
        グローバルインデックスベースのデータ(形状: [total_neurons])を受け取り、
        Simulatorが受け付けるグループベースの辞書 {"pop_name": tensor} に変換する。
        """
        return self.layout.split_global_to_local(global_tensor)
