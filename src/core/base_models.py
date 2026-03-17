# src/core/base_models.py
from abc import ABC, abstractmethod
import numpy as np

class BaseNeuron(ABC):
    """すべてのニューロンモデルの親クラス"""
    
    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons

    @abstractmethod
    def get_initial_states(self) -> dict:
        """
        GPUに転送する状態変数の初期値（Numpy配列）を辞書で返す
        例: {"V": np.zeros(N), "U": np.zeros(N)}
        """
        pass

    @abstractmethod
    def get_cuda_components(self) -> dict:
        """
        Jinja2テンプレートに流し込む、このモデル専用の数式（C++文字列）を返す
        """
        pass

    @abstractmethod
    def get_constant_memory(self) -> dict:
        """
        __constant__ メモリに転送するデータの辞書を返す
        例: {"AllParams": np.array(...)}
        使わないモデルは空の辞書 {} を返せばOK
        """
        pass
    