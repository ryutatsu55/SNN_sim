from abc import ABC, abstractmethod

class BaseNeuronModel(ABC):
    """
    バックエンド(GeNN等)にモデル情報を渡すための統一インターフェース
    """
    def __init__(self, config, dt):
        self.config = config
        self.dt = dt

    @property
    @abstractmethod
    def model_class(self):
        """GeNNのモデル定義オブジェクトを返す"""
        pass

    @property
    @abstractmethod
    def params(self) -> dict:
        """GeNNに渡す定数パラメータ辞書"""
        pass

    @property
    @abstractmethod
    def vars(self) -> dict:
        """GeNNに渡す変数初期値辞書"""
        pass