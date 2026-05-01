from abc import ABC, abstractmethod
from typing import Union, Any, Dict

class BasePlasticityModel(ABC):
    """
    バックエンド(GeNN等)にWeight Updateモデル情報を渡すための統一インターフェース
    シナプス結合の重み更新（STDPやSTPなど）を制御します。

    【運用のルール】
    1. 変数管理: vars(重みg), pre_vars(スパイク等), post_varsを適切に初期化します。
    2. ロジック記述: プレ/ポスト発火時の重み更新ロジックをC++文字列として定義します。
    3. インターフェース統一: NetworkBuilderは本クラスのプロパティを一律に呼び出すため、
        全ての具象クラスでこれらを正しく実装する必要があります。
    4. 二重登録防止: クラス変数(_registered_snippets)を活用し、同一モデルの多重定義を回避します。
    """
    def __init__(self, config, dt, weight, delay, num_pre, num_post):
        self.config = config
        self.dt = dt
        self.weight = weight
        self.delay = delay
        self.num_pre = num_pre
        self.num_post = num_post
        # カスタムスニペットを保持し、GCによる破棄を防ぐための変数
        self._custom_snippet_obj = None

    @property
    @abstractmethod
    def snippet(self) -> Union[str, Any]:
        """GeNNの snippet 引数に渡すオブジェクト"""
        pass

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """GeNNに渡す定数パラメータ辞書"""
        pass

    @property
    @abstractmethod
    def vars(self) -> Dict[str, Any]:
        """GeNNに渡す変数初期値辞書"""
        pass

    @property
    @abstractmethod
    def pre_vars(self) -> Dict[str, Any]:
        """GeNNに渡すpreneuron変数初期値辞書"""
        pass

    @property
    @abstractmethod
    def post_vars(self) -> Dict[str, Any]:
        """GeNNに渡すpostneuron変数初期値辞書"""
        pass

    # --- 以下、PyGeNNの init_weight_update で要求される参照系のデフォルト実装 ---
    @property
    def var_refs(self) -> Dict[str, Any]:
        return {}

    @property
    def pre_var_refs(self) -> Dict[str, Any]:
        return {}

    @property
    def post_var_refs(self) -> Dict[str, Any]:
        return {}
        
    @property
    def psm_var_refs(self) -> Dict[str, Any]:
        return {}