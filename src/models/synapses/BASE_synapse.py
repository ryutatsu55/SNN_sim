from abc import ABC, abstractmethod
from typing import Union, Any, Dict

class BaseSynapseModel(ABC):
    """
    バックエンド(GeNN等)にポストシナプスメカニズムの情報を渡すための統一インターフェース
    スパイクを受け取った後の「電流」や「コンダクタンス」の挙動を定義します。

    【運用のルール】
    1. 標準モデル: snippetプロパティで名称を文字列("ExpCurr"等)で返します。GeNN内部でV等が自動解決されます。
    2. カスタムモデル: create_postsynaptic_modelを使用。二重登録エラー防止のためクラス変数でのキャッシュを推奨。
    3. 変数参照: self.popを通じてターゲットニューロンの変数(膜電位V等)をvar_refsで参照可能です。
    4. GC対策: 生成したカスタムオブジェクトは self._custom_snippet_obj に保持し、NetworkBuilder側でも保護します。
    """
    def __init__(self, config, dt, pop=None):
        self.config = config
        self.dt = dt
        self.pop = pop  # ターゲットニューロン集団への参照（必要な場合のみ）
        # GC対策: カスタムスニペットを生成した場合、インスタンス変数として保持する
        self._custom_snippet_obj = None 

    @property
    @abstractmethod
    def snippet(self) -> Union[str, Any]:
        """
        GeNNの snippet 引数に渡すオブジェクト。
        標準モデルなら文字列(例: "ExpCurr")、カスタムモデルなら生成したスニペットクラスを返す。
        """
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
    def var_refs(self) -> Dict[str, Any]:
        """
        GeNNに渡す変数参照辞書。
        標準モデルでは使わないことが多いが、カスタムモデルのためにデフォルトを用意。
        """
        return {}