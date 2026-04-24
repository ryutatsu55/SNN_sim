from typing import Callable, Dict, Any, Type

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[Any]] = {}

    def register(self, name: str) -> Callable:
        """クラスをレジストリに登録するデコレータ"""
        def inner_wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            if name in self._registry:
                raise ValueError(f"Name '{name}' is already registered in {self.name}.")
            self._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def get(self, name: str) -> Type[Any]:
        """登録されたクラスを取得する"""
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available options: {list(self._registry.keys())}"
            )
        return self._registry[name]

# システム全体のレジストリインスタンス
DATA_LOADERS = Registry("DataLoaders")
NEURON_MODELS = Registry("NeuronModels")
SPATIAL_MODELS = Registry("SpatialModels")
CONNECTION_MODELS = Registry("ConnectionModels") # 結合行列(マスク)の生成
WEIGHT_MODELS = Registry("WeightModels")     # 重みの生成
DELAY_MODELS = Registry("DelayModels")       # 伝播遅延の生成
PLASTICITY_MODELS = Registry("PlasticityModels")     # シナプス可塑性の生成
SYNAPSE_MODELS = Registry("SynapseModels")   # シナプス力学(STP等)の生成
