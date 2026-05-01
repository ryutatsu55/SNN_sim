from .BASE_synapse import BaseSynapseModel
from typing import Any, Dict
import pygenn
from src.core.registry import SYNAPSE_MODELS

@SYNAPSE_MODELS.register("custom_example")
class CustomSynapseExample(BaseSynapseModel):
    # クラス変数でスニペットをキャッシュ（二重登録防止）
    _snippet_cache = {}

    def __init__(self, config, dt, pop=None):
        super().__init__(config, dt, pop)
        
        class_name = "custom_example"
        if class_name not in CustomSynapseExample._snippet_cache:
            # 未登録の場合のみ生成
            CustomSynapseExample._snippet_cache[class_name] = pygenn.model.create_postsynaptic_model(
                class_name,
                params=["tau_decay"],
                vars=[("I", "scalar")],
                sim_code="injectCurrent(inSyn);" 
            )
        
        self._custom_snippet_obj = CustomSynapseExample._snippet_cache[class_name]

    @property
    def snippet(self):
        return self._custom_snippet_obj

    @property
    def params(self):
        return {"tau_decay": self.config.tau_decay}

    @property
    def vars(self):
        return {"I": 0.0}
    
    @property
    def var_refs(self) -> Dict[str, Any]:
        return {"V": pygenn.genn_model.create_var_ref(self.pop, "V")}