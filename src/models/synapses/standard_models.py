from typing import Union, Any, Dict
import pygenn
from .BASE_synapse import BaseSynapseModel
from src.core.registry import SYNAPSE_MODELS

@SYNAPSE_MODELS.register("ExpCurr")
class ExpCurrSynapse(BaseSynapseModel):
    @property
    def snippet(self):
        return "ExpCurr" # 文字列を返す

    @property
    def params(self):
        return {"tau": self.config.tau}

    @property
    def vars(self):
        return {}

@SYNAPSE_MODELS.register("ExpCond")
class ExpCond(BaseSynapseModel):
    @property
    def snippet(self):
        return "ExpCond" # 文字列を返す

    @property
    def params(self):
        return {
            "tau": self.config.tau,
            "E": self.config.E
            }

    @property
    def vars(self):
        return {}
    
    @property
    def var_refs(self) -> Dict[str, Any]:
        """
        GeNNに渡す変数参照辞書
        """
        return {"V": pygenn.create_var_ref(self.pop, "V")}
