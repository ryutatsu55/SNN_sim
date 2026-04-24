import pygenn
import numpy as np
from src.core.registry import PLASTICITY_MODELS
from .BASE_plasticity import BasePlasticityModel

@PLASTICITY_MODELS.register("STDP")
class STDP(BasePlasticityModel):
    def __init__(self, config, dt, weight, delay):
        super().__init__(config, dt, weight, delay)
        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()

    def _prepare_genn_data(self):
        params = {
            "tauPlus": float(self.config.tauPlus),
            "tauMinus": float(self.config.tauMinus),
            "Aplus": float(self.config.Aplus),
            "Aminus": float(self.config.Aminus),
            "Wmin": float(self.config.Wmin),
            "Wmax": float(self.config.Wmax)
        }
        vars = {
            "g": self.weight.astype('float32')
        }
        pre_vars = {}
        post_vars = {}

        return params, vars, pre_vars, post_vars
    
    @property
    def model_class(self):
        return pygenn.weight_update_models.STDP()
    
    @property
    def params(self): return self._params

    @property
    def vars(self): return self._vars

    @property
    def pre_vars(self): return self._pre_vars
    
    @property
    def post_vars(self): return self._post_vars

@PLASTICITY_MODELS.register("StaticPulse")
class StaticPulse(BasePlasticityModel):
    """
    Pulse-coupled, static synapse with heterogeneous weight. 
    No learning rule is applied to the synapse and for each pre-synaptic spikes,
    the synaptic conductances are simply added to the postsynaptic input variable.
    """
    def __init__(self, config, dt, weight, delay):
        super().__init__(config, dt, weight, delay)
        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()

    def _prepare_genn_data(self):
        params = {}
        vars = {
            "g": self.weight.astype('float32')
        }
        pre_vars = {}
        post_vars = {}

        return params, vars, pre_vars, post_vars
    
    @property
    def model_class(self):
        return pygenn.weight_update_models.StaticPulse()
    
    @property
    def params(self): return self._params

    @property
    def vars(self): return self._vars

    @property
    def pre_vars(self): return self._pre_vars
    
    @property
    def post_vars(self): return self._post_vars

@PLASTICITY_MODELS.register("StaticPulseConstantWeight")
class StaticPulseConstantWeight(BasePlasticityModel):
    """
    Pulse-coupled, static synapse with homogeneous weight. 
    No learning rule is applied to the synapse and for each pre-synaptic spikes,
    the synaptic conductances are simply added to the postsynaptic input variable.
    """
    def __init__(self, config, dt, weight, delay):
        super().__init__(config, dt, weight, delay)
        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()

    def _prepare_genn_data(self):
        params = {
            "g": float(self.config.g)  # 全シナプスで同じ重み
        }
        vars = {}
        pre_vars = {}
        post_vars = {}

        return params, vars, pre_vars, post_vars
    
    @property
    def model_class(self):
        return pygenn.weight_update_models.StaticPulseConstantWeight()
    
    @property
    def params(self): return self._params

    @property
    def vars(self): return self._vars

    @property
    def pre_vars(self): return self._pre_vars
    
    @property
    def post_vars(self): return self._post_vars