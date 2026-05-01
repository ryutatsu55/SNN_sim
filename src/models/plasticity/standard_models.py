import pygenn
import numpy as np
from src.core.registry import PLASTICITY_MODELS
from .BASE_plasticity import BasePlasticityModel

@PLASTICITY_MODELS.register("STDP")
class STDP(BasePlasticityModel):
    def __init__(self, config, dt, weight, delay, num_pre, num_post):
        super().__init__(config, dt, weight, delay, num_pre, num_post)
        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()
        # 初期化時に一度だけオブジェクトを生成して保持する
        self._snippet = "STDP"

    def _prepare_genn_data(self):
        params = {
            "tauPlus": float(self.config.tauPlus),
            "tauMinus": float(self.config.tauMinus),
            "Aplus": float(self.config.Aplus),
            "Aminus": float(self.config.Aminus),
            "Wmin": float(self.config.Wmin),
            "Wmax": float(self.config.Wmax)
        }
        vars = {"g": self.weight.astype('float32')}
        pre_vars = {}
        post_vars = {}

        return params, vars, pre_vars, post_vars

    @property
    def snippet(self): return self._snippet
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
    def __init__(self, config, dt, weight, delay, num_pre, num_post):
        super().__init__(config, dt, weight, delay, num_pre, num_post)
        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()
        self._snippet = "StaticPulse"

    def _prepare_genn_data(self):
        params = {}
        vars = {"g": self.weight.astype('float32')}
        pre_vars = {}
        post_vars = {}

        return params, vars, pre_vars, post_vars
    
    @property
    def snippet(self): return self._snippet
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
    def __init__(self, config, dt, weight, delay, num_pre, num_post):
        super().__init__(config, dt, weight, delay, num_pre, num_post)
        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()
        self._snippet = "StaticPulseConstantWeight"

    def _prepare_genn_data(self):
        params = {"g": float(self.config.g)}
        vars = {}
        pre_vars = {}
        post_vars = {}

        return params, vars, pre_vars, post_vars
    
    @property
    def snippet(self): return self._snippet
    @property
    def params(self): return self._params
    @property
    def vars(self): return self._vars
    @property
    def pre_vars(self): return self._pre_vars
    @property
    def post_vars(self): return self._post_vars