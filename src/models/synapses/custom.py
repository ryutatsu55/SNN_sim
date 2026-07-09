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
            CustomSynapseExample._snippet_cache[class_name] = pygenn.create_postsynaptic_model(
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
        return {"V": pygenn.create_var_ref(self.pop, "V")}


# =====================================================================
# g_max 飽和つき conductance-based 指数シナプス (ExpCond ベース)
# ---------------------------------------------------------------------
# GeNN 組み込み ExpCond は
#     injectCurrent(inSyn * (E - V)); inSyn *= expDecay;
# であり、inSyn が後ニューロンに集まる総コンダクタンス g_total = Σ x_release·w·g_max·g_scale·decay
# となる。この g_total を g_max で飽和させる 2 変種を用意する。
#   - ExpCondGClip : g_total を [0, g_max] に破壊的にクランプしてから計算 (状態も上限で頭打ち)
#   - ExpCondGCap  : g_total は素のまま減衰させ、計算 (injectCurrent) にのみ min(g_total, g_max) を使う
# g_max は plasticity 側と同値だが、設計をシンプルに保つため synapses.yaml で独立に与える。
# =====================================================================
class _BaseGmaxExpCond(BaseSynapseModel):
    """g_max 飽和つき ExpCond の共通基底。sim_code のみ派生で差し替える。"""
    # クラス変数でスニペットをキャッシュ（二重登録防止）。g_max は params 経由で渡すため
    # スニペット構造には焼き込まれず、クラス名 1 つでパラメータ変種を共有できる。
    _snippet_cache = {}

    #: create_postsynaptic_model に渡す GeNN クラス名（派生で上書き必須）
    _class_name = None
    #: injectCurrent する C++ sim_code（派生で上書き必須）
    _sim_code = None

    def __init__(self, config, dt, pop=None):
        super().__init__(config, dt, pop)
        cache = type(self)._snippet_cache
        if self._class_name not in cache:
            cache[self._class_name] = pygenn.create_postsynaptic_model(
                self._class_name,
                params=["tau", "E", "g_max"],
                neuron_var_refs=[("V", "scalar")],
                sim_code=self._sim_code,
            )
        self._custom_snippet_obj = cache[self._class_name]

    @property
    def snippet(self):
        return self._custom_snippet_obj

    @property
    def params(self):
        return {
            "tau": float(self.config.tau),
            "E": float(self.config.E),
            "g_max": float(self.config.g_max),
        }

    @property
    def vars(self):
        return {}

    @property
    def var_refs(self) -> Dict[str, Any]:
        return {"V": pygenn.create_var_ref(self.pop, "V")}


@SYNAPSE_MODELS.register("ExpCondGClip")
class ExpCondGClip(_BaseGmaxExpCond):
    """クリップ型: 受け取った g_total(inSyn) を [0, g_max] に破壊的にクランプしてから計算。

    超過分は毎ステップ恒久的に捨てられ、コンダクタンス状態そのものが g_max を超えない。
    """
    _class_name = "ExpCondGClip"
    _sim_code = """
        inSyn = fmin(fmax(inSyn, 0.0), g_max);
        injectCurrent(inSyn * (E - V));
        inSyn *= exp(-dt / tau);
    """


@SYNAPSE_MODELS.register("ExpCondGCap")
class ExpCondGCap(_BaseGmaxExpCond):
    """飽和計算型: g_total(inSyn) は素のまま減衰させ、計算にのみ min(g_total, g_max) を使う。

    inSyn は真の累積値のまま減衰し続けるので、電流だけが飽和する。入力が引けば即応答が戻る。
    """
    _class_name = "ExpCondGCap"
    _sim_code = """
        const scalar g_eff = fmin(fmax(inSyn, 0.0), g_max);
        injectCurrent(g_eff * (E - V));
        inSyn *= exp(-dt / tau);
    """