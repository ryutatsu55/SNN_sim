import pygenn
from .BASE_neuron import BaseNeuronModel
from src.core.registry import NEURON_MODELS

@NEURON_MODELS.register("LIF")
class LIF(BaseNeuronModel):
    """
    PyGeNNの組み込みLIFモデルを使用するクラス。
    膜電位の減衰、閾値判定、リセット、および不応期のシミュレーションを行います。
    """
    def __init__(self, config, dt):
        super().__init__(config, dt)

    @property
    def model_class(self):
        """
        GeNN組み込みのLIFモデル定義を返します。
        """
        return pygenn.genn_model.neuron_models.LIF()

    @property
    def params(self) -> dict:
        """
        YAML設定(self.config)からGeNNのLIFモデルが要求する定数パラメータをマッピングします。
        """
        return {
            "C":         float(self.config.C),         # 膜容量 [nF]
            "TauM":      float(self.config.TauM),      # 膜時定数 [ms]
            "Vrest":     float(self.config.Vrest),     # 静止膜電位 [mV]
            "Vthresh":   float(self.config.Vthresh),   # 発火閾値 [mV]
            "Vreset":    float(self.config.Vreset),    # リセット電位 [mV]
            "Ioffset":   float(self.config.Ioffset),   # 定常注入電流 [nA]
            "TauRefrac": float(self.config.TauRefrac)  # 不応期 [ms]
        }

    @property
    def vars(self) -> dict:
        """
        ニューロンの初期状態変数を設定します。
        """
        return {
            "V": self.config.Vrest, # 初期膜電位（通常は静止電位）
            "RefracTime": 0.0       # 残り不応期
        }
    
@NEURON_MODELS.register("test_LIF")
class test_LIF(BaseNeuronModel):
    """
    テスト用カスタムLIFモデル
    膜電位の減衰、閾値判定、リセット、および不応期のシミュレーションを行います。
    """
    def __init__(self, config, dt):
        super().__init__(config, dt)

    @property
    def model_class(self):
        # 共通の Piecewise Quadratic 関数 (f, g, h) の定義
        # fma (Fused Multiply-Add) を使用して計算精度と速度を向上
        sim_code = """
            Isyn_rec = Isyn;
            if (RefracTime <= 0.0) {
                scalar Alpha = dt / TauM;
                V += - (V - Vrest) * Alpha + (Iext + Isyn + Ioffset) * (TauM / C) * Alpha;
            }
            else {
                RefracTime -= dt;
            }
        """

        reset_code = """
            RefracTime = TauRefrac;
            V = Vreset;

        """

        return pygenn.create_neuron_model(
            "test_lif",
            params=list(self.params.keys()),
            vars=[
                ("V", "scalar"), 
                ("RefracTime", "scalar"),
                ("Iext", "scalar"), 
                ("Isyn_rec", "scalar")
                ],
            sim_code=sim_code,
            threshold_condition_code="V >= Vthresh",
            reset_code=reset_code
        )

    @property
    def params(self) -> dict:
        """
        YAML設定(self.config)からGeNNのLIFモデルが要求する定数パラメータをマッピングします。
        """
        return {
            "C":         float(self.config.C),         # 膜容量 [nF]
            "TauM":      float(self.config.TauM),      # 膜時定数 [ms]
            "Vrest":     float(self.config.Vrest),     # 静止膜電位 [mV]
            "Vthresh":   float(self.config.Vthresh),   # 発火閾値 [mV]
            "Vreset":    float(self.config.Vreset),    # リセット電位 [mV]
            "Ioffset":   float(self.config.Ioffset),   # 定常注入電流 [nA]
            "TauRefrac": float(self.config.TauRefrac)  # 不応期 [ms]
        }

    @property
    def vars(self) -> dict:
        """
        ニューロンの初期状態変数を設定します。
        """
        return {
            "V": self.config.Vrest, # 初期膜電位（通常は静止電位）
            "RefracTime": 0.0,       # 残り不応期
            "Iext" : 0.0,
            "Isyn_rec": 0.0
        }