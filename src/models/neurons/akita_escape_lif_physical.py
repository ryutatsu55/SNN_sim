"""物理単位版の escape-noise 付き conductance-based LIF (膜抵抗 R_m を明示的に持つ)。

既存の `akita_escape_lif` は Akita Supplementary Eq.S1

    tau_m dV/dt = (v_rest - V) + g_exc(E_exc - V) + g_inh(E_inh - V)

をそのまま実装しており、ここでの g は「漏れコンダクタンス g_rest で割った**無次元量**」である
(元の物理式 C_m dV/dt = g_rest(v_rest - V) + g_syn(E_syn - V) を g_rest で割った形)。
そのため g_max を実測コンダクタンス (nS など) と直接突き合わせられない。

本モデルは膜抵抗 R_m を明示的に導入し、積分式を

    V += ((Vrest - V) + Rm * (Isyn + Iext)) * (dt / TauM)

とすることで、シナプスコンダクタンスを**物理単位のまま**扱えるようにする。
可塑性モデル (custom_Akita) 側は無次元化を行わない (実効コンダクタンスをそのまま渡す) ので、
単位系はどちらのニューロンモデルを選ぶかだけで決まる。

単位系:
    コンダクタンス g, g_max : µS      (AMPA 単一シナプス 0.0005-0.0015 µS = 0.5-1.5 nS)
    膜抵抗 R_m              : MΩ      (皮質錐体細胞 100-200)
    電流 Isyn, Iext         : nA
    電位                    : mV
    導出 g_rest = 1/R_m     : µS      (R_m=100 -> 0.01 µS = 10 nS)
    導出 C_m = tau_m/R_m    : nF      (30 ms / 100 MΩ -> 0.3 nF = 300 pF)

    整合性: µS x mV = nA、MΩ x nA = mV、ms / MΩ = nF。

無次元版との対応:
    g_max_無次元 = g_max_物理 x R_m

**注意 — Iext と外部入力の単位が無次元版と異なる**:
    無次元版では `Iext` は「既に膜抵抗を掛けた後」の mV 相当だったが、本モデルでは **nA** である
    (積分式で Rm が掛かるため)。`sim.push()` で Iext を書くスクリプトや、
    `GaussianNoise` current source の mean/sd を物理版 config で使う場合は値の読み替えが要る。
"""

import pygenn

from src.core.registry import NEURON_MODELS
from .BASE_neuron import BaseNeuronModel
# escape noise のスケールと b の自動計算は電位のみの関数で単位に依存しないため既存実装を再利用する
from .akita_escape_lif import calculate_escape_noise_scale, resolve_escape_gain


def conductance_lif_delta_physical(
    v: float,
    isyn_nA: float,
    dt: float,
    tau_m: float,
    r_m: float,
    v_rest: float,
    i_ext_nA: float = 0.0,
) -> float:
    """物理単位版の 1 ステップ膜電位変化 [mV] (GeNN の sim_code と同じ数式)。

    Args:
        v: 現在の膜電位 [mV]
        isyn_nA: シナプス電流 [nA] (= g[µS] * (E - V)[mV])
        dt: タイムステップ [ms]
        tau_m: 膜時定数 [ms]
        r_m: 膜抵抗 [MΩ]
        v_rest: 静止電位 [mV]
        i_ext_nA: 外部注入電流 [nA]
    """
    return (dt / tau_m) * ((v_rest - v) + r_m * (isyn_nA + i_ext_nA))


def synaptic_current_nA(v: float, g_exc_uS: float, g_inh_uS: float,
                        e_exc: float, e_inh: float) -> float:
    """コンダクタンス [µS] と電位 [mV] からシナプス電流 [nA] を作る。

    µS x mV = nA。GeNN 側では後シナプスモデル (ExpCond) が同じ計算をして Isyn に入れる。
    """
    return ((e_exc - v) * g_exc_uS) + ((e_inh - v) * g_inh_uS)


def leak_conductance_uS(r_m: float) -> float:
    """膜抵抗 [MΩ] から漏れコンダクタンス [µS] を導出する。"""
    return 1.0 / r_m


def membrane_capacitance_nF(tau_m: float, r_m: float) -> float:
    """膜時定数 [ms] と膜抵抗 [MΩ] から膜容量 [nF] を導出する (C_m = tau_m / R_m)。"""
    return tau_m / r_m


@NEURON_MODELS.register("akita_escape_lif_physical")
class AkitaEscapeLIFPhysical(BaseNeuronModel):
    """膜抵抗 R_m を持つ物理単位版の escape-noise conductance-based LIF。

    無次元版 `AkitaEscapeLIF` との差は積分式に Rm が掛かる点のみで、
    escape noise・不応期・リセットの扱いは同一。
    """

    def __init__(self, config, dt):
        super().__init__(config, dt)
        self._r_m = float(self.config.R_m)
        if not (self._r_m > 0.0) or self._r_m == float("inf"):
            raise ValueError(
                f"R_m は正の有限値である必要があります (got {self.config.R_m})。"
                " 単位は MΩ で、皮質錐体細胞の代表値は 100-200 です。"
            )
        # b は数値のほか "auto" を許す (v=v_th で発火確率が 1.0 になる b を自動計算)
        self._b = resolve_escape_gain(
            self.config.b,
            dt=self.dt,
            f_rest=float(self.config.f_rest),
            v_rest=float(self.config.v_rest),
            v_th=float(self.config.v_th),
        )
        self._scale_c = calculate_escape_noise_scale(
            dt=self.dt,
            frest=float(self.config.f_rest),
            v_rest=float(self.config.v_rest),
            v_th=float(self.config.v_th),
            b=self._b,
        )

    @property
    def model_class(self):
        # 無次元版との唯一の違い: Isyn / Iext に膜抵抗 Rm を掛けて mV に変換する。
        #   Isyn [nA] x Rm [MΩ] = mV
        sim_code = """
            Isyn_rec = Isyn;
            if (RefracTime <= 0.0) {
                V += ((Vrest - V) + Rm * (Isyn + Iext)) * (dt / TauM);
                SpikeProb = fmin(CScale * exp((V - Vthresh) / B), 1.0);
            }
            else {
                RefracTime = fmax(RefracTime - dt, 0.0);
                SpikeProb = 0.0;
            }
        """

        reset_code = """
            V = Vreset;
            RefracTime = TauRefrac;
            SpikeProb = 0.0;
        """

        return pygenn.create_neuron_model(
            # 無次元版は "akita_escape_lif" を使うため、別名にしないと GeNN 側で衝突する
            "akita_escape_lif_physical",
            params=list(self.params.keys()),
            vars=[
                ("V", "scalar"),
                ("RefracTime", "scalar"),
                ("Iext", "scalar"),
                ("SpikeProb", "scalar"),
                ("Isyn_rec", "scalar"),
            ],
            sim_code=sim_code,
            threshold_condition_code="gennrand_uniform() < SpikeProb",
            reset_code=reset_code,
        )

    @property
    def params(self) -> dict:
        return {
            "TauM": float(self.config.tau_m),
            "Rm": self._r_m,
            "Vrest": float(self.config.v_rest),
            "Vthresh": float(self.config.v_th),
            "Vreset": float(self.config.v_rest),
            "B": self._b,
            "CScale": float(self._scale_c),
            "TauRefrac": float(self.config.tau_refrac),
        }

    @property
    def vars(self) -> dict:
        return {
            "V": float(self.config.v_rest),
            "RefracTime": 0.0,
            "Iext": 0.0,      # [nA] — 無次元版では mV 相当だった点に注意
            "SpikeProb": 0.0,
            "Isyn_rec": 0.0,
        }

    # --- 導出量 (config には持たせず R_m と tau_m から計算する) ---

    @property
    def g_rest_uS(self) -> float:
        """漏れコンダクタンス [µS] = 1 / R_m。"""
        return leak_conductance_uS(self._r_m)

    @property
    def c_m_nF(self) -> float:
        """膜容量 [nF] = tau_m / R_m。"""
        return membrane_capacitance_nF(float(self.config.tau_m), self._r_m)
