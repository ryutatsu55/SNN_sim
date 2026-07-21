import math

import pygenn

from src.core.registry import NEURON_MODELS
from .BASE_neuron import BaseNeuronModel


def calculate_escape_noise_scale(dt: float, frest: float, v_rest: float, v_th: float, b: float) -> float:
    """Supplementary Eq. S6。"""
    return frest * (0.001 * dt) * math.exp(-(v_rest - v_th) / b)


def resolve_escape_gain(b, dt: float, f_rest: float, v_rest: float, v_th: float) -> float:
    """escape gain b を解決する。数値ならそのまま、"auto" なら自動計算する。

    このニューロンモデルの発火確率は硬い閾値ではなく滑らかな escape noise
        SpikeProb(V) = CScale · exp((V - v_th) / b),
        CScale = f_rest · 0.001 · dt · exp(-(v_rest - v_th) / b)   (Eq.S6)
    なので、V が v_th に達しても発火確率は CScale (通常は微小) にしかならず、
    「閾値」が実質的に機能しない。b="auto" はこの CScale を 1 に固定する b を選ぶ。

    V=v_th で SpikeProb = CScale = 1 と置くと
        f_rest · 0.001 · dt · exp((v_th - v_rest) / b) = 1
        ⟹ b = (v_th - v_rest) / ln(1 / (f_rest · 0.001 · dt))

    Args:
        b: 数値、または (大小無視で) "auto"。
        dt: タイムステップ [ms]。
        f_rest: 静止時の自発発火率 [Hz]。
        v_rest: 静止膜電位 [mV]。
        v_th: 閾値膜電位 [mV]。
    """
    if isinstance(b, str):
        if b.strip().lower() != "auto":
            raise ValueError(f"b は数値または 'auto' を指定してください (got {b!r})。")
        p_rest = f_rest * (0.001 * dt)  # 静止時 (V=v_rest) の 1 ステップ発火確率
        if not (0.0 < p_rest < 1.0):
            raise ValueError(
                "b='auto' の計算には 0 < f_rest·0.001·dt < 1 が必要です "
                f"(f_rest={f_rest}, dt={dt}, 積={p_rest})。"
            )
        if v_th <= v_rest:
            raise ValueError(
                f"b='auto' の計算には v_th > v_rest が必要です (v_th={v_th}, v_rest={v_rest})。"
            )
        return (v_th - v_rest) / math.log(1.0 / p_rest)
    return float(b)


def conductance_lif_delta(v: float, isyn: float, dt: float, tau_m: float, v_rest: float, i_ext: float = 0.0) -> float:
    """Supplementary Eq. S1 を Isyn 入力として Euler 法で 1 ステップ進める。"""
    return (dt / tau_m) * ((v_rest - v) + isyn + i_ext)


def conductance_synaptic_current(v: float, g_exc: float, g_inh: float, e_exc: float, e_inh: float) -> float:
    """Supplementary Eq. S1 のシナプス項を計算する。"""
    return ((e_exc - v) * g_exc) + ((e_inh - v) * g_inh)


def conductance_lif_delta_from_conductances(
    v: float,
    g_exc: float,
    g_inh: float,
    dt: float,
    tau_m: float,
    v_rest: float,
    e_exc: float,
    e_inh: float,
    i_ext: float = 0.0,
) -> float:
    """Supplementary Eq. S1 を conductance 入力から Euler 法で 1 ステップ進める。"""
    isyn = conductance_synaptic_current(v, g_exc, g_inh, e_exc, e_inh)
    return conductance_lif_delta(v=v, isyn=isyn, dt=dt, tau_m=tau_m, v_rest=v_rest, i_ext=i_ext)


def escape_noise_probability(v: float, v_th: float, b: float, scale_c: float) -> float:
    """Supplementary Eq. S5。"""
    return min(scale_c * math.exp((v - v_th) / b), 1.0)


def evolve_escape_lif_step(
    v: float,
    refrac_time: float,
    isyn: float,
    i_ext: float,
    dt: float,
    tau_m: float,
    v_rest: float,
    v_th: float,
    b: float,
    scale_c: float,
    tau_refrac: float,
    random_uniform: float,
) -> tuple[float, float, bool, float]:
    """escape-noise LIF の純 Python ステップ更新。"""
    if refrac_time > 0.0:
        next_refrac = max(refrac_time - dt, 0.0)
        return v, next_refrac, False, 0.0

    updated_v = v + conductance_lif_delta(v, isyn, dt, tau_m, v_rest, i_ext)
    spike_prob = escape_noise_probability(updated_v, v_th, b, scale_c)
    spiked = random_uniform < spike_prob
    if spiked:
        return v_rest, tau_refrac, True, spike_prob
    return updated_v, 0.0, False, spike_prob


@NEURON_MODELS.register("akita_escape_lif")
class AkitaEscapeLIF(BaseNeuronModel):
    """
    Akita APL 2023 Supplementary の escape-noise 付き conductance-based LIF。
    """
    def __init__(self, config, dt):
        super().__init__(config, dt)
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
        sim_code = """
            if (RefracTime <= 0.0) {
                V += ((Vrest - V) + Isyn + Iext) * (dt / TauM);
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
            "akita_escape_lif",
            params=list(self.params.keys()),
            vars=[
                ("V", "scalar"),
                ("RefracTime", "scalar"),
                ("Iext", "scalar"),
                ("SpikeProb", "scalar"),
            ],
            sim_code=sim_code,
            threshold_condition_code="gennrand_uniform() < SpikeProb",
            reset_code=reset_code,
        )

    @property
    def params(self) -> dict:
        return {
            "TauM": float(self.config.tau_m),
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
            "Iext": 0.0,
            "SpikeProb": 0.0,
        }
