import math

from src.core.registry import NEURON_MODELS
from .BASE_neuron import BaseNeuronModel

try:
    import pygenn
except ImportError:  # pragma: no cover - GeNN未導入環境では数式検証のみ行う
    pygenn = None


def calculate_escape_noise_scale(dt: float, frest: float, v_rest: float, v_th: float, b: float) -> float:
    """Supplementary Eq. S6。"""
    return frest * dt * math.exp(-(v_rest - v_th) / b)


def conductance_lif_delta(v: float, isyn: float, dt: float, tau_m: float, v_rest: float, i_ext: float = 0.0) -> float:
    """Supplementary Eq. S1 を Euler 法で 1 ステップ進める。"""
    return (dt / tau_m) * ((v_rest - v) + isyn + i_ext)


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
        self._scale_c = calculate_escape_noise_scale(
            dt=self.dt,
            frest=float(self.config.f_rest),
            v_rest=float(self.config.v_rest),
            v_th=float(self.config.v_th),
            b=float(self.config.b),
        )

    @property
    def model_class(self):
        if pygenn is None:
            raise ImportError("pygenn is required to build akita_escape_lif.")

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
            "B": float(self.config.b),
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
