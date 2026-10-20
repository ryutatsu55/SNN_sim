"""切断後に一部のシナプス群の長期可塑性 (STDP) を止める。

`task.freeze_plasticity` にシナプス群の名前 (`config.synapses` のキー。例 `from_Inh`) を
並べると、その群の STDP の学習率を 0 にしたネットワークで切断後を走らせる。
**短期可塑性 (STP: U, tau_rec) は止めない** —— 伝達の仕方は損傷前と同じまま、重みだけを
切断の瞬間の値に固定する。回復が i-STDP と e-STDP のどちらから来るかを切り分けるための操作。

**止めるのは切断の瞬間から。** 親 run (損傷前の発達) は止めていない。

run の `config.yaml` の `synapses` は**親と同じ値のまま**残り、止めたことは `task` 側
(`freeze_plasticity`) と `lesion.json` (`frozen_plasticity`) に残る。
ネットワークの記述は親から変えず、「この run で何をしたか」は task が持つ、という
lesion 実験の約束 (`__main__.py`) に合わせるため。
"""
from __future__ import annotations

from scripts.lesion.protocol.cut import cut_label

# 可塑性モデルごとの「学習率」のパラメータ名。0 にすると重みが動かなくなるもの。
# custom_Akita は e-STDP が A_E、i-STDP が A_I を持つ (どちらを持つかは mode で決まる)。
LEARNING_RATES = {
    "custom_Akita": ("A_E", "A_I"),
}


def _field(protocol, key: str, default=None):
    """task の値を読む。ランチャは dict、run_one は検証済みの config.task を渡す。"""
    if isinstance(protocol, dict):
        return protocol.get(key, default)
    return getattr(protocol, key, default)


def frozen_groups(protocol) -> list[str]:
    """`task.freeze_plasticity` を名前のリストで返す。書いていなければ空。"""
    groups = _field(protocol, "freeze_plasticity") or []
    if isinstance(groups, str):
        raise ValueError(
            f"task.freeze_plasticity はリストで書いてください (例: [from_Inh])。got {groups!r}")
    groups = [str(g) for g in groups]
    if len(set(groups)) != len(groups):
        raise ValueError(f"task.freeze_plasticity に重複があります: {groups}")
    return groups


def run_label(protocol) -> str:
    """損傷 run のディレクトリ名。切断 spec の label に、止めた群を `+freeze-<群>` で足す。

    可塑性を止めない run の名前は `cut_label()` と同じ (既存の run と変わらない)。
    同じ切断で止める群だけが違う run が同じディレクトリを取り合わないようにするため。
    """
    label = cut_label(_field(protocol, "cut") or [], str(_field(protocol, "cut_combine", "or")))
    groups = frozen_groups(protocol)
    if groups:
        label += "+freeze-" + "-".join(sorted(groups))
    return label


def freeze_plasticity(config, groups: list[str]):
    """`groups` の STDP の学習率を 0 にした config の**複製**を返す。元の config は変えない。

    群の名前が無い、または学習率の分かっていない可塑性モデルなら止める
    (黙って何もしないと、止めたつもりの run が普通の run と同じになる)。
    """
    if not groups:
        return config
    frozen = config.model_copy(deep=True)
    for name in groups:
        if name not in frozen.synapses:
            raise ValueError(
                f"task.freeze_plasticity の {name!r} は config.synapses にありません "
                f"(あるもの: {sorted(frozen.synapses)})")
        plasticity = frozen.synapses[name].plasticity
        rates = LEARNING_RATES.get(plasticity.type)
        if rates is None:
            raise ValueError(
                f"{name}: 可塑性 {plasticity.type!r} の学習率が分かりません。"
                f" freeze.LEARNING_RATES に足してください。")
        extra = plasticity.model_extra or {}
        present = [rate for rate in rates if rate in extra]
        if not present:
            raise ValueError(f"{name}: 学習率 {rates} のどれも config にありません。")
        for rate in present:
            setattr(plasticity, rate, 0.0)
    return frozen
