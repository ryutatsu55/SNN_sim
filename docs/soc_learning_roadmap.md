# 自己組織化臨界性(SOC) 学習ロードマップ

自己組織化臨界性(SOC)を「**定量的にチューニングし、亜臨界→超臨界→臨界の発達を理解する**」ための体系的な学習ロードマップ。
本プロジェクト（Akita SoC-SNN 再現）で扱う要素——**b（escape noise の gain）・STP・I-STDP・ΔCr・双安定性**——に紐づけて整理する。

> 本プロジェクトの調査から得た示唆: σ=1（臨界）の安定吸引を左右する主因は **escape noise の gain (b)**（b=8mV で初めてσ≈1を安定保持）。この理解を数式で説明・設計できるようになることが目標。

---

## Tier 0: まず全体像を掴む（レビューで地図を得る）

- **Theoretical foundations of studying criticality in the brain** (Tian et al., 2022, Network Neuroscience)
  最初に読むべき「地図」。臨界を **OC（普通の臨界）/ qC（準臨界）/ SOC（自己組織化臨界）/ SOqC（自己組織化準臨界）** の4類型に神経科学の言葉で整理し、測定の落とし穴も解説。「σ=1に安定吸引できるか」問題はまさにこの分類。
  <https://consensus.app/papers/details/ff69aa6ace0558308c16a405edf8060f/>
- **Self-Organization Toward Criticality by Synaptic Plasticity** (Zeraati et al., 2020)
  可塑性でSOCする仕組みのレビュー。各種可塑性則を「無効化後どれだけ臨界が持続するか」で分類。本研究に最も直結。
  <https://consensus.app/papers/details/b4ed1a3c69e058dfa7ba2b8323462f4f/>
- **Self-Organized Criticality in the Brain** (Plenz et al., 2021) — 実験的証拠・E/I制御・connectivityの俯瞰。
  <https://consensus.app/papers/details/9f71587e82f95f768b067812369f30b9/>
- **Criticality, Connectivity, and Neural Disorder** (Heiney et al., 2021, Front. Comput. Neurosci.) — 物理/実験/計算横断のレビュー。
  <https://consensus.app/papers/details/0c9840f191f55bfda2f4a115fab286b3/>

## Tier 1: 臨界・相転移・べき乗則の物理（教科書）

言葉（σ, 臨界指数τ, スケーリング, branching process）を身につける。
- **Christensen & Moloney, "Complexity and Criticality"** (Imperial College Press, 2005) — SOCの標準教科書（sandpile → branching process → power law）。
- **Sethna, "Statistical Mechanics: Entropy, Order Parameters, and Complexity"** (無料: sethna.lassp.cornell.edu) — 相転移・臨界指数・スケーリングの物理。
- **Pruessner, "Self-Organised Criticality"** (2012) — SOCを厳密にやる時の網羅的リファレンス。
- 分岐過程（σ=1が臨界）: **Harris, "The Theory of Branching Processes"** — branching比σの数学的土台。

## Tier 2: ニューロン・シナプス・可塑性の基礎（教科書＝本モデルの土台）

- **Gerstner, Kistler, Naud, Paninski, "Neuronal Dynamics"** (Cambridge 2014, 無料: neuronaldynamics.epfl.ch) — Akita論文の引用3。
  LIF・**escape noise（発火強度 ρ(u)=(1/τ)exp((u−θ)/Δu)。この Δu が本研究の b）**・STDP・STP・ネットワーク動態を数式で。
  **「なぜ b が gain で、σ=1 安定性を左右するか」はここで完全に理解できる**（Ch.7 escape noise, Ch.9 firing rate, Ch.19 plasticity）。

## Tier 3: ニューロナルアバランシェと定量ツール（計測系）

- **Beggs & Plenz 2003 (J. Neurosci) "Neuronal avalanches in neocortical circuits"** — アバランシェの定義・**σ=1・τ=−3/2**。foundational（Akita引用12）。
- **Clauset, Shalizi, Newman 2009 (SIAM Review) "Power-law distributions in empirical data"** — MLE・LLR・KS検定。**本プロジェクトのLLR実装の元**（Akita引用14）。
- **`powerlaw` Python package** (Alstott, Bullmore, Plenz 2014, PLoS ONE) — アバランシェfittingの標準。
- **Wilting & Priesemann 2018 (Nat. Commun.) MR estimator** — subsampling下でのσ推定。**100ニューロンの部分観測でσを正しく測る**ために重要。

## Tier 4: SOCの「自己調整機構」（＝各パラメータの意味）

- **Dynamical synapses causing self-organized criticality in neural networks** (Levina et al., 2007, Nature Physics)
  **STP（短期抑圧）がSOCを生む**。本モデルの U, τrec の役割の原典。
  <https://consensus.app/papers/details/910fd864f2cb5a138cfcaa60769ae904/>
- **Phase transitions and self-organized criticality in networks of stochastic spiking neurons** (Brochini et al., 2016, Scientific Reports)
  **確率発火の firing function Φ(V) と「dynamic gain」がSOCを制御**。**b(gain) の発見に最も直結**：gainの形が臨界を決めるという解析。必読。
  <https://consensus.app/papers/details/82debede87085b058f78583d8f71069d/>
- **Self-organized criticality model for brain plasticity** (de Arcangelis et al., 2006, PRL)
  <https://consensus.app/papers/details/0cdad21dad7953c5ad10a7f0bef44f4e/>
- **Homeostatic Structural Plasticity Can Build Critical Networks** (van Ooyen et al., 2019)
  <https://consensus.app/papers/details/a9f93cb1ecf15f809a213bdf5b1464c6/>
- **Self-organized criticality in neural networks from activity-based rewiring** (Landmann et al., 2020, PRE)
  <https://consensus.app/papers/details/d77a0c0fd34253458bc6d28b32fdb633/>

## Tier 5: E-I balance と抑制可塑性（＝本モデルの I-STDP）

- **Vogels, Sprekeler, Zenke, Clopath, Gerstner 2011 (Science) "Inhibitory plasticity balances excitation and inhibition"**
  **I-STDPの原典**。抑制可塑性が目標発火率へ homeostatic に引き込む機構。本研究の「過抑制/暴走」「I-STDPのセットポイント」はこの理論で理解できる。

## Tier 6: 発達フロー 亜臨界→超臨界→臨界（本プロジェクトの核心関心）

- **Self-Organized Criticality in Developing Neuronal Networks** (Tetzlaff et al., 2010, PLoS Comput. Biol.)
  **発達で 低活動→超臨界→亜臨界→臨界 の4相**。**本プロジェクトで使うΔCr指標の原典**。「抑制の遅延成熟」を予測＝Akitaの筋そのもの。**Akita理解の直接の親、最重要**。
  <https://consensus.app/papers/details/37474c0819ba5c85a2462b466d82aadc/>
- **Growing Critical: Self-Organized Criticality in a Developing Neural System** (Kossio et al., 2018, PRL)
  成長して臨界へ、Hawkes過程で解析（発達を数式で追える）。
  <https://consensus.app/papers/details/67462117288e59448450e27f8abb1371/>
- **Structural Modularity Tunes Mesoscale Criticality in Biological Neuronal Networks** (Okujeni et al., 2023, J. Neurosci.)
  **実験で supercritical→subcritical→critical をmodular構造が調整**。局所は超臨界でも全体で臨界という視点。
  <https://consensus.app/papers/details/19c40b6be04f5051bb8b133d7e5c3802/>
- **Neurobiologically Realistic Determinants of Self-Organized Criticality in Networks of Spiking Neurons** (Rubinov et al., 2011, PLoS Comput. Biol.)
  STDP+軸索遅延で **subcritical→supercritical遷移**、低外部駆動・強内部相互作用が臨界に有利。**パラメータ感度分析の教科書的モデル**。
  <https://consensus.app/papers/details/d25bb00e4eff530f95e3bfd2f384fa3d/>
- **Noise and STDP drive SOC** (Ikeda, Akita, Takahashi, 2023, Appl. Phys. Lett.) — 本プロジェクトの対象論文。
  <https://consensus.app/papers/details/140c2a586e145cf7bd6efc79f8e8ad07/>

## Tier 7: 「双安定性」を深く（本研究で観測した現象）

- **Self-organized bistability and its possible relevance for brain dynamics** (Buendía et al., 2019, Phys. Rev. Research)
  **SOC vs SOB（自己組織化双安定性）**。観測した「離散バースト⇄連続融合のナイフエッジ双安定」はこの枠組み。「タイムスケール分離が不十分だとSOBに寄る」等、不安定性の物理的言語。
  <https://consensus.app/papers/details/6146f9982a0852acb1c3943445919164/>
- **Criticality meets learning (SORN)** (Del Papa et al., 2017, PLoS ONE) — 学習と臨界の関係、membrane noiseと臨界のトレードオフ（bとタスク性能の話）。
  <https://consensus.app/papers/details/408870084df75558a8bd5c9ca6f3ba64/>

---

## 推奨する読む順番（体系的パス）

1. **Tier 2 の Gerstner "Neuronal Dynamics"**（escape noise=b, STDP, STP章）で土台
2. **Tier 1** で臨界の物理語彙
3. **Beggs & Plenz + Clauset**（Tier 3）で計測
4. **Levina + Brochini**（gain/STP機構＝b発見の理論）
5. **Vogels-Sprekeler**（I-STDP）
6. **Tetzlaff + Rubinov + Akita**（発達フロー）
7. **Tian / Zeraati / Buendía** で統合（SOC/SOqC/SOB、双安定性）

**一番効く4本柱**: Gerstner（b理論）→ Brochini（gainとSOC）→ Tetzlaff（発達ΔCr）→ Vogels-Sprekeler（I-STDP）。

## 定量チューニングへの直結

- **σ（branching比）を正しく測る**: Wilting & Priesemann の MR estimator（部分観測補正）を導入すると、ΔCr/LLRより直接的にσをトラッキングでき、パラメータ掃引の目的関数にできる。
- **制御パラメータの理論的意味**: gain(b)→Brochini, STP(U,τrec)→Levina, E-I/I-STDP→Vogels-Sprekeler, 外部駆動/結合→Rubinov。「bがσ=1安定性を決める」はBrochiniの firing-function-gain 論と整合。
- **発達フローの目的関数化**: Tetzlaff のΔCr時系列（超臨界ピーク→臨界へ降下）を再現ターゲットにできる。

## 関連（本プロジェクト内ドキュメント）

- 再現の技術メモ: [`akita_soc_reproduction_memo.md`](akita_soc_reproduction_memo.md)
