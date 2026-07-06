# GPU vs CPU バックエンド性能比較（STDP 遅延モード別）

## 計測条件

- マシン: RTX 5070 Ti (Blackwell/sm_120), WSL2, CUDA 13.2
- ネットワーク: 100 ニューロン全結合（80 exci / 20 inhi, 9,900 シナプス）= 実 akita 構成
- 実行: 実運用スクリプト `scripts/akita_soc_fig2.py` の `run_steps` をそのまま使用（`scripts/bench_akita.py`）。計測バイアスを避けるため `timing_enabled` は OFF、素の wall-clock。
- CPU = single_threaded_cpu バックエンド、GPU = CUDA バックエンド。

> **補足（CUDA バックエンド復旧）**: CUDA バックエンドは当初 `genn_model.build()` で segfault していた。原因は apt パッケージ `libnvidia-compute-580-server`（Linux 用ドライバ）の混入で、その `libnvidia-ptxjitcompiler.so.580` がホスト WSL ドライバ(596.36)と不整合で JIT クラッシュ。実行前に以下を設定すれば非破壊で回避できる:
> ```bash
> export CUDA_PATH=/usr/local/cuda
> export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nv_dispsi.inf_amd64_0cc79ee359bdef40:/usr/local/cuda/lib64
> ```
> 恒久修正は `sudo apt remove libnvidia-compute-580-server`。

## 結果

### axonal モード（均一遅延・イベント駆動）

| | develop µs/step | record µs/step | 72h sim 外挿 |
|---|---:|---:|---:|
| CPU | 10.94 | 11.30 | ≈ 7.9 h |
| GPU | 22.85 | 22.49 | ≈ 16.5 h |

→ axonal では **CPU が約 2.1 倍速い**。

### dc モード（per-synapse 遅延・syn_dynamics）

| | develop µs/step | record µs/step | 72h sim 外挿 |
|---|---:|---:|---:|
| CPU | 48.63 | 52.89 | ≈ 35.0 h |
| GPU | 33.48 | 34.41 | ≈ 24.1 h |

→ dc では **GPU が約 1.45 倍速い**。

### まとめ

| モード | CPU µs/step | GPU µs/step | 速い方 | CPU 72h | GPU 72h |
|---|---:|---:|:--:|---:|---:|
| axonal（均一遅延・イベント駆動） | **10.94** | 22.85 | CPU (2.1×) | 7.9 h | 16.5 h |
| dc（per-synapse・syn_dynamics） | 48.63 | **33.48** | GPU (1.45×) | 35.0 h | 24.1 h |

## なぜ backend の優劣が反転するか

- **dc は毎ステップ全シナプスを走査する重い `syn_dynamics` が支配的。** CPU 直列だと 48µs だが、GPU では 9,900 シナプスを並列処理して 33µs に落ちる。まさに GPU が得意な仕事。
- **axonal は per-step のシナプス走査が無い**（イベント駆動）ので仕事量が小さく、GPU の per-step 起動レイテンシ floor（WSL2 で ~数十µs）が相対的に効いて負ける。

つまり **「重い per-step 仕事があるか」で backend の優劣が反転する。**

## 現状の推奨

- 均一遅延で足りる → **axonal + CPU**（全体最速 7.9 h）
- per-synapse 遅延が必須 → **dc + GPU**（24.1 h。dc + CPU の 35 h より 1.45× 速い）

## 参考: スケール依存（pure-step 計測, profile_syndyn.py）

| シナプス数 | CPU dc | GPU dc | CPU axonal | GPU axonal |
|---:|---:|---:|---:|---:|
| 9.9k (100n) | 45 | 79※ | 11 | 57※ |
| 160k (400n) | 569 | ~80 | 44 | ~58 |
| 640k (800n) | ~2,400 | 80 | 86 | 59 |
| 10.2M (3200n) | — | 358 | — | 58 |

※ profile_syndyn の GPU 値は `timing_enabled` の CUDA event オーバーヘッドで上振れ。実 wall は bench_akita 参照。CPU dc は O(N_syn) 線形、GPU は ~600k シナプスまで平坦。

---

## 実装後の実測（イベント駆動 pre_arrival_syn_code, 2026-07-07）

下の「改善予測」で想定した **per-synapse 遅延のネイティブ・イベント駆動化を GeNN fork に実装した**
（`pre_arrival_syn_code` / `arrival_delay_var`, 設計書 `/home/tanii/genn/docs/per_synapse_arrival_design.md`）。
毎ステップ全シナプス走査の `syn_dynamics` を廃し、各シナプスの到着時刻にのみ重み更新が発火する。
custom_Akita の delay_corrected / nearest_dc を event-driven 化し、実測した。

- 計測: `bench_akita.py` develop レート、config = `bench_dc_arrival.yaml`（`akita_soc.yaml` から
  `delay_by_target` を外し per-synapse 遅延=dc にした版, nearest モード）、CUDA 13.2、同一マシン。
- 実装方式は D-2（既存スパイクキュー再利用＋行を遅延ソート）。**予測が想定した GPU の atomic scatter は不要**
  だが、到着カーネルの並列度は「発火ソースニューロン数」（C-a 方式）で、シナプス数ではない。

### 実測 µs/step（旧 dc = syn_dynamics との比較）

| scale | 旧 dc CPU | **新 dc CPU** | 旧 dc GPU | **新 dc GPU** | axonal CPU | axonal GPU |
|---|---:|---:|---:|---:|---:|---:|
| 100n (9.9k syn) | 48.63 | **11.4** | 33.48 | **40.8** | 10.94 | 22.85 |
| 400n (160k syn) | (569※) | **44.4** | (~80※) | **46.5** | — | — |

※ 400n 旧 dc はスケール表(profile_syndyn)の pure-step 値で計測法が異なるため直接比較不可（参考）。

### 読み取り（予測との突き合わせ）

1. **CPU 100n: 48.6 → 11.4 µs/step（約 4.3× 高速化）。予測(~13µs/3.7×)より良い。**
   per-synapse 遅延コストが **axonal(10.94µs) とほぼ同一**に落ちた ＝ 予測どおり「per-synapse 遅延がほぼ無料」に。
2. **GPU 100n: 33.5 → 40.8 µs/step（約 1.22× の *悪化* ＝ 予測は外れ）。**
   予測は GPU も ~26µs へ改善としたが、実際は **悪化**。原因は方式の違い: 到着カーネル(C-a)は
   **発火ソースニューロン数(=100)ぶんのスレッド**しか使わず、旧 syn_dynamics の 9,900 スレッド
   （全シナプス並列）より GPU 占有率が大幅に低い。GPU は「重い総仕事量」より「高並列」を好むため、
   総仕事量が少ない到着カーネルでも小規模では占有率不足で負ける。
3. **GPU はスケールでほぼ平坦**: 100n 40.8µs → 400n 46.5µs（シナプス 16× でも 1.14×）。イベント駆動
   ＝コスト ∝ 到着イベント数で N_syn に比例しないことを実測で確認。CPU は 11.4 → 44.4µs（16× で ~4×,
   準線形。ネットワーク密化で発火・イベントが増える分）。
4. **backend の優劣が反転（＝予測どおり CPU 有利へ）**: 100n では **新 dc は CPU が GPU の ~3.6× 速い**
   (11.4 vs 40.8)。crossover は ~400n（CPU 44.4 ≈ GPU 46.5）。それ以上の大規模では GPU が平坦で有利。

### 推奨（実装後）

- **per-synapse 遅延 100n（akita 本番）→ 新 dc + CPU が最善**: 11.4µs → 72h ≈ **8.2h**。
  旧 best（dc+GPU 24.1h）から **約 2.9× 改善**、かつ axonal+CPU(7.9h)にほぼ並ぶ。
- **~400n 以上の大規模 → GPU**（到着カーネルが平坦、CPU は準線形で増える）。
- 均一遅延で足りるなら従来どおり axonal+CPU が最速（7.9h）。

**総括**: ネイティブ・イベント駆動化により、**CPU では per-synapse 遅延がほぼ無料**になり(akita 100n で
dc+GPU 24h→8.2h)、下の予測の主眼（CPU・大規模で大改善、backend 推奨が CPU へ反転）は実測で裏付けられた。
一方 **GPU 小規模は予測に反して悪化**した。これは C-a 到着カーネルの並列度がソースニューロン数どまりで、
小規模では GPU 占有率が不足するため。大規模化 or C-b（prefix-sum でシナプス粒度にフラット化）へ移行すれば
GPU 小規模の劣化も解消できる見込み。

---

## per-synapse 遅延を GeNN ネイティブサポートした場合の改善予測

> **注（2026-07-07 追記）**: 本節は実装前の予測。実装・実測の結果は上の
> 「実装後の実測」節を参照。要点: CPU の大幅改善（予測どおり, むしろ予測超え）と backend 推奨の
> CPU への反転は裏付けられたが、**GPU 小規模は予測(改善)に反して悪化**した（C-a 到着カーネルの
> 占有率不足）。

### 何を「ネイティブサポート」とするか

現状の dc は、per-synapse 遅延の到着時刻に重みを更新するために **毎ステップ全シナプスを走査する `syn_dynamics`** でポーリングしている（O(N_syn × N_steps)）。これを、**到着時刻にイベント駆動で重み更新が発火する仕組み**（シナプス index を到着ステップのバケツに入れる「スパース到着キュー」）を GeNN コアに実装した場合を想定する。

その場合、per-synapse 遅延の重み更新コストは **均一遅延の axonal と同じ「イベント駆動」クラス**に落ちる:

- コスト ∝ 発火数 × fanout（＝実到着イベント数）。N_steps にも N_syn にも比例しない。
- 毎ステップの全シナプス走査（syn_dynamics カーネル）が消える。

したがって **dc のコストは axonal のコストへ漸近する**（＋異種遅延キュー管理の小さなオーバーヘッド Δ）。この Δ は、発火時に各シナプスを到着バケツへ散らす scatter とバケツ管理の分。CPU では小さく、GPU では atomic 追記が要るためやや大きい。

### 各条件での改善予測

axonal 実測値を「イベント駆動の下限」とし、そこに管理オーバーヘッド Δ を上乗せして見積もる（**すべて推定値**）。

| 条件 | 現状 dc | ネイティブ後（予測） | 改善率（予測） | 根拠 |
|---|---:|---:|:--:|---|
| **CPU 100n (9.9k syn)** | 48.6 µs | **~13 µs** | **~3.7×** | axonal 10.9µs + 小さな Δ |
| **GPU 100n (9.9k syn)** | 33.5 µs | **~26 µs** | **~1.3×** | axonal 22.9µs + GPU scatter Δ |
| **CPU 大規模 (160k syn)** | 569 µs | **~50 µs** | **~11×** | axonal 44µs + Δ。O(N_syn) 除去が効く |
| **GPU 大規模 (10M syn)** | 358 µs | **~60 µs** | **~6×** | axonal 58µs 平坦へ漸近 |

### 読み取り

1. **改善は CPU と大規模ほど大きい。** dc の重さは O(N_syn) 直列走査に由来するので、それを除去する効果は CPU（並列化されていない）と、シナプス数が多いほど大きい。
2. **GPU 小規模の改善は小さい（~1.3×）。** GPU はもともと syn_dynamics を並列処理できており、per-step 起動 floor が支配的なので、走査を消しても floor は残る。
3. **最大の意義は「per-synapse 遅延がほぼ無料になる」こと。** ネイティブ後は dc が axonal 相当のコストになるため、100n では:
   - dc-native + CPU ≈ 13µs（72h ≈ 9.4h）— 現状の axonal+CPU（11µs, 7.9h）にほぼ並ぶ
   - → **per-synapse 遅延を、均一遅延とほぼ同コストで使えるようになる。**
4. **backend 推奨も反転する。** ネイティブ後の 100n では、イベント駆動で仕事が軽くなるため **再び CPU が有利**（GPU の起動 floor を避けられる）。dc-native は CPU ~13µs vs GPU ~26µs。

### まとめ（ネイティブ実装の投資対効果）

| 使い方 | 現状の最善 | ネイティブ後の最善（予測） |
|---|---|---|
| 均一遅延 | axonal+CPU 7.9h | 変化なし 7.9h |
| per-synapse・100n | dc+GPU 24.1h | **dc-native+CPU ~9.4h（約 2.6× 改善）** |
| per-synapse・大規模 | dc+GPU | dc-native（CPU/GPU とも大幅改善、~6–11×） |

**結論**: ネイティブ per-synapse 遅延（イベント駆動到着キュー）を実装すれば、per-synapse 遅延のコストを均一遅延並みに下げられる。特に **CPU・大規模で効果が大きく、100n でも dc+GPU 24h → ~9.4h（約 2.6 倍）** の改善が見込める。ただし実装は GeNN コアへのシナプス index 付き遅延キュー追加（GPU では scatter の atomic 処理）を要する大改造で、上記は測定済み axonal 値からの外挿推定である点に留意。
