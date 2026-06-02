#!/usr/bin/env python3
"""
【02_detailed_verification.py】
IEEE 754 レベルでの詳細な精度検証

【目的】
- float32 と float64 の内部ビット表現をビット差分で比較
- struct.pack/unpack でIEEE 754 ビット表現を直接操作
- nextafter() と np.spacing() で ULP を実測値で確認
- 複数の時刻パターンでの dt 計算精度を検証
- 理論値と実測値が完全に一致することを証明

【矛盾の解説】
初期の説明では「float32 の ULP = 128ms」と言っていたが、実際には 16ms。
これは binary exponent を正しく計算していなかったため。
正しい計算: ULP = 2^exponent / 2^mantissa_bits
  float32: 2^27 / 2^23 = 16 ms
  float64: 2^27 / 2^52 = 0.00000003 ms

【実行結果の確認項目】
✓ nextafter() での実測値が理論値と完全に一致
✓ np.spacing() での計算が両者と一致
✓ float32 では複雑な値でもビット差分が 1 bit（16ms）
✓ float64 では複雑な値でもビット差分が 1 bit（0.00000003ms）
✓ dt 計算での誤差が理論値と対応
"""

import numpy as np
import struct
import math

print("=" * 80)
print("IEEE 754 ビットレベルでの詳細な精度検証")
print("=" * 80)

# ========== ビット操作関数 ==========

def float32_to_bits(f):
    """float32 を 32-bit 符号付き整数に変換"""
    return struct.unpack('>i', struct.pack('>f', f))[0]

def float64_to_bits(f):
    """float64 を 64-bit 符号付き整数に変換"""
    return struct.unpack('>q', struct.pack('>d', f))[0]

# ========== 理論値と実測値の統一 ==========
print(f"\n{'='*80}")
print(f"72時間（259,200,000 ms）での ULP 理論値と実測値")
print(f"{'='*80}")

t_large_ms = 259_200_000

# 理論値の計算
exponent = math.floor(math.log2(t_large_ms))
ulp_f32_theory = 2**exponent / (2**23)
ulp_f64_theory = 2**exponent / (2**52)

print(f"\n理論値（IEEE 754 binary exponent から計算）:")
print(f"  指数: {exponent}")
print(f"  float32: ULP = 2^{exponent} / 2^23 = {ulp_f32_theory} ms")
print(f"  float64: ULP = 2^{exponent} / 2^52 = {ulp_f64_theory:.15e} ms")

# 実測値（nextafter）
t_f32 = np.float32(t_large_ms + 0.789456)
t_f32_next = np.nextafter(t_f32, np.float32(1e9))
ulp_f32_nextafter = float(t_f32_next - t_f32)

t_f64 = np.float64(t_large_ms + 0.789456)
t_f64_next = np.nextafter(t_f64, np.float64(1e9))
ulp_f64_nextafter = float(t_f64_next - t_f64)

# 実測値（np.spacing）
ulp_f32_spacing = np.spacing(np.float32(t_large_ms))
ulp_f64_spacing = np.spacing(np.float64(t_large_ms))

print(f"\n実測値 - nextafter() で測定:")
print(f"  float32: {ulp_f32_nextafter} ms")
print(f"  float64: {ulp_f64_nextafter:.15e} ms")

print(f"\n実測値 - np.spacing() で計算:")
print(f"  float32: {ulp_f32_spacing} ms")
print(f"  float64: {ulp_f64_spacing:.15e} ms")

print(f"\n検証:")
print(f"  float32 理論値 {ulp_f32_theory} vs 実測値 {ulp_f32_nextafter}: {'✓ 一致' if abs(ulp_f32_theory - ulp_f32_nextafter) < 1e-6 else '✗ 不一致'}")
print(f"  float64 理論値 {ulp_f64_theory:.15e} vs 実測値 {ulp_f64_nextafter:.15e}: {'✓ 一致' if abs(ulp_f64_theory - ulp_f64_nextafter) < 1e-20 else '✗ 不一致'}")

# ========== 複数時刻での dt 計算精度 ==========
print(f"\n{'='*80}")
print(f"複数の spike time パターンでの dt 計算精度")
print(f"{'='*80}")

test_cases = [
    ("1秒前", 258_999_999.123),
    ("1時間前", 258_996_000.456),
    ("12時間前", 258_756_000.789),
    ("48時間前", 257_472_000.321),
    ("71.67時間前", 258_200_000.654),
]

print(f"\n現在時刻: t = {t_large_ms}.789456 ms")
print(f"{'Spike time':>20} | {'float32 dt':>15} | {'float64 dt':>15} | {'error (f32)':>15}")
print("-" * 75)

errors_f32 = []
errors_f64 = []

for label, spike_ms in test_cases:
    # float32での計算
    t_f32 = np.float32(t_large_ms + 0.789456)
    spike_f32 = np.float32(spike_ms)
    dt_f32 = float(t_f32 - spike_f32)

    # float64での計算
    t_f64 = np.float64(t_large_ms + 0.789456)
    spike_f64 = np.float64(spike_ms)
    dt_f64 = float(t_f64 - spike_f64)

    # 理想値
    ideal_dt = (t_large_ms + 0.789456) - spike_ms

    # 誤差
    error_f32 = abs(dt_f32 - ideal_dt)
    error_f64 = abs(dt_f64 - ideal_dt)
    errors_f32.append(error_f32)
    errors_f64.append(error_f64)

    print(f"{label:>20} | {dt_f32:>15.0f} | {dt_f64:>15.6f} | {error_f32:>15.6f}")

print(f"\n最大誤差:")
print(f"  float32: {max(errors_f32):.6f} ms ⚠️")
print(f"  float64: {max(errors_f64):.15e} ms ✓")

# ========== ビット表現での比較 ==========
print(f"\n{'='*80}")
print(f"IEEE 754 ビット表現での精度比較")
print(f"{'='*80}")

# float32
val_f32 = np.float32(258_500_000.123456)
val_f32_bits = float32_to_bits(float(val_f32))
val_f32_next = np.nextafter(val_f32, np.float32(1e9))
val_f32_next_bits = float32_to_bits(float(val_f32_next))

print(f"\nfloat32(258,500,000.123456):")
print(f"  値: {val_f32}")
print(f"  ビット差分: {val_f32_next_bits - val_f32_bits} (=1 bit)")
print(f"  物理的な差: {float(val_f32_next - val_f32):.6f} ms")
print(f"  理論値との対応: 2^floor(log2(258500000))/2^23 = {2**math.floor(math.log2(258_500_000))/(2**23):.6f} ms")

# float64
val_f64 = np.float64(258_500_000.123456)
val_f64_bits = float64_to_bits(float(val_f64))
val_f64_next = np.nextafter(val_f64, np.float64(1e9))
val_f64_next_bits = float64_to_bits(float(val_f64_next))

print(f"\nfloat64(258,500,000.123456):")
print(f"  値: {val_f64:.10f}")
print(f"  ビット差分: {val_f64_next_bits - val_f64_bits} (=1 bit)")
print(f"  物理的な差: {float(val_f64_next - val_f64):.15e} ms")
print(f"  理論値との対応: 2^floor(log2(258500000))/2^52 = {2**math.floor(math.log2(258_500_000))/(2**52):.15e} ms")

# ========== STDP 精度比較 ==========
print(f"\n{'='*80}")
print(f"STDP計算での精度による影響")
print(f"{'='*80}")

A_E = 0.02
tau_E = 20.0

# 異なる時刻での dt 値を複雑に作成
dts_test = [
    ("10 ms", 10.123456),
    ("100 ms", 100.789012),
    ("1000 ms", 1000.456789),
    ("100,000 ms", 100_000.321654),
]

print(f"\ndW = A_E * exp(-dt/tau_E) の計算精度:")
print(f"{'dt (ms)':>15} | {'float32 dW':>18} | {'float64 dW':>18} | {'相対差':>15}")
print("-" * 80)

for label, dt_val in dts_test:
    dW_f32 = float(np.float32(A_E) * np.exp(-np.float32(dt_val) / np.float32(tau_E)))
    dW_f64 = float(np.float64(A_E) * np.exp(-np.float64(dt_val) / np.float64(tau_E)))
    rel_diff = abs(dW_f32 - dW_f64) / dW_f64 if dW_f64 != 0 else 0

    print(f"{label:>15} | {dW_f32:>18.15f} | {dW_f64:>18.15f} | {rel_diff:>15.10e}")

# ========== 結論 ==========
print(f"\n{'='*80}")
print(f"ビットレベルでの検証結論")
print(f"{'='*80}")

print(f"""
【理論値と実測値の完全一致】
  理論値（binary exponent）:
    float32: 2^27 / 2^23 = 16 ms
    float64: 2^27 / 2^52 = 0.00000003 ms

  実測値（nextafter/spacing）:
    float32: 16 ms ✓
    float64: 0.00000003 ms ✓

【矛盾の説明】
  初期説明で「float32 ULP = 128ms」としていたのは誤解。
  正しくは、IEEE 754 の binary exponent に基づいて計算すべき。
  exponent = 27 の場合:
    • float32: 1 bit = 2^27 / 2^23 = 16 ms
    • float64: 1 bit = 2^27 / 2^52 = 0.00000003 ms

【dt 計算精度への影響】
  float32:
    • 最大誤差: 1.67 ms
    • これが STDP タイミング計算に影響
    • 複数計算で累積し weight 更新が狂う

  float64:
    • 最大誤差: 0 ms（無視可能）
    • STDP タイミング計算が正確
    • weight が安定的に更新される

✓ ビット表現レベルでの検証により、
  float64 修正による精度改善が理論と実測で確認される
""")
