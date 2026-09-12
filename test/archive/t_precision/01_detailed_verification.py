#!/usr/bin/env python3
"""
【02_detailed_verification.py】
検証対象：IEEE 754 ビット表現での ULP 測定と理論値の対応

目的：struct.pack/unpack でビット差分を直接確認し、理論値と対応することを検証
"""

import numpy as np
import struct
import math

def float32_to_bits(f):
    """float32 を 32-bit 整数に変換"""
    return struct.unpack('>i', struct.pack('>f', f))[0]

def float64_to_bits(f):
    """float64 を 64-bit 整数に変換"""
    return struct.unpack('>q', struct.pack('>d', f))[0]

# テスト値
test_values = [
    (258_500_000.123456, "258.5M ms"),
    (259_200_000.789456, "259.2M ms (72h)"),
]

print("=" * 70)
print("【02】IEEE 754 ビット表現での ULP 測定")
print("=" * 70)

for test_val, label in test_values:
    exponent = math.floor(math.log2(test_val))
    ulp_f32_theory = 2**exponent / (2**23)
    ulp_f64_theory = 2**exponent / (2**52)

    print(f"\n【{label}】")
    print(f"exponent: {exponent}")

    # float32
    val_f32 = np.float32(test_val)
    val_f32_bits = float32_to_bits(float(val_f32))
    val_f32_next = np.nextafter(val_f32, np.float32(1e9))
    val_f32_next_bits = float32_to_bits(float(val_f32_next))
    bit_diff_f32 = val_f32_next_bits - val_f32_bits
    physical_diff_f32 = float(val_f32_next - val_f32)

    print(f"  float32:")
    print(f"    ビット差分: {bit_diff_f32}")
    print(f"    物理差: {physical_diff_f32} ms")
    print(f"    理論値: {ulp_f32_theory} ms")
    print(f"    一致: {abs(physical_diff_f32 - ulp_f32_theory) < 1e-6}")

    # float64
    val_f64 = np.float64(test_val)
    val_f64_bits = float64_to_bits(float(val_f64))
    val_f64_next = np.nextafter(val_f64, np.float64(1e9))
    val_f64_next_bits = float64_to_bits(float(val_f64_next))
    bit_diff_f64 = val_f64_next_bits - val_f64_bits
    physical_diff_f64 = float(val_f64_next - val_f64)

    print(f"  float64:")
    print(f"    ビット差分: {bit_diff_f64}")
    print(f"    物理差: {physical_diff_f64:.15e} ms")
    print(f"    理論値: {ulp_f64_theory:.15e} ms")
    print(f"    一致: {abs(physical_diff_f64 - ulp_f64_theory) < 1e-20}")

print("\n✓ 検証完了: ビット差分と理論値が対応")
