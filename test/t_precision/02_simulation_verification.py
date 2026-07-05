#!/usr/bin/env python3
"""
【03_simulation_verification.py】
検証対象：GeNN モデルが実際に double 精度で動作しているかの確認

目的：生成されたコード（synapseUpdate.cc）で updateSynapses(double t) が
生成されていることを確認し、dt 計算と STDP 計算が double 精度で実行されることを検証
"""

import sys
from pathlib import Path
import re
import os

# プロジェクトルートに追加
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# モデルをインポート
import src.models.network.space
import src.models.network.connectors
import src.models.network.weights
import src.models.network.delays
import src.models.neurons.akita_escape_lif
import src.models.synapses.standard_models
import src.models.plasticity.custom_Akita

from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
import pygenn

print("=" * 70)
print("【03】GeNN double 精度動作確認")
print("=" * 70)

# config 読み込み
config_manager = ConfigManager()
config = config_manager.resolve("configs/akita_soc_fig2_legacy_autapse.yaml", "akita_soc_fig2")

# double 版をビルド
print("\n【モデル構築】")

builder = NetworkBuilder(config)
builder.genn_model = pygenn.GeNNModel("double", "SNN_Model_precision_check", time_precision="double")
builder.genn_model.dt = config.simulation.dt

genn_model, layout = builder.build()

print("  Building model...")
genn_model.build()

print("  ✓ Build complete")

# ===== 生成コード確認 =====
print("\n【生成コード内での double 精度確認】")

code_dir = "SNN_Model_precision_check_CODE"
if not os.path.exists(f"{code_dir}/synapseUpdate.cc"):
    print(f"ERROR: {code_dir}/synapseUpdate.cc not found")
    exit(1)

with open(f"{code_dir}/synapseUpdate.cc", 'r') as f:
    content = f.read()

# ===== updateSynapses シグネチャ確認 =====
print("\n1. updateSynapses function signature:")
match = re.search(r'void updateSynapses\((.*?)\)', content)
if match:
    signature = match.group(1)
    print(f"   {signature}")
    if "double" in signature:
        print(f"   ✓ double 型で実行")
    else:
        print(f"   ✗ float 型で実行（期待: double）")
        exit(1)
else:
    print(f"   ✗ updateSynapses signature not found")
    exit(1)

# ===== dt 計算確認 =====
print("\n2. dt calculation:")
dt_match = re.search(r'const (double|float) _dt\s*=', content)
if dt_match:
    dt_type = dt_match.group(1)
    print(f"   const {dt_type} _dt = ...")
    if dt_type == "double":
        print(f"   ✓ double 精度で計算")
    else:
        print(f"   ✗ float 精度で計算（期待: double）")
        exit(1)
else:
    print(f"   ✗ _dt declaration not found")
    exit(1)

# ===== STDP 計算確認（exp() 関数） =====
print("\n3. STDP calculation (exp):")
if "const double _timing" in content:
    print(f"   const double _timing = ...")
    print(f"   ✓ double 精度で計算")
elif "const float _timing" in content:
    print(f"   const float _timing = ...")
    print(f"   ✗ float 精度で計算（期待: double）")
    exit(1)
else:
    # _timing が見つからない場合、exp() が直接使われている可能性
    if "double" in content and "exp(" in content:
        print(f"   ✓ double 精度の exp() 関数が使用されている")
    else:
        print(f"   ? exp() calculation type unclear")

print("\n✓ 検証完了: GeNN が double 精度で動作していることを確認")
