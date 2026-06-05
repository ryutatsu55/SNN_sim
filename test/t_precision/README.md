# Precision Fix Verification

## 修正内容

72時間シミュレーション時の重み飽和問題を解決するため、NetworkBuilder.py で以下を修正：

```python
# 修正前: float32 では精度不足
self.genn_model = pygenn.GeNNModel("float", "SNN_Model")

# 修正後: float64 で精度確保
self.genn_model = pygenn.GeNNModel("double", "SNN_Model", time_precision="double")
```

## テスト実行

| テスト | 検証対象 | 実行方法 |
|--------|---------|---------|
| **01_detailed_verification.py** | IEEE 754 ビット表現での ULP 測定 | `python test/t_precision/01_detailed_verification.py` |
| **02_simulation_verification.py** | GeNN が double 精度で動作しているか確認 | `python test/t_precision/02_simulation_verification.py` |

---

**検証完了日**: 2026-06-02  
**ステータス**: ✅ 完了
