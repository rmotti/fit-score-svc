"""
Gera age_scaler.pkl e fee_scaler.pkl a partir dos min/max observados no notebook.

Seção 10 do FitScoreMl.ipynb usou um único MinMaxScaler sobre
[age_at_transfer, log_fee_proxy]. Este script reconstrói dois scalers
separados com os mesmos limites, conforme esperado por scoring.py.

Valores extraídos dos outputs do notebook:
  age_at_transfer: min=14.8, max=43.3  (cell 14)
  transfer_fee max: 222_000_000 EUR     (cell 14)
  fee_proxy = transfer_fee.fillna(market_value_in_eur), log1p aplicado
"""

import pickle
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"

# --- age_scaler ---
age_min = 14.8
age_max = 43.3

age_scaler = MinMaxScaler()
age_scaler.fit([[age_min], [age_max]])

with open(ARTIFACTS / "age_scaler.pkl", "wb") as f:
    pickle.dump(age_scaler, f)

# --- fee_scaler ---
# max fee observado no dataset: 222_000_000 EUR → log1p
fee_log_min = 0.0
fee_log_max = math.log1p(222_000_000)

fee_scaler = MinMaxScaler()
fee_scaler.fit([[fee_log_min], [fee_log_max]])

with open(ARTIFACTS / "fee_scaler.pkl", "wb") as f:
    pickle.dump(fee_scaler, f)

# Verificação rápida
print(f"age_scaler  — min: {age_min}, max: {age_max}")
print(f"  normalize(14.8) = {age_scaler.transform([[14.8]])[0][0]:.4f}  (esperado: 0.0)")
print(f"  normalize(43.3) = {age_scaler.transform([[43.3]])[0][0]:.4f}  (esperado: 1.0)")
print(f"  normalize(25.0) = {age_scaler.transform([[25.0]])[0][0]:.4f}")

print(f"\nfee_scaler  — log_min: {fee_log_min:.4f}, log_max: {fee_log_max:.4f}")
print(f"  normalize(log1p(0))       = {fee_scaler.transform([[0.0]])[0][0]:.4f}  (esperado: 0.0)")
print(f"  normalize(log1p(222M))    = {fee_scaler.transform([[fee_log_max]])[0][0]:.4f}  (esperado: 1.0)")
print(f"  normalize(log1p(10M))     = {fee_scaler.transform([[math.log1p(10_000_000)]])[0][0]:.4f}")
print(f"  normalize(log1p(50M))     = {fee_scaler.transform([[math.log1p(50_000_000)]])[0][0]:.4f}")

print("\nSalvos em artifacts/:")
print("  age_scaler.pkl")
print("  fee_scaler.pkl")
