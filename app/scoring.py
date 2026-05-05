import numpy as np
import pandas as pd
import gower
import math
import logging
from typing import Optional
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

OBJECTIVE_FILTERS = {
    "balanced": lambda df: df,
    "rebuild":  lambda df: df[df["age_norm"] <= 0.35],
    "youth":    lambda df: df[df["age_norm"] <= 0.25],
    "title":    lambda df: df[df["fee_norm"] >= 0.50],
}

FEATURES = ["nationality", "origin_league", "age_norm", "fee_norm", "fee_type"]
CAT_FEATURES = [True, True, False, False, True]  # mesmo índice que FEATURES

# Fallback: min/max observados no dataset de treino (FitScoreMl.ipynb, cell 14+20)
_AGE_TRAIN_MIN = 14.8
_AGE_TRAIN_MAX = 43.3
_FEE_LOG_TRAIN_MIN = 0.0
_FEE_LOG_TRAIN_MAX = 19.2182  # log1p(222_000_000) — max fee observado


def normalize_age(age: int, scaler: Optional[MinMaxScaler]) -> float:
    if scaler is not None:
        return float(scaler.transform([[age]])[0][0])
    return (age - _AGE_TRAIN_MIN) / (_AGE_TRAIN_MAX - _AGE_TRAIN_MIN)


def normalize_fee(market_value_eur: Optional[float], scaler: Optional[MinMaxScaler]) -> float:
    if market_value_eur is None or market_value_eur <= 0:
        return 0.0
    log_fee = math.log1p(market_value_eur)
    if scaler is not None:
        return float(scaler.transform([[log_fee]])[0][0])
    return (log_fee - _FEE_LOG_TRAIN_MIN) / (_FEE_LOG_TRAIN_MAX - _FEE_LOG_TRAIN_MIN)


def compute_fit_score(
    club_profiles: dict,
    position_index: dict,
    club_name: str,
    position_group: str,
    nationality: Optional[str],
    origin_league: Optional[str],
    age: int,
    market_value_eur: Optional[float],
    fee_type: str,
    objective: str,
    age_scaler: Optional[MinMaxScaler],
    fee_scaler: Optional[MinMaxScaler],
    sample_size: int = 2000,
    min_profile_size: int = 5,
) -> dict:
    key = (club_name, position_group)
    profile = club_profiles.get(key)

    if profile is None or len(profile) == 0:
        return {"fit_score": None, "confidence": "none", "profile_size": 0, "profile_found": False}

    profile = OBJECTIVE_FILTERS[objective](profile).copy()
    profile_size = len(profile)

    if profile_size < min_profile_size:
        return {"fit_score": None, "confidence": "low", "profile_size": profile_size, "profile_found": True}

    if profile_size > sample_size:
        profile = profile.sample(sample_size, random_state=42)

    age_norm = normalize_age(age, age_scaler)
    fee_norm = normalize_fee(market_value_eur, fee_scaler)
    age_norm = max(0.0, min(1.0, age_norm))
    fee_norm = max(0.0, min(1.0, fee_norm))

    candidate_row = {
        "nationality":    nationality or "Other",
        "origin_league":  origin_league or "unknown",
        "age_norm":       age_norm,
        "fee_norm":       fee_norm,
        "fee_type":       fee_type,
    }

    profile_features = profile[FEATURES].copy()
    profile_features = profile_features.fillna("Other")

    candidate_df = pd.DataFrame([candidate_row])
    combined = pd.concat([candidate_df, profile_features], ignore_index=True)

    try:
        dist_matrix = gower.gower_matrix(combined, cat_features=CAT_FEATURES)
    except Exception as e:
        logger.error(f"Erro ao calcular Gower distance: {e}")
        return {"fit_score": None, "confidence": "none", "profile_size": profile_size, "profile_found": True}

    distances = dist_matrix[0, 1:]
    fit_score = float(1.0 - float(np.mean(distances)))
    fit_score = round(max(0.0, min(1.0, fit_score)), 4)

    if profile_size >= 30:
        confidence = "high"
    elif profile_size >= 10:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "fit_score": fit_score,
        "confidence": confidence,
        "profile_size": profile_size,
        "profile_found": True,
    }
