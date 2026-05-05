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


def recommend_candidates(
    club_profiles: dict,
    position_index: dict,
    club_name: str,
    position_group: str,
    objective: str,
    max_market_value_eur: Optional[float],
    fee_type: Optional[str],
    fee_scaler: Optional[MinMaxScaler],
    top_k: int = 20,
    sample_size: int = 2000,
    min_profile_size: int = 5,
) -> dict:
    key = (club_name, position_group)
    profile = club_profiles.get(key)

    if profile is None or len(profile) == 0:
        return {"error": "profile_not_found", "profile_size": 0, "candidates_evaluated": 0, "results": []}

    profile = OBJECTIVE_FILTERS[objective](profile).copy()
    profile_size = len(profile)

    if profile_size < min_profile_size:
        return {"error": "profile_too_small", "profile_size": profile_size, "candidates_evaluated": 0, "results": []}

    if profile_size > sample_size:
        profile = profile.sample(sample_size, random_state=42)

    candidates = position_index.get(position_group)
    if candidates is None or len(candidates) == 0:
        return {"error": "no_candidates", "profile_size": profile_size, "candidates_evaluated": 0, "results": []}

    candidates = candidates.copy()

    # Cap candidates before Gower to avoid OOM (matrix is O(n_cand × profile_size))
    MAX_CANDIDATES = 3000
    if len(candidates) > MAX_CANDIDATES:
        candidates = candidates.sample(MAX_CANDIDATES, random_state=42)

    if max_market_value_eur is not None:
        max_fee_norm = normalize_fee(max_market_value_eur, fee_scaler)
        max_fee_norm = max(0.0, min(1.0, max_fee_norm))
        candidates = candidates[candidates["fee_norm"] <= max_fee_norm]

    if fee_type is not None:
        candidates = candidates[candidates["fee_type"] == fee_type]

    if len(candidates) == 0:
        return {"error": None, "profile_size": profile_size, "candidates_evaluated": 0, "results": []}

    n_candidates = len(candidates)
    profile_features = profile[FEATURES].fillna("Other")
    candidate_features = candidates[FEATURES].fillna("Other")

    combined = pd.concat([candidate_features, profile_features], ignore_index=True)

    try:
        dist_matrix = gower.gower_matrix(combined, cat_features=CAT_FEATURES)
    except Exception as e:
        logger.error(f"Erro ao calcular Gower distance (recommend): {e}")
        return {"error": "gower_error", "profile_size": profile_size, "candidates_evaluated": n_candidates, "results": []}

    mean_distances = dist_matrix[:n_candidates, n_candidates:].mean(axis=1)
    fit_scores = np.clip(1.0 - mean_distances, 0.0, 1.0)

    candidates = candidates.reset_index(drop=True).copy()
    candidates["fit_score"] = fit_scores

    candidates = candidates.sort_values("fit_score", ascending=False)
    candidates = candidates.drop_duplicates(subset=["player_id"], keep="first")
    top = candidates.head(top_k)

    extra_cols = [c for c in candidates.columns if c not in FEATURES and c != "fit_score"]

    results = []
    for _, row in top.iterrows():
        results.append({
            "fit_score": round(float(row["fit_score"]), 4),
            "player_id": int(row["player_id"]) if "player_id" in row and pd.notna(row.get("player_id")) else None,
            "player_name": row.get("player_name") if pd.notna(row.get("player_name")) else None,
            "nationality": row.get("nationality"),
            "origin_league": row.get("origin_league"),
            "fee_type": row.get("fee_type"),
        })

    return {
        "error": None,
        "profile_size": profile_size,
        "candidates_evaluated": n_candidates,
        "results": results,
    }
