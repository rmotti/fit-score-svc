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

FEATURES = ["nationality", "origin_league", "age_norm"]
CAT_FEATURES = [True, True, False]  # mesmo índice que FEATURES

# O fit responde "é o TIPO de jogador que o clube contrata" — e usa só dado confiável.
# Custo (fee_norm/fee_type) foi REMOVIDO das features: ~69% das transferências do dataset
# vêm como "free"/0 (fee ausente/não-divulgado registrado como grátis — Chelsea, Man City
# etc. aparecem 60-70% "de graça"), então era ~70% ruído. Além disso custo/orçamento já é
# tratado fora do fit, no componente marketValue do scout score (relativo ao budget, com
# dado confiável) — manter aqui era redundante. Pesos iguais entre as 3 dimensões restantes.
# IMPORTANTE: usado em TODA gower_matrix (baseline E scoring) — a régua precisa ser a mesma.
FEATURE_WEIGHTS = np.array([1.0, 1.0, 1.0])

# Cap padrão de amostragem do perfil. Usado no scoring E na pré-computação dos
# baselines (profiles.load_artifacts) — precisam usar o MESMO subset pra calibração
# ser consistente. Perfis reais hoje têm no máx ~257 membros, então o cap nunca
# dispara na prática; mantê-lo idêntico nos dois caminhos é uma garantia, não um custo.
DEFAULT_SAMPLE_SIZE = 2000

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


def _prepare_profile(profile: pd.DataFrame, objective: str, sample_size: int) -> pd.DataFrame:
    """Filtra o perfil pelo objetivo e aplica o cap de amostragem.

    Fonte única de verdade pro subset do perfil — usada tanto no scoring quanto na
    pré-computação do baseline, pra que candidato e baseline sejam medidos contra
    exatamente o mesmo conjunto de contratações.
    """
    profile = OBJECTIVE_FILTERS[objective](profile).copy()
    if len(profile) > sample_size:
        profile = profile.sample(sample_size, random_state=42)
    return profile


def compute_baseline(profile: pd.DataFrame) -> Optional[np.ndarray]:
    """Distâncias Gower leave-one-out de cada membro do perfil contra os demais.

    Representa "como uma contratação típica desse clube encaixa nesse clube" — a
    régua contra a qual o candidato é calibrado. Retorna o vetor ORDENADO (asc), ou
    None se o perfil tiver < 2 membros (sem par possível).
    """
    n = len(profile)
    if n < 2:
        return None
    feats = profile[FEATURES].fillna("Other")
    try:
        m = gower.gower_matrix(feats, weight=FEATURE_WEIGHTS, cat_features=CAT_FEATURES)
    except Exception as e:
        logger.error(f"Erro ao computar baseline Gower: {e}")
        return None
    np.fill_diagonal(m, np.nan)            # exclui a distância do membro a si mesmo
    loo = np.nanmean(m, axis=1)
    return np.sort(loo)


def calibrate(cand_dist, baseline: np.ndarray):
    """Mapeia a distância média do candidato pra 0–100 contra o baseline do perfil.

    Score = % de contratações reais do clube que encaixam PIOR que o candidato.
    Menor distância → score maior. Interpolação linear na CDF empírica do baseline
    evita saturação no topo. Aceita escalar ou array (np.interp vetoriza e faz clamp
    nos extremos, então o resultado já fica em [0, 100]).
    """
    n = len(baseline)
    cdf = np.interp(cand_dist, baseline, np.linspace(0.0, 1.0, n))
    return (1.0 - cdf) * 100.0


def build_all_baselines(club_profiles: dict, sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict:
    """Pré-computa o baseline de cada (club, position_group, objective).

    Chamado uma vez no boot do serviço. Custo medido: ~3s / ~0.3 MB pros 4.010 perfis
    × 4 objetivos. Chave do dict resultante: (club_name, position_group, objective).
    """
    baselines = {}
    for (club, pos), profile in club_profiles.items():
        for objective in OBJECTIVE_FILTERS:
            subset = _prepare_profile(profile, objective, sample_size)
            base = compute_baseline(subset)
            if base is not None:
                baselines[(club, pos, objective)] = base
    return baselines


def compute_fit_score(
    club_profiles: dict,
    position_index: dict,
    club_baselines: dict,
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
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_profile_size: int = 5,
) -> dict:
    key = (club_name, position_group)
    profile = club_profiles.get(key)

    if profile is None or len(profile) == 0:
        return {"fit_score": None, "confidence": "none", "profile_size": 0, "profile_found": False}

    profile = _prepare_profile(profile, objective, sample_size)
    profile_size = len(profile)

    if profile_size < min_profile_size:
        return {"fit_score": None, "confidence": "low", "profile_size": profile_size, "profile_found": True}

    baseline = club_baselines.get((club_name, position_group, objective))
    if baseline is None:
        # Não deveria ocorrer (profile_size >= 5 garante baseline), mas é o degrau
        # seguro: sem régua de calibração não há score.
        return {"fit_score": None, "confidence": "low", "profile_size": profile_size, "profile_found": True}

    age_norm = max(0.0, min(1.0, normalize_age(age, age_scaler)))

    candidate_row = {
        "nationality":    nationality or "Other",
        "origin_league":  origin_league or "unknown",
        "age_norm":       age_norm,
    }

    profile_features = profile[FEATURES].copy()
    profile_features = profile_features.fillna("Other")

    candidate_df = pd.DataFrame([candidate_row])
    combined = pd.concat([candidate_df, profile_features], ignore_index=True)

    try:
        dist_matrix = gower.gower_matrix(combined, weight=FEATURE_WEIGHTS, cat_features=CAT_FEATURES)
    except Exception as e:
        logger.error(f"Erro ao calcular Gower distance: {e}")
        return {"fit_score": None, "confidence": "none", "profile_size": profile_size, "profile_found": True}

    distances = dist_matrix[0, 1:]
    fit_score = round(float(calibrate(float(np.mean(distances)), baseline)), 1)

    return {
        "fit_score": fit_score,
        "confidence": _confidence(profile_size),
        "profile_size": profile_size,
        "profile_found": True,
    }


def _confidence(profile_size: int) -> str:
    if profile_size >= 30:
        return "high"
    if profile_size >= 10:
        return "medium"
    return "low"


# Conceitos do breakdown do fit — um por feature da Gower. Cada item lista as features
# que o compõem; o peso do conceito é a soma dos pesos das suas features.
_FIT_CONCEPTS = [
    ("nationality", ["nationality"]),
    ("origin_league", ["origin_league"]),
    ("age", ["age_norm"]),
]


def _feature_cand_distances(profile: pd.DataFrame, feature: str, cand_value) -> np.ndarray:
    """Distância da feature do candidato a cada membro do perfil (mesma lógica da Gower:
    categórica = 0/1; numérica = |a-b| nos valores já normalizados)."""
    idx = FEATURES.index(feature)
    col = profile[feature].fillna("Other")
    if CAT_FEATURES[idx]:
        return (col.values != cand_value).astype(float)
    return np.abs(col.values.astype(float) - float(cand_value))


def _feature_loo(profile: pd.DataFrame, feature: str) -> np.ndarray:
    """Distância leave-one-out de cada membro aos OUTROS naquela feature (a régua de
    tipicidade da dimensão). Categórica: fração de outros com valor diferente."""
    idx = FEATURES.index(feature)
    col = profile[feature].fillna("Other")
    n = len(col)
    if CAT_FEATURES[idx]:
        same_incl = col.map(col.value_counts()).values.astype(float)  # inclui o próprio
        return (n - same_incl) / (n - 1)
    x = col.values.astype(float)
    return np.array([np.abs(x - x[i]).sum() / (n - 1) for i in range(n)])


def _concept_context(profile: pd.DataFrame, key: str, age_scaler) -> str:
    """Short summary of what the club typically signs on that dimension."""
    if key in ("nationality", "origin_league"):
        feature = "nationality" if key == "nationality" else "origin_league"
        vc = profile[feature].fillna("Other").value_counts(normalize=True).head(2)
        return " · ".join(f"{v} {p:.0%}" for v, p in vc.items())
    # age
    med_norm = float(profile["age_norm"].median())
    if age_scaler is not None:
        med = float(age_scaler.inverse_transform([[med_norm]])[0][0])
    else:
        med = med_norm * (_AGE_TRAIN_MAX - _AGE_TRAIN_MIN) + _AGE_TRAIN_MIN
    return f"club usually ~{med:.0f} y/o"


def _candidate_value_str(key: str, nationality, origin_league, age) -> str:
    if key == "nationality":
        return nationality or "Other"
    if key == "origin_league":
        return origin_league or "unknown"
    return f"{age} y/o"  # age


def explain_fit_score(
    club_profiles: dict,
    club_baselines: dict,
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
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_profile_size: int = 5,
) -> dict:
    """Fit geral + breakdown calibrado por conceito (atribuição, não decomposição exata:
    o geral é o percentil da distância agregada; cada linha é o percentil da dimensão)."""
    empty = {"fit_score": None, "confidence": "none", "profile_size": 0,
             "profile_found": False, "breakdown": []}

    profile = club_profiles.get((club_name, position_group))
    if profile is None or len(profile) == 0:
        return empty

    profile = _prepare_profile(profile, objective, sample_size).reset_index(drop=True)
    profile_size = len(profile)
    baseline = club_baselines.get((club_name, position_group, objective))
    if profile_size < min_profile_size or baseline is None:
        return {"fit_score": None, "confidence": "low", "profile_size": profile_size,
                "profile_found": True, "breakdown": []}

    age_norm = max(0.0, min(1.0, normalize_age(age, age_scaler)))
    cand_vals = {
        "nationality": nationality or "Other",
        "origin_league": origin_league or "unknown",
        "age_norm": age_norm,
    }

    # fit geral — reusa compute_fit_score pra garantir o MESMO número do /score
    overall = compute_fit_score(
        club_profiles, {}, club_baselines, club_name, position_group,
        nationality, origin_league, age, market_value_eur, fee_type, objective,
        age_scaler, fee_scaler, sample_size, min_profile_size,
    )

    total_w = float(FEATURE_WEIGHTS.sum())
    breakdown = []
    for key, feats in _FIT_CONCEPTS:
        cand_per_member = np.mean([_feature_cand_distances(profile, f, cand_vals[f]) for f in feats], axis=0)
        loo = np.mean([_feature_loo(profile, f) for f in feats], axis=0)
        score = round(float(calibrate(cand_per_member.mean(), np.sort(loo))), 1)
        weight = sum(FEATURE_WEIGHTS[FEATURES.index(f)] for f in feats) / total_w
        breakdown.append({
            "key": key,
            "weight": round(weight, 3),
            "score": score,
            "candidate_value": _candidate_value_str(key, nationality, origin_league, age),
            "club_context": _concept_context(profile, key, age_scaler),
        })

    return {
        "fit_score": overall["fit_score"],
        "confidence": _confidence(profile_size),
        "profile_size": profile_size,
        "profile_found": True,
        "breakdown": breakdown,
    }


def recommend_candidates(
    club_profiles: dict,
    position_index: dict,
    club_baselines: dict,
    club_name: str,
    position_group: str,
    objective: str,
    max_market_value_eur: Optional[float],
    fee_type: Optional[str],
    fee_scaler: Optional[MinMaxScaler],
    top_k: int = 20,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_profile_size: int = 5,
) -> dict:
    key = (club_name, position_group)
    profile = club_profiles.get(key)

    if profile is None or len(profile) == 0:
        return {"error": "profile_not_found", "profile_size": 0, "candidates_evaluated": 0, "results": []}

    profile = _prepare_profile(profile, objective, sample_size)
    profile_size = len(profile)

    if profile_size < min_profile_size:
        return {"error": "profile_too_small", "profile_size": profile_size, "candidates_evaluated": 0, "results": []}

    baseline = club_baselines.get((club_name, position_group, objective))
    if baseline is None:
        return {"error": "profile_too_small", "profile_size": profile_size, "candidates_evaluated": 0, "results": []}

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
        dist_matrix = gower.gower_matrix(combined, weight=FEATURE_WEIGHTS, cat_features=CAT_FEATURES)
    except Exception as e:
        logger.error(f"Erro ao calcular Gower distance (recommend): {e}")
        return {"error": "gower_error", "profile_size": profile_size, "candidates_evaluated": n_candidates, "results": []}

    mean_distances = dist_matrix[:n_candidates, n_candidates:].mean(axis=1)
    fit_scores = calibrate(mean_distances, baseline)  # 0–100, calibrado vs baseline

    candidates = candidates.reset_index(drop=True).copy()
    candidates["fit_score"] = fit_scores

    candidates = candidates.sort_values("fit_score", ascending=False)
    candidates = candidates.drop_duplicates(subset=["player_id"], keep="first")
    top = candidates.head(top_k)

    extra_cols = [c for c in candidates.columns if c not in FEATURES and c != "fit_score"]

    results = []
    for _, row in top.iterrows():
        results.append({
            "fit_score": round(float(row["fit_score"]), 1),
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


def get_club_archetype(
    club_profiles: dict,
    club_name: str,
    position_group: str,
    objective: str,
    age_scaler: Optional[MinMaxScaler],
    fee_scaler: Optional[MinMaxScaler],
    top_k_categories: int = 5,
    min_profile_size: int = 5,
) -> dict:
    key = (club_name, position_group)
    profile = club_profiles.get(key)

    if profile is None or len(profile) == 0:
        return {"error": "profile_not_found", "profile_size": 0}

    profile = OBJECTIVE_FILTERS[objective](profile).copy()
    profile_size = len(profile)

    if profile_size < min_profile_size:
        return {"error": "profile_too_small", "profile_size": profile_size}

    ages_norm = profile["age_norm"].values.reshape(-1, 1)
    if age_scaler is not None:
        ages = age_scaler.inverse_transform(ages_norm).flatten()
    else:
        ages = (ages_norm.flatten() * (_AGE_TRAIN_MAX - _AGE_TRAIN_MIN)) + _AGE_TRAIN_MIN

    fees_norm = profile["fee_norm"].values.reshape(-1, 1)
    if fee_scaler is not None:
        fees_log = fee_scaler.inverse_transform(fees_norm).flatten()
    else:
        fees_log = (fees_norm.flatten() * (_FEE_LOG_TRAIN_MAX - _FEE_LOG_TRAIN_MIN)) + _FEE_LOG_TRAIN_MIN
    fees = np.expm1(fees_log)

    def top_distribution(series: pd.Series, k: int) -> list:
        counts = series.fillna("Other").value_counts()
        total = len(series)
        return [
            {"value": str(val), "count": int(cnt), "pct": round(float(cnt) / total, 3)}
            for val, cnt in counts.head(k).items()
        ]

    if profile_size >= 30:
        confidence = "high"
    elif profile_size >= 10:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "profile_size": profile_size,
        "confidence": confidence,
        "archetype": {
            "age": {
                "median": round(float(np.median(ages)), 1),
                "p25": round(float(np.percentile(ages, 25)), 1),
                "p75": round(float(np.percentile(ages, 75)), 1),
            },
            "market_value_eur": {
                "median": round(float(np.median(fees))),
                "p25": round(float(np.percentile(fees, 25))),
                "p75": round(float(np.percentile(fees, 75))),
            },
            "fee_type": top_distribution(profile["fee_type"], top_k_categories),
            "nationality": top_distribution(profile["nationality"], top_k_categories),
            "origin_league": top_distribution(profile["origin_league"], top_k_categories),
        },
    }
