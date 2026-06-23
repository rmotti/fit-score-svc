import pickle
import json
import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.scoring import build_all_baselines

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

# Agrupamento de posições em buckets amplos (paliativo p/ o dataset pequeno: a mediana
# por (clube × posição) é ~7 transferências; agregando em buckets a mediana sobe p/ ~9 e
# a fração de perfis robustos (n>=10) vai de 21% p/ 48%). Os buckets são adicionados em
# memória ALÉM dos 13 grupos granulares — as 13 chaves antigas continuam funcionando,
# então é backward-compatible. GK fica isolado (não há com o que agrupar).
# Trade-off: perde-se granularidade tática (CF+SS+RW+LW viram "ATT"); aceitável enquanto
# o cohort granular é raso demais p/ calibrar bem. Reverter = remover esta seção.
POSITION_BUCKETS: dict[str, str] = {
    "GK": "GK",
    "CB": "DEF", "RB": "DEF", "LB": "DEF",
    "DM": "MID", "CM": "MID", "AM": "MID", "RM": "MID", "LM": "MID",
    "CF": "ATT", "SS": "ATT", "RW": "ATT", "LW": "ATT",
}

club_profiles: dict = {}
position_index: dict = {}
club_baselines: dict = {}
metadata: dict = {}
age_scaler: MinMaxScaler | None = None
fee_scaler: MinMaxScaler | None = None


def load_artifacts():
    global club_profiles, position_index, club_baselines, metadata, age_scaler, fee_scaler

    logger.info("Carregando artefatos...")

    with open(ARTIFACTS_DIR / "club_profiles.pkl", "rb") as f:
        club_profiles = pickle.load(f)

    with open(ARTIFACTS_DIR / "position_index.pkl", "rb") as f:
        position_index = pickle.load(f)

    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    scaler_path = ARTIFACTS_DIR / "fee_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            fee_scaler = pickle.load(f)
        logger.info("fee_scaler.pkl carregado")
    else:
        logger.warning("fee_scaler.pkl não encontrado — fee_norm será estimado")

    age_scaler_path = ARTIFACTS_DIR / "age_scaler.pkl"
    if age_scaler_path.exists():
        with open(age_scaler_path, "rb") as f:
            age_scaler = pickle.load(f)
        logger.info("age_scaler.pkl carregado")
    else:
        logger.warning("age_scaler.pkl não encontrado — age_norm será estimado")

    logger.info(
        f"Artefatos carregados: {len(club_profiles)} perfis, "
        f"{len(position_index)} grupos de posição"
    )

    # Buckets de posição: concatena os perfis granulares de cada bucket numa nova chave
    # (club, "MID" / "DEF" / "ATT" / "GK"). Feito ANTES dos baselines p/ que estes saiam
    # de graça (build_all_baselines deriva de tudo que está no dict). As chaves granulares
    # são preservadas; nenhuma função de scoring precisa mudar — só recebe o nome do bucket.
    _add_position_buckets()
    logger.info(
        f"Buckets de posição aplicados: {len(club_profiles)} perfis no total "
        f"(13 granulares + buckets agregados)"
    )

    # Baselines de calibração (leave-one-out por club×posição×objetivo). Derivados
    # dos perfis em memória — imutáveis e idênticos em toda réplica, igual aos perfis.
    # ~3s pros 4.010 perfis × 4 objetivos; evita re-rodar o notebook / novo artefato.
    t0 = time.perf_counter()
    club_baselines = build_all_baselines(club_profiles)
    logger.info(
        f"Baselines de calibração: {len(club_baselines)} "
        f"(club×posição×objetivo) em {time.perf_counter() - t0:.1f}s"
    )


def _add_position_buckets():
    """Adiciona entradas de bucket (club, BUCKET) a club_profiles e position_index,
    concatenando os perfis/candidatos das posições granulares mapeadas pra cada bucket.
    Idempotente: pula buckets que coincidem com um grupo já existente (ex: GK)."""
    global club_profiles, position_index

    # club_profiles: agrega por (club, bucket)
    grouped: dict[tuple[str, str], list] = {}
    for (club, pos), profile in list(club_profiles.items()):
        bucket = POSITION_BUCKETS.get(pos)
        if bucket is None or bucket == pos:
            continue  # posição sem bucket, ou bucket == granular (GK) — nada a agregar
        grouped.setdefault((club, bucket), []).append(profile)

    for (club, bucket), frames in grouped.items():
        if (club, bucket) in club_profiles:
            continue  # já existe (não sobrescreve)
        club_profiles[(club, bucket)] = pd.concat(frames, ignore_index=True)

    # position_index: agrega os candidatos por bucket (mesma lógica)
    idx_grouped: dict[str, list] = {}
    for pos, candidates in list(position_index.items()):
        bucket = POSITION_BUCKETS.get(pos)
        if bucket is None or bucket == pos:
            continue
        idx_grouped.setdefault(bucket, []).append(candidates)

    for bucket, frames in idx_grouped.items():
        if bucket in position_index:
            continue
        position_index[bucket] = pd.concat(frames, ignore_index=True)
