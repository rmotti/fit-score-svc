import pickle
import json
import logging
import time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from app.scoring import build_all_baselines

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

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

    # Baselines de calibração (leave-one-out por club×posição×objetivo). Derivados
    # dos perfis em memória — imutáveis e idênticos em toda réplica, igual aos perfis.
    # ~3s pros 4.010 perfis × 4 objetivos; evita re-rodar o notebook / novo artefato.
    t0 = time.perf_counter()
    club_baselines = build_all_baselines(club_profiles)
    logger.info(
        f"Baselines de calibração: {len(club_baselines)} "
        f"(club×posição×objetivo) em {time.perf_counter() - t0:.1f}s"
    )
