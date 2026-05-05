import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app import profiles as store
from app.schemas import (
    ScoreRequest, ScoreResponse,
    BatchRequest, BatchResponse, BatchResultItem,
)
from app.scoring import compute_fit_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load_artifacts()
    yield


app = FastAPI(title="Fit Score Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "profiles_loaded": len(store.club_profiles),
        "position_groups": len(store.position_index),
        "fee_scaler": store.fee_scaler is not None,
        "age_scaler": store.age_scaler is not None,
        "model_version": store.metadata.get("date_range", [None])[-1],
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    result = compute_fit_score(
        club_profiles=store.club_profiles,
        position_index=store.position_index,
        club_name=req.club_name,
        position_group=req.position_group,
        nationality=req.candidate.nationality,
        origin_league=req.candidate.origin_league,
        age=req.candidate.age,
        market_value_eur=req.candidate.market_value_eur,
        fee_type=req.candidate.fee_type,
        objective=req.objective,
        age_scaler=store.age_scaler,
        fee_scaler=store.fee_scaler,
        sample_size=req.sample_size,
    )
    return ScoreResponse(
        club_name=req.club_name,
        position_group=req.position_group,
        objective=req.objective,
        result=result,
    )


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(req: BatchRequest):
    results = []
    for item in req.candidates:
        result = compute_fit_score(
            club_profiles=store.club_profiles,
            position_index=store.position_index,
            club_name=req.club_name,
            position_group=req.position_group,
            nationality=item.candidate.nationality,
            origin_league=item.candidate.origin_league,
            age=item.candidate.age,
            market_value_eur=item.candidate.market_value_eur,
            fee_type=item.candidate.fee_type,
            objective=req.objective,
            age_scaler=store.age_scaler,
            fee_scaler=store.fee_scaler,
            sample_size=req.sample_size,
        )
        results.append(BatchResultItem(candidate_id=item.candidate_id, **result))

    return BatchResponse(
        club_name=req.club_name,
        position_group=req.position_group,
        objective=req.objective,
        results=results,
    )


@app.get("/metadata")
def metadata():
    return store.metadata
