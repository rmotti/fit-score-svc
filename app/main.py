import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app import profiles as store
from app.schemas import (
    ScoreRequest, ScoreResponse,
    ExplainRequest, ExplainResponse,
    BatchRequest, BatchResponse, BatchResultItem,
    RecommendRequest, RecommendResponse,
    ArchetypeRequest, ArchetypeResponse,
)
from app.scoring import compute_fit_score, explain_fit_score, recommend_candidates, get_club_archetype

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
        club_baselines=store.club_baselines,
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


@app.post("/score/explain", response_model=ExplainResponse)
def score_explain(req: ExplainRequest):
    result = explain_fit_score(
        club_profiles=store.club_profiles,
        club_baselines=store.club_baselines,
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
    return ExplainResponse(
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
            club_baselines=store.club_baselines,
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


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    data = recommend_candidates(
        club_profiles=store.club_profiles,
        position_index=store.position_index,
        club_baselines=store.club_baselines,
        club_name=req.club_name,
        position_group=req.position_group,
        objective=req.objective,
        max_market_value_eur=req.max_market_value_eur,
        fee_type=req.fee_type,
        fee_scaler=store.fee_scaler,
        top_k=req.top_k,
        sample_size=req.sample_size,
    )
    if data.get("error") == "profile_not_found":
        raise HTTPException(status_code=404, detail=f"Perfil não encontrado para '{req.club_name}' / {req.position_group}")
    return RecommendResponse(
        club_name=req.club_name,
        position_group=req.position_group,
        objective=req.objective,
        profile_size=data["profile_size"],
        candidates_evaluated=data["candidates_evaluated"],
        results=data["results"],
    )


@app.post("/profile", response_model=ArchetypeResponse)
def profile(req: ArchetypeRequest):
    data = get_club_archetype(
        club_profiles=store.club_profiles,
        club_name=req.club_name,
        position_group=req.position_group,
        objective=req.objective,
        age_scaler=store.age_scaler,
        fee_scaler=store.fee_scaler,
        top_k_categories=req.top_k_categories,
    )
    if data.get("error") == "profile_not_found":
        raise HTTPException(status_code=404, detail=f"Perfil não encontrado para '{req.club_name}' / {req.position_group}")
    if data.get("error") == "profile_too_small":
        raise HTTPException(status_code=422, detail=f"Perfil muito pequeno (< 5 transferências) para '{req.club_name}' / {req.position_group}")
    return ArchetypeResponse(
        club_name=req.club_name,
        position_group=req.position_group,
        objective=req.objective,
        profile_size=data["profile_size"],
        confidence=data["confidence"],
        archetype=data["archetype"],
    )


@app.get("/metadata")
def metadata():
    return store.metadata
