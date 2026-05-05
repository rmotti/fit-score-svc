from pydantic import BaseModel, Field
from typing import Literal, Optional

Objective = Literal["rebuild", "youth", "title", "balanced"]
Confidence = Literal["high", "medium", "low", "none"]

class CandidateInput(BaseModel):
    nationality: Optional[str] = None
    origin_league: Optional[str] = None
    age: int = Field(..., ge=15, le=45)
    market_value_eur: Optional[float] = Field(None, ge=0)  # euros absolutos
    fee_type: Literal["paid", "free", "undisclosed"] = "paid"

class ScoreRequest(BaseModel):
    club_name: str
    position_group: Literal["GK","CB","LB","RB","DM","CM","LM","RM","AM","LW","RW","SS","CF"]
    candidate: CandidateInput
    objective: Objective = "balanced"
    sample_size: int = Field(2000, ge=100, le=10000)

class ScoreResult(BaseModel):
    fit_score: Optional[float]
    confidence: Confidence
    profile_size: int
    profile_found: bool

class ScoreResponse(BaseModel):
    club_name: str
    position_group: str
    objective: str
    result: ScoreResult

class BatchCandidate(BaseModel):
    candidate_id: str  # sofifaId como string
    candidate: CandidateInput

class BatchRequest(BaseModel):
    club_name: str
    position_group: Literal["GK","CB","LB","RB","DM","CM","LM","RM","AM","LW","RW","SS","CF"]
    objective: Objective = "balanced"
    candidates: list[BatchCandidate] = Field(..., min_length=1, max_length=100)
    sample_size: int = Field(2000, ge=100, le=10000)

class BatchResultItem(BaseModel):
    candidate_id: str
    fit_score: Optional[float]
    confidence: Confidence
    profile_size: int

class BatchResponse(BaseModel):
    club_name: str
    position_group: str
    objective: str
    results: list[BatchResultItem]


class RecommendRequest(BaseModel):
    club_name: str
    position_group: Literal["GK","CB","LB","RB","DM","CM","LM","RM","AM","LW","RW","SS","CF"]
    objective: Objective = "balanced"
    max_market_value_eur: Optional[float] = Field(None, ge=0)
    fee_type: Optional[Literal["paid", "free", "undisclosed"]] = None
    top_k: int = Field(20, ge=1, le=100)
    sample_size: int = Field(2000, ge=100, le=10000)


class RecommendResultItem(BaseModel):
    fit_score: float
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    nationality: Optional[str] = None
    origin_league: Optional[str] = None
    fee_type: Optional[str] = None


class RecommendResponse(BaseModel):
    club_name: str
    position_group: str
    objective: str
    profile_size: int
    candidates_evaluated: int
    results: list[RecommendResultItem]
