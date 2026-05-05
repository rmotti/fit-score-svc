# Fit Score Service

A machine learning-powered REST API that evaluates how well football (soccer) players fit specific team profiles and positions. It computes similarity scores between candidates and historical transfer patterns using Gower distance metrics.

## Overview

The service supports four core use cases:

- **Scorer** — validate transfer rumors by computing a fit score for an individual candidate
- **Batch Scorer** — process multiple candidates simultaneously
- **Recommender** — suggest best-fitting players for a specific club/position combination
- **Health & Metadata** — system status and model information

The model was trained on 62,383 transfers (1999–2027) spanning 3,872 clubs and 13 position groups.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Framework | FastAPI 0.115.0 |
| Server | Uvicorn 0.30.6 |
| ML | scikit-learn, gower, numpy, pandas |
| Deployment | Docker / Railway |

## Project Structure

```
fit-score-svc/
├── app/
│   ├── main.py         # FastAPI app and endpoint definitions
│   ├── schemas.py      # Pydantic request/response models
│   ├── scoring.py      # Core ML logic (Gower distance computation)
│   └── profiles.py     # Artifact loading and caching
├── artifacts/
│   ├── club_profiles.pkl      # (club, position) → transfer history
│   ├── position_index.pkl     # position → all candidates
│   ├── age_scaler.pkl         # MinMaxScaler for age normalization
│   ├── fee_scaler.pkl         # MinMaxScaler for market value normalization
│   ├── model_metadata.json    # Model statistics and metadata
│   └── FitScoreMl.ipynb       # Full ML pipeline notebook
├── generate_scalers.py        # Script to regenerate scaler artifacts
├── Dockerfile
├── railway.toml
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with artifact status and model version |
| `GET` | `/metadata` | Full model metadata (clubs, leagues, nationalities, positions) |
| `POST` | `/score` | Compute fit score for a single candidate |
| `POST` | `/score/batch` | Compute fit scores for 1–100 candidates |
| `POST` | `/recommend` | Return top-K players matching a club/position profile |

Interactive API docs are available at `/docs` when the server is running.

## Getting Started

### Prerequisites

- Python 3.11+
- ML artifacts in the `artifacts/` directory (not tracked in git — see [Artifacts](#artifacts))

### Local Development

```bash
# Clone the repository
git clone https://github.com/rmotti/fit-score-svc.git
cd fit-score-svc

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Unix/macOS

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t fit-score-svc .
docker run -p 8000:8000 fit-score-svc
```

### Railway

The project includes a `railway.toml` configured with:

- Builder: Dockerfile
- Health check: `GET /health` (30s timeout)
- Restart policy: `on_failure`

Push to the connected GitHub repository to trigger an automatic deployment.

## Artifacts

The ML artifacts (`.pkl` files) are excluded from version control. To regenerate the scalers from known training bounds, run:

```bash
python generate_scalers.py
```

The club profiles and position index must be generated from the training pipeline in `artifacts/FitScoreMl.ipynb`.

## Model Details

| Property | Value |
|---|---|
| Total transfers | 62,383 |
| Clubs | 3,872 |
| Club profiles | 4,010 |
| Position groups | GK, CB, LB, RB, DM, CM, LM, RM, AM, LW, RW, SS, CF |
| Origin leagues | 32 + "Other" |
| Nationalities | 69 + "Other" |
| Transfer objectives | rebuild, youth, title, balanced |
| Date range | Dec 1999 – Jul 2027 |

### How the Model Works

#### 1. Artifacts loaded at startup

On startup, `profiles.py` loads four artifacts into memory:

- **`club_profiles.pkl`** — dictionary `(club, position) → DataFrame` with the transfer history for that club/position pair. 4,010 profiles covering 3,872 clubs.
- **`position_index.pkl`** — dictionary `position → DataFrame` with all players in the dataset grouped by position. Used by the recommender endpoint.
- **`age_scaler.pkl` / `fee_scaler.pkl`** — scikit-learn `MinMaxScaler` instances fitted on training data to normalize age and market value.

#### 2. Features

The model operates on 5 features per player:

| Feature | Type | Description |
|---|---|---|
| `nationality` | Categorical | Player nationality |
| `origin_league` | Categorical | League of origin |
| `age_norm` | Numerical | Age normalized to [0, 1] |
| `fee_norm` | Numerical | Market value normalized to [0, 1] |
| `fee_type` | Categorical | Transfer type (e.g. free, loan, fee) |

#### 3. Normalization

Before any distance is computed, numerical values are normalized:

- **Age**: applied directly through `MinMaxScaler` (range: 14.8–43.3 years).
- **Market value**: `log1p` is applied first to compress the exponential scale, then `MinMaxScaler`. The training maximum is `log1p(222,000,000 EUR)`.

#### 4. Gower Distance — the core

The central computation uses **Gower distance**, a metric that natively handles mixed feature types:

- **Numerical features**: normalized absolute difference.
- **Categorical features**: `0` if equal, `1` if different.
- **Final value**: weighted average across features, always in `[0, 1]`.

This avoids the need for manual encoding when mixing "same league?" with "age difference".

#### 5. Fit Score computation (`/score`)

```
candidate → feature vector
club profile → DataFrame of historical transfers

combined = [candidate] + [all profile rows]
dist_matrix = gower_matrix(combined)

distances = dist_matrix[row 0, columns 1:]   # candidate ↔ each historical transfer
fit_score = 1.0 - mean(distances)
```

The higher the similarity between the candidate and the club's historical signings for that position, the higher the score. `1.0` = perfect fit, `0.0` = no fit.

Objective filters are applied **before** the calculation, restricting the historical profile:

| Objective | Filter |
|---|---|
| `balanced` | no filter |
| `rebuild` | `age_norm ≤ 0.35` (≈ up to 25 years old) |
| `youth` | `age_norm ≤ 0.25` (≈ up to 22 years old) |
| `title` | `fee_norm ≥ 0.50` (high-value players) |

#### 6. Recommendation (`/recommend`)

Instead of evaluating one candidate, the recommender scores **all players in `position_index`** at once:

```
combined = [all candidates for the position] + [club's historical profile]
dist_matrix = gower_matrix(combined)

mean_distances = mean distance from each candidate to all profile rows
fit_scores = 1.0 - mean_distances

→ sort by fit_score, deduplicate by player_id, return top-K
```

Optional pre-filters: `max_market_value_eur` and `fee_type`.

#### 7. Confidence

The `confidence` field reflects the size of the historical profile after applying the objective filter:

| Profile size | Confidence |
|---|---|
| ≥ 30 transfers | `high` |
| ≥ 10 transfers | `medium` |
| ≥ 5 transfers | `low` |
| < 5 transfers | no result returned |

#### Flow summary

```
request (club, position, candidate, objective)
    ↓
filter club history by objective
    ↓
normalize candidate age and fee
    ↓
build combined matrix [candidate + history]
    ↓
Gower distance → candidate ↔ history distances
    ↓
fit_score = 1 - mean(distances)
    ↓
response { fit_score, confidence, profile_size }
```

## License

MIT
