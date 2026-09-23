import numpy as np
from scipy.stats import poisson
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="GrowX API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def dixon_coles_tau(x: int, y: int, lambda_x: float, mu_y: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - (lambda_x * mu_y * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_x * rho)
    elif x == 1 and y == 0:
        return 1.0 + (mu_y * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0

class ScoreDetail(BaseModel):
    score: str
    probabilite_pct: float
    cote_equitable: float

class PredictionResponse(BaseModel):
    xg_home: float
    xg_away: float
    top_3_scores: List[ScoreDetail]
    matrix: List[List[float]]
    gombo_couverture: str
    gombo_securite: str

@app.get("/predict", response_model=PredictionResponse)
def predict_score(
    xg_home: float = Query(1.85, gt=0),
    xg_away: float = Query(1.20, gt=0),
    rho: float = Query(-0.13)
):
    matrice = np.zeros((6, 6))
    prob_btts = 0.0
    
    for h in range(6):
        for a in range(6):
            p_h = poisson.pmf(h, xg_home)
            p_a = poisson.pmf(a, xg_away)
            tau = dixon_coles_tau(h, a, xg_home, xg_away, rho)
            val = max(0.0, p_h * p_a * tau)
            matrice[h, a] = val
            if h > 0 and a > 0:
                prob_btts += val

    matrice /= np.sum(matrice)
    
    scores_flat = []
    for h in range(6):
        for a in range(6):
            prob = float(matrice[h, a])
            fair_odds = round(1.0 / prob, 2) if prob > 0 else 0.0
            scores_flat.append(
                ScoreDetail(
                    score=f"{h} - {a}",
                    probabilite_pct=round(prob * 100, 2),
                    cote_equitable=fair_odds
                )
            )
            
    scores_tries = sorted(scores_flat, key=lambda x: x.probabilite_pct, reverse=True)[:3]
    top3_str = " OU ".join([s.score for s in scores_tries])
    
    pari_securite = "Les 2 équipes marquent (BTTS : OUI)" if prob_btts > 0.50 else "Moins de 3.5 Buts dans le match"

    return PredictionResponse(
        xg_home=xg_home,
        xg_away=xg_away,
        top_3_scores=scores_tries,
        matrix=np.round(matrice * 100, 2).tolist(),
        gombo_couverture=f"Score Exact Multi : {top3_str}",
        gombo_securite=pari_securite
    )

