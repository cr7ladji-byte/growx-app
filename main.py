import math
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GrowX API", version="1.0")

# ==========================================
# CONFIGURATION CORS (Résout l'erreur Failed to fetch)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (Render frontend, local, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Autorise tous les en-têtes
)

# ==========================================
# MODÈLES DE DONNÉES EN ENTRÉE
# ==========================================
class MatchInput(BaseModel):
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_team_xg: Optional[float] = None
    away_team_xg: Optional[float] = None

# ==========================================
# FONCTIONS MATHEMATIQUES (Poisson & Dixon-Coles)
# ==========================================
def poisson_pmf(k: int, mu: float) -> float:
    """Calcule la probabilité de Poisson pour k buts avec une moyenne mu."""
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.pow(mu, k) * math.exp(-mu)) / math.factorial(k)

def dixon_coles_adjustment(i: int, j: int, home_xg: float, away_xg: float, rho: float = -0.04) -> float:
    """Ajustement de Dixon-Coles pour les faibles scores (0-0, 1-0, 0-1, 1-1)."""
    if i == 0 and j == 0:
        return 1.0 - (home_xg * away_xg * rho)
    elif i == 0 and j == 1:
        return 1.0 + (home_xg * rho)
    elif i == 1 and j == 0:
        return 1.0 + (away_xg * rho)
    elif i == 1 and j == 1:
        return 1.0 - rho
    return 1.0

# ==========================================
# ENDPOINTS API
# ==========================================
@app.get("/")
def read_root():
    return {"message": "API GrowX opérationnelle et prête."}

@app.post("/predict")
def predict_match(data: MatchInput):
    # Gestion des noms de clés flexibles (home_xg ou home_team_xg)
    h_xg = data.home_xg if data.home_xg is not None else data.home_team_xg
    a_xg = data.away_xg if data.away_xg is not None else data.away_team_xg

    if h_xg is None or a_xg is None:
        raise HTTPException(
            status_code=422, 
            detail="Les valeurs xG à domicile et à l'extérieur sont requises."
        )

    # 1. Calcul de la matrice 6x6 (buts de 0 à 5)
    matrix = []
    scores_list = []
    total_prob = 0.0

    for i in range(6):  # Buts Équipe Domicile (0 à 5)
        row = []
        for j in range(6):  # Buts Équipe Extérieur (0 à 5)
            # Calcul Poisson de base
            p_i = poisson_pmf(i, h_xg)
            p_j = poisson_pmf(j, a_xg)
            prob = p_i * p_j
            
            # Ajustement Dixon-Coles
            adj = dixon_coles_adjustment(i, j, h_xg, a_xg)
            prob *= adj
            
            row.append(prob)
            scores_list.append({
                "score": f"{i}-{j}",
                "home": i,
                "away": j,
                "probability": prob
            })
            total_prob += prob
        matrix.append(row)

    # Normalisation des probabilités sur la matrice 6x6
    if total_prob > 0:
        for i in range(6):
            for j in range(6):
                matrix[i][j] /= total_prob
        for item in scores_list:
            item["probability"] /= total_prob

    # 2. Tri des scores pour obtenir le Top 3
    scores_sorted = sorted(scores_list, key=lambda x: x["probability"], reverse=True)
    top_scores = scores_sorted[:3]

    # 3. Calcul du Pari Gombo Spécial (Cumul des 4 scores les plus probables)
    gombo_items = scores_sorted[:4]
    gombo_selection = [item["score"] for item in gombo_items]
    gombo_probability = sum(item["probability"] for item in gombo_items)

    return {
        "status": "success",
        "home_xg": h_xg,
        "away_xg": a_xg,
        "top_scores": top_scores,
        "gombo_selection": gombo_selection,
        "gombo_probability": gombo_probability,
        "matrix": matrix
    }
