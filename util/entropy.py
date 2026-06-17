import numpy as np

def entropy_genero(row):
    probs = [
        row["p_mujer"],
        row["p_hombre"],
        row["p_nobinarie"]
    ]
    probs = [p for p in probs if p > 0]
    return -sum(p * np.log2(p) for p in probs)

def entropy_clasesocial(row):
    probs = [
        row["p_alta"],
        row["p_media"],
        row["p_baja"]
    ]
    probs = [p for p in probs if p > 0]
    return -sum(p * np.log2(p) for p in probs)

