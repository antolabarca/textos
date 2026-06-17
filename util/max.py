def max_genero(row):
    probs = [
        row["p_mujer"],
        row["p_hombre"],
        row["p_nobinarie"]
    ]
    return max(probs)

def max_clasesocial(row):
    probs = [
        row["p_alta"],
        row["p_media"],
        row["p_baja"]
    ]
    return max(probs)
