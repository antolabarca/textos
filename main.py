import pandas as pd
from prompts.inference_prompt import build_inference_prompt
from models.openrouter import generate_with_openrouter
from util.safe_parse_json import safe_parse_json
from util.entropy import entropy_clasesocial, entropy_genero
from util.validate import validate_distribution
from util.max import max_clasesocial, max_genero
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    cohen_kappa_score, 
    balanced_accuracy_score)
import os
import json
from dotenv import load_dotenv
from test_data import data


data_df = pd.DataFrame(data)

raw_results = []
results = []

prompts = []
for _, row in data_df.iterrows():
    prompt = build_inference_prompt(row["text"])
    ans = generate_with_openrouter(prompt, "openai/gpt-4.1-mini")
    raw_results.append(ans)
    #print(repr(ans))
    parsed_ans = safe_parse_json(ans)
    if not validate_distribution(parsed_ans["genero"]["distribution"]):
        print("Invalid distribution")
    if not validate_distribution(parsed_ans["clasesocial"]["distribution"]):
        print("Invalid distribution")
    results.append({
        "id": row["id"],
        "text": row["text"],
        "pred_genero": parsed_ans["genero"]["response"],
        "conf_genero": parsed_ans["genero"]["confidence"],
        "p_mujer": parsed_ans["genero"]["distribution"]["mujer"],
        "p_hombre": parsed_ans["genero"]["distribution"]["hombre"],
        "p_nobinarie": parsed_ans["genero"]["distribution"]["nobinarie"],
        "pred_clasesocial": parsed_ans["clasesocial"]["response"],
        "conf_clasesocial": parsed_ans["clasesocial"]["confidence"],
        "p_alta": parsed_ans["clasesocial"]["distribution"]["alta"],
        "p_media": parsed_ans["clasesocial"]["distribution"]["media"],
        "p_baja": parsed_ans["clasesocial"]["distribution"]["baja"]
    })

raw_results_df = pd.DataFrame(raw_results)
results_df = pd.DataFrame(results)
#print(results_df.head())

print(results_df["pred_genero"].value_counts())
print(results_df["pred_clasesocial"].value_counts())
print(" ")
print("confianza género")
print(results_df["conf_genero"].mean())
print(results_df.groupby("pred_genero")["conf_genero"].mean())
print(" ")
print("confianza clasesocial")
print(results_df["conf_clasesocial"].mean())
print(results_df.groupby("pred_clasesocial")["conf_clasesocial"].mean())

results_df["entropia_genero"] = results_df.apply(
    entropy_genero,
    axis=1
)

results_df["entropia_clasesocial"] = results_df.apply(
    entropy_clasesocial,
    axis=1
)

print("correlación entre confianza y entropía")
print(results_df[["conf_genero","entropia_genero"]].corr())
print(results_df[["conf_clasesocial","entropia_clasesocial"]].corr())

print(" ")

results_df["max_genero"] = results_df.apply(
    max_genero,
    axis=1
)
results_df["max_clasesocial"] = results_df.apply(
    max_clasesocial,
    axis=1
)
# print(results_df.head())

merged_df = data_df.merge(
    results_df,
    on="id",
    how="inner"
)

#print(merged_df[["id","true_genero", "pred_genero"]].head())
print(" ")
print("accuracy")
print("genero")
print(accuracy_score(merged_df["true_genero"],merged_df["pred_genero"]))
print("clasesocial")
print(accuracy_score(merged_df["true_clasesocial"],merged_df["pred_clasesocial"]))

print(" ")
print("balanced accuracy")
print("genero")
print(balanced_accuracy_score(merged_df["true_genero"],merged_df["pred_genero"]))
print("clasesocial")
print(balanced_accuracy_score(merged_df["true_clasesocial"],merged_df["pred_clasesocial"]))

print(" ")
print("matrices de confusión")
print("genero")
print(confusion_matrix(merged_df["true_genero"],merged_df["pred_genero"]))
print("clasesocial")
print(confusion_matrix(merged_df["true_clasesocial"],merged_df["pred_clasesocial"]))

print(" ")
print("classification report")
print("genero")
print(classification_report(merged_df["true_genero"], merged_df["pred_genero"]))
print("clasesocial")
print(classification_report(merged_df["true_clasesocial"], merged_df["pred_clasesocial"]))

print(" ")
print("cohen kappa score")
print("genero")
print(cohen_kappa_score(merged_df["true_genero"], merged_df["pred_genero"]))
print("clasesocial")
print(cohen_kappa_score(merged_df["true_clasesocial"], merged_df["pred_clasesocial"]))

merged_df["correct_genero"] = (
    merged_df["true_genero"]
    ==
    merged_df["pred_genero"]
)

merged_df["correct_clasesocial"] = (
    merged_df["true_clasesocial"]
    ==
    merged_df["pred_clasesocial"]
)

print(" ")
print("confianza vs correctitud")
print("genero")
print(merged_df.groupby("correct_genero")["conf_genero"].mean())
print("clasesocial")
print(merged_df.groupby("correct_clasesocial")["conf_clasesocial"].mean())

print(" ")
print("entropía vs correctitud")
print("genero")
print(merged_df.groupby("correct_genero")["entropia_genero"].mean())
print("clasesocial")
print(merged_df.groupby("correct_clasesocial")["entropia_clasesocial"].mean())

summary = pd.DataFrame({
    "attribute": ["genero", "clasesocial"],
    "accuracy": [
        accuracy_score(
            merged_df["true_genero"],
            merged_df["pred_genero"]
        ),
        accuracy_score(
            merged_df["true_clasesocial"],
            merged_df["pred_clasesocial"]
        )
    ],
    "cohen kappa": [
        cohen_kappa_score(
            merged_df["true_genero"],
            merged_df["pred_genero"]
        ),
        cohen_kappa_score(
            merged_df["true_clasesocial"],
            merged_df["pred_clasesocial"]
        )
    ]
})

print(summary)