
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ml" / "models"

preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
model = joblib.load(MODEL_DIR / "best_model.pkl")

X_test = pd.read_csv(MODEL_DIR / "X_test.csv")

# Use a manageable sample for explainability
sample_size = min(200, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=42)

X_transformed = preprocessor.transform(X_sample)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)

with open(MODEL_DIR / "feature_names.json") as f:
    feature_names = json.load(f)

# Build a DataFrame so feature names are retained
X_explain = pd.DataFrame(
    X_transformed, columns=feature_names
)

# Explain the probability of churn (class 1).
# A model-agnostic explainer works across the four classifiers.
background = X_explain.sample(
    n=min(50, len(X_explain)), random_state=42
)

explainer = shap.Explainer(
    lambda data: model.predict_proba(
        np.asarray(data)
    )[:, 1],
    background,
)

shap_values = explainer(X_explain)

# Global feature importance: mean absolute SHAP value
importance = np.abs(shap_values.values).mean(axis=0)

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": importance,
}).sort_values("mean_abs_shap", ascending=False)

global_importance.to_csv(
    MODEL_DIR / "shap_global_importance.csv", index=False
)

print("\nTop 10 global churn drivers:")
print(global_importance.head(10).to_string(index=False))

# Individual explanations
individual_results = []

for row_idx in range(len(X_explain)):
    row_values = shap_values.values[row_idx]
    top_indices = np.argsort(np.abs(row_values))[::-1][:5]

    top_factors = []
    for idx in top_indices:
        top_factors.append({
            "feature": feature_names[int(idx)],
            "shap_value": float(row_values[idx]),
            "direction": (
                "increases churn prediction"
                if row_values[idx] > 0
                else "decreases churn prediction"
            ),
        })

    individual_results.append({
        "sample_index": int(row_idx),
        "predicted_churn_probability": float(
            model.predict_proba(X_transformed[row_idx:row_idx + 1])[0, 1]
        ),
        "top_factors": top_factors,
    })

with open(MODEL_DIR / "shap_customer_explanations.json", "w") as f:
    json.dump(individual_results, f, indent=2)

print("\nSaved global and individual SHAP explanations.")
