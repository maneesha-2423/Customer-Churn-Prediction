
from pathlib import Path
import json
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ml" / "models"
df = pd.read_csv(ROOT / "ml" / "processed" / "cleaned_data.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

preprocessor = ColumnTransformer([
    ("numeric", SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("categorical", SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

estimators = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42, n_jobs=1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        random_state=42, eval_metric="logloss", n_jobs=1
    ),
}

with open(MODEL_DIR / "model_comparison.json") as f:
    comparison = json.load(f)

best_name = comparison[0]["model"]

# Use the hyperparameters selected during GridSearchCV.
cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=2, random_state=42
)

scores = {}

for item in comparison:
    name = item["model"]
    estimator = estimators[name]
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", estimator),
    ])

    pipeline.set_params(**item["best_params"])

    print(f"Calculating repeated CV scores: {name}")
    scores[name] = cross_val_score(
        pipeline, X_train, y_train,
        scoring="roc_auc", cv=cv, n_jobs=-1
    ).tolist()

results = {}

for name, other_scores in scores.items():
    if name == best_name:
        continue

    try:
        statistic, p_value = wilcoxon(
            scores[best_name],
            other_scores,
            alternative="two-sided",
        )
        results[name] = {
            "comparison": f"{best_name} vs {name}",
            "best_model_mean_roc_auc": float(
                sum(scores[best_name]) / len(scores[best_name])
            ),
            "other_model_mean_roc_auc": float(
                sum(other_scores) / len(other_scores)
            ),
            "wilcoxon_statistic": float(statistic),
            "p_value": float(p_value),
            "significant_at_0_05": bool(p_value < 0.05),
        }
    except ValueError as exc:
        results[name] = {"error": str(exc)}

output = {
    "selected_model": best_name,
    "folds": 5,
    "repeats": 2,
    "scores": scores,
    "comparisons": results,
    "caveat": (
        "Exploratory paired comparisons: hyperparameters were selected "
        "using the same training data. Repeated folds are not independent "
        "observations, so p-values should be interpreted cautiously."
    ),
}

with open(MODEL_DIR / "statistical_test.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSelected model:", best_name)
print(json.dumps(results, indent=2))
print("\nSaved statistical_test.json")
