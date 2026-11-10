
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "processed" / "cleaned_data.csv"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

# Split before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Identify column types from training data
numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = [
    col for col in X_train.columns if col not in numeric_cols
]

numeric_pipe = SkPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = SkPipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

# Model definitions and parameter grids
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=42),
        {"classifier__C": [0.1, 1.0, 10.0]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=1),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
        },
    ),
    "XGBoost": (
        XGBClassifier(
            random_state=42, eval_metric="logloss", n_jobs=1
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        },
    ),
}

results = []
best_score = -1
best_name = None
best_pipeline = None

for name, (model, params) in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", model),
    ])

    search = GridSearchCV(
        pipeline,
        param_grid=params,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    score = float(search.best_score_)
    print("Best CV ROC-AUC:", round(score, 4))
    print("Best parameters:", search.best_params_)

    results.append({
        "model": name,
        "best_cv_roc_auc": score,
        "best_params": search.best_params_,
    })

    if score > best_score:
        best_score = score
        best_name = name
        best_pipeline = search.best_estimator_

# Save selected model and fitted preprocessor
joblib.dump(
    best_pipeline.named_steps["preprocessor"],
    MODEL_DIR / "preprocessor.pkl",
)
joblib.dump(
    best_pipeline.named_steps["classifier"],
    MODEL_DIR / "best_model.pkl",
)

# Save feature names for SHAP
feature_names = (
    best_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
    .tolist()
)

with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

# Save comparison results
with open(MODEL_DIR / "model_comparison.json", "w") as f:
    json.dump(sorted(
        results,
        key=lambda item: item["best_cv_roc_auc"],
        reverse=True,
    ), f, indent=2)

# Save split data so evaluation uses the exact same test rows
X_test.to_csv(MODEL_DIR / "X_test.csv", index=False)
y_test.to_csv(MODEL_DIR / "y_test.csv", index=False)

with open(MODEL_DIR / "training_info.json", "w") as f:
    json.dump({
        "selected_model": best_name,
        "best_cv_roc_auc": best_score,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": 42,
    }, f, indent=2)

print("\nSelected model:", best_name)
print("Best CV ROC-AUC:", round(best_score, 4))
print("Model artefacts saved to:", MODEL_DIR)

