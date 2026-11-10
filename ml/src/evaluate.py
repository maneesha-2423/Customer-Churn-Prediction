
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# Project paths
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ml" / "models"


def evaluate_model():
    # Check that required files exist
    required_files = [
        "preprocessor.pkl",
        "best_model.pkl",
        "X_test.csv",
        "y_test.csv",
    ]

    for filename in required_files:
        filepath = MODEL_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required file not found: {filepath}\n"
                "Please run src/train.py successfully first."
            )

    # Load the trained preprocessor and model
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    model = joblib.load(MODEL_DIR / "best_model.pkl")

    # Load test data
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    y_test = (
        pd.read_csv(MODEL_DIR / "y_test.csv")
        .iloc[:, 0]
        .astype(int)
    )

    # Transform test features using the fitted preprocessor
    X_transformed = preprocessor.transform(X_test)

    # Generate predictions
    y_pred = model.predict(X_transformed)

    # Calculate probabilities when supported
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_transformed)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y_test, y_prob))
            if y_prob is not None
            else None
        ),
        "pr_auc": (
            float(average_precision_score(y_test, y_prob))
            if y_prob is not None
            else None
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # Save results to JSON
    output_path = MODEL_DIR / "metrics.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Display results
    print("\n========== FINAL TEST RESULTS ==========")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif key == "confusion_matrix":
            print(f"{key}:")
            print(value)
        elif key != "classification_report":
            print(f"{key}: {value}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    print(f"Metrics saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
