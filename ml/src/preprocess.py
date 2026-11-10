from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "dataset" / "Telco_churn.xlsx.xlsx"
OUTPUT_DIR = ROOT / "ml" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Remove customer ID
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

# Clean and encode target
if "Churn" not in df.columns:
    raise ValueError("Churn column not found in dataset.")

df["Churn"] = df["Churn"].astype("string").str.strip()

df = df[df["Churn"].isin(["Yes", "No"])].copy()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

# Save cleaned dataset
output_path = OUTPUT_DIR / "cleaned_data.csv"
df.to_csv(output_path, index=False)

print("Cleaned shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("\nChurn distribution:\n", df["Churn"].value_counts())
print(f"\nCleaned dataset saved to: {output_path}")