"""Load, clean, and split the UCI Bank Marketing dataset.

Run from the repo root:
    python src/data_prep.py

Outputs -> data/processed/:
    X_train.parquet, X_test.parquet, y_train.parquet, y_test.parquet
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/bank-full.csv")
PROCESSED_DIR = Path("data/processed")
TARGET = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# LEAKAGE WARNING: 'duration' = length of the final sales call. It is only
# known AFTER the call happens, so a real system cannot use it when deciding
# whom to call. Keeping it makes offline metrics look great and production
# models fail. Classic interview trap — we drop it on purpose.
DROP_COLUMNS = ["duration"]


def load_data(path: Path):
    """UCI ships this file semicolon-separated, not comma-separated."""
    df = pd.read_csv(path, sep=";")
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    return df


def clean_data(df: pd.DataFrame):
    df = df.drop(columns=DROP_COLUMNS)
    # Convert every yes/no column (including the target) to 1/0
    for col in df.columns:
        if set(df[col].unique()) <= {"yes", "no"}:
            df[col] = (df[col] == "yes").astype(int)
    return df


def encode_features(df: pd.DataFrame):
    """One-hot encode the remaining categorical columns."""
    categoricals = df.select_dtypes(include="object").columns.tolist()
    return pd.get_dummies(df, columns=categoricals, dtype=int)


def main() -> None:
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)

    X = encode_features(df.drop(columns=TARGET))
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test.to_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet")
    y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet")

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Features after encoding: {X.shape[1]}")
    print(f"Positive rate (train): {y_train.mean():.1%}")


if __name__ == "__main__":
    main()
