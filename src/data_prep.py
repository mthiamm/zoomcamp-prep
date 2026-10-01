# """Load, clean, and split the UCI Bank Marketing dataset.

# Run from the repo root:
#     python src/data_prep.py

# Outputs -> data/processed/:
#     X_train.parquet, X_test.parquet, y_train.parquet, y_test.parquet
# """

# from pathlib import Path

# import pandas as pd
# from sklearn.model_selection import train_test_split

# RAW_DATA_PATH = Path("data/raw/bank-full.csv")
# PROCESSED_DIR = Path("data/processed")
# TARGET = "y"
# RANDOM_STATE = 42
# TEST_SIZE = 0.2

# # LEAKAGE WARNING: 'duration' = length of the final sales call. It is only
# # known AFTER the call happens, so a real system cannot use it when deciding
# # whom to call. Keeping it makes offline metrics look great and production
# # models fail. Classic interview trap — we drop it on purpose.
# DROP_COLUMNS = ["duration"]


# def load_data(path: Path):
#     """UCI ships this file semicolon-separated, not comma-separated."""
#     df = pd.read_csv(path, sep=";")
#     print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
#     return df


# def clean_data(df: pd.DataFrame):
#     df = df.drop(columns=DROP_COLUMNS)
#     # Convert every yes/no column (including the target) to 1/0
#     for col in df.columns:
#         if set(df[col].unique()) <= {"yes", "no"}:
#             df[col] = (df[col] == "yes").astype(int)
#     return df


# def encode_features(df: pd.DataFrame):
#     """One-hot encode the remaining categorical columns."""
#     categoricals = df.select_dtypes(include="object").columns.tolist()
#     return pd.get_dummies(df, columns=categoricals, dtype=int)


# def main() -> None:
#     df = load_data(RAW_DATA_PATH)
#     df = clean_data(df)

#     X = encode_features(df.drop(columns=TARGET))
#     y = df[TARGET]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
#     )

#     PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
#     X_train.to_parquet(PROCESSED_DIR / "X_train.parquet")
#     X_test.to_parquet(PROCESSED_DIR / "X_test.parquet")
#     y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet")
#     y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet")

#     print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
#     print(f"Features after encoding: {X.shape[1]}")
#     print(f"Positive rate (train): {y_train.mean():.1%}")


# if __name__ == "__main__":
#     main()
# # ===========================

"""Load, clean, and split the UCI Bank Marketing dataset.

Run from the repo root:
    python src/data_prep.py

Outputs -> data/processed/:
    train.parquet, val.parquet, test.parquet
    feature_columns.json   (schema contract for serving)
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/bank-full.csv")
PROCESSED_DIR = Path("data/processed")

TARGET = "y"
RANDOM_STATE = 42
VAL_SIZE = 0.2  # 20% of the full data
TEST_SIZE = 0.2  # 20% of the full data
# -> leaves 60% for training

# LEAKAGE WARNING: 'duration' = length of the final sales call. It is only
# known AFTER the call happens, so a real system cannot use it when deciding
# whom to call. Keeping it makes offline metrics look great and production
# models fail. Classic interview trap — we drop it on purpose.
LEAKAGE_COLUMNS = ["duration"]


def load_data(path: Path) -> pd.DataFrame:
    """UCI ships this file semicolon-separated, not comma-separated."""
    df = pd.read_csv(path, sep=";")
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    return df


def detect_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect numeric vs categorical columns from dtypes.

    NOTE: dtypes are a proxy for semantics. 'day' is int64 but cyclical;
    'pdays' uses -1 as a sentinel. We leave that for feature engineering
    in the training pipeline, not here.
    """
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=LEAKAGE_COLUMNS)

    # Convert every yes/no column (including the target) to 1/0
    for col in df.columns:
        if set(df[col].unique()) <= {"yes", "no"}:
            df[col] = (df[col] == "yes").astype(int)
    return df


def split_data(df: pd.DataFrame):
    """Stratified 60/20/20 train/val/test on the target."""
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    # VAL_SIZE is expressed as a fraction of the FULL dataset, so convert
    # it to a fraction of train_val before the second split.
    val_frac_of_train_val = VAL_SIZE / (1.0 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_frac_of_train_val,
        random_state=RANDOM_STATE,
        stratify=train_val[TARGET],
    )
    return train, val, test


def encode_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode on train, then align val/test to train's columns.

    This is the leakage-safe pattern: fit the encoder on train only,
    then reindex val/test so they can never introduce unseen columns.
    """
    y_train = train[TARGET]
    y_val = val[TARGET]
    y_test = test[TARGET]

    X_train = train.drop(columns=TARGET)
    X_val = val.drop(columns=TARGET)
    X_test = test.drop(columns=TARGET)

    X_train = pd.get_dummies(X_train, dtype=int)
    X_val = pd.get_dummies(X_val, dtype=int).reindex(
        columns=X_train.columns, fill_value=0
    )
    X_test = pd.get_dummies(X_test, dtype=int).reindex(
        columns=X_train.columns, fill_value=0
    )

    # Reattach target so we write single files
    X_train[TARGET] = y_train
    X_val[TARGET] = y_val
    X_test[TARGET] = y_test

    return X_train, X_val, X_test, X_train.columns.tolist()


def main() -> None:
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)

    numeric_cols, categorical_cols = detect_column_types(df.drop(columns=TARGET))
    print(f"Dropped leakage columns: {LEAKAGE_COLUMNS}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    train, val, test = split_data(df)
    X_train, X_val, X_test, feature_columns = encode_features(train, val, test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    X_val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    X_test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    # Write the schema contract — M4 (serving) will read this to validate inputs
    schema = {
        "target": TARGET,
        "leakage_columns": LEAKAGE_COLUMNS,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
    }
    (PROCESSED_DIR / "feature_columns.json").write_text(json.dumps(schema, indent=2))

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Features after encoding: {len(feature_columns)}")
    print(
        f"Positive rate — train: {X_train[TARGET].mean():.3f} | "
        f"val: {X_val[TARGET].mean():.3f} | test: {X_test[TARGET].mean():.3f}"
    )


if __name__ == "__main__":
    main()
