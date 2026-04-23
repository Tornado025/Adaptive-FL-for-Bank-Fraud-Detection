"""
Phase 1 -- Data Acquisition & Preprocessing
============================================
Loads the raw IEEE-CIS Fraud Detection CSVs (train_transaction.csv &
train_identity.csv), merges them, cleans missing values, encodes
categoricals, normalises numerics, and saves ieee_clean.csv.

Usage:
    py data/src/preprocess.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  (everything relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = REPO_ROOT                       # CSVs are in the repo root
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
DB_DIR = os.path.join(REPO_ROOT, "data", "databases")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


# ===================================================================
# 1. LOAD & MERGE
# ===================================================================
def load_and_merge() -> pd.DataFrame:
    """Load transaction + identity CSVs and merge on TransactionID."""
    print("[1/5] Loading raw CSVs ...")
    txn_path = os.path.join(RAW_DIR, "train_transaction.csv")
    id_path = os.path.join(RAW_DIR, "train_identity.csv")

    if not os.path.isfile(txn_path) or not os.path.isfile(id_path):
        sys.exit(
            f"ERROR: Cannot find raw CSVs.\n"
            f"  Expected:\n    {txn_path}\n    {id_path}\n"
            f"  Download from Kaggle and place them in the repo root."
        )

    txn = pd.read_csv(txn_path)
    idn = pd.read_csv(id_path)
    df = txn.merge(idn, on="TransactionID", how="left")
    # Downsample early to speed up pipeline, as downstream needs max 200k rows
    df = df.sample(n=min(200000, len(df)), random_state=42).reset_index(drop=True)
    print(f"       Merged and sampled shape: {df.shape}")
    return df


# ===================================================================
# 2. IMPUTE MISSING VALUES
# ===================================================================
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NaNs according to the project spec."""
    print("[2/5] Imputing missing values ...")
    from pandas.api.types import is_numeric_dtype

    # -- TransactionAmt: median per ProductCD group -----------------
    df["TransactionAmt"] = df.groupby("ProductCD")["TransactionAmt"].transform(
        lambda s: s.fillna(s.median())
    )

    # -- card columns (card1-card6) : mode or median ----------------
    card_cols = [c for c in df.columns if c.startswith("card")]
    for col in card_cols:
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("unknown")

    # -- addr1, addr2: fill with -1 --------------------------------
    df["addr1"] = df["addr1"].fillna(-1)
    df["addr2"] = df["addr2"].fillna(-1)

    # -- dist1, dist2: fill with 0 ---------------------------------
    for c in ["dist1", "dist2"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # -- email domains: fill with "unknown" -------------------------
    for c in ["P_emaildomain", "R_emaildomain"]:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")

    # -- id_* columns -----------------------------------------------
    id_cols = [c for c in df.columns if c.startswith("id_")]
    for col in id_cols:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(-1)

    # -- DeviceType, DeviceInfo -------------------------------------
    for c in ["DeviceType", "DeviceInfo"]:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")

    # -- M columns (M1-M9) -----------------------------------------
    m_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
    for col in m_cols:
        df[col] = df[col].fillna("unknown")

    # -- V1-V339: drop cols >80% missing, impute rest w/ median -----
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    v_missing_frac = df[v_cols].isna().mean()
    drop_v = v_missing_frac[v_missing_frac > 0.80].index.tolist()
    df.drop(columns=drop_v, inplace=True)
    print(f"       Dropped {len(drop_v)} V-columns (>80% missing)")

    # refresh v_cols after drop
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    for col in v_cols:
        df[col] = df[col].fillna(df[col].median())

    # -- C, D columns -- fill remaining numerics with median --------
    for prefix in ["C", "D"]:
        cols = [c for c in df.columns
                if c.startswith(prefix) and c[len(prefix):].isdigit()]
        for col in cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # -- Catch-all: any remaining NaNs ------------------------------
    for col in df.columns:
        if df[col].isna().any():
            if is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("unknown")

    remaining_nans = int(df.isna().sum().sum())
    print(f"       Remaining NaN count: {remaining_nans}")
    assert remaining_nans == 0, "There should be zero NaNs after imputation!"
    return df


# ===================================================================
# 3. CATEGORICAL ENCODING
# ===================================================================
def _extract_email_bucket(domain: str) -> str:
    """Map e-mail domains to ~10 buckets."""
    if domain == "unknown":
        return "unknown"
    domain = str(domain).lower().strip()
    known = [
        "gmail", "yahoo", "hotmail", "outlook", "aol",
        "comcast", "icloud", "msn", "live", "att",
    ]
    for k in known:
        if k in domain:
            return k
    return "other"


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns per spec."""
    print("[3/5] Encoding categoricals ...")

    # -- simple label-encode columns --------------------------------
    label_cols = ["ProductCD", "card4", "card6"]
    m_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
    label_cols += m_cols

    le_map: dict[str, LabelEncoder] = {}
    for col in label_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_map[col] = le

    # -- email domains: bucket then label encode --------------------
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            df[col] = df[col].apply(_extract_email_bucket)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_map[col] = le

    # -- DeviceType: label encode -----------------------------------
    if "DeviceType" in df.columns:
        df["DeviceType"] = df["DeviceType"].astype(str)
        le = LabelEncoder()
        df["DeviceType"] = le.fit_transform(df["DeviceType"])
        le_map["DeviceType"] = le

    # -- DeviceInfo: keep top-20, rest -> "other", label encode -----
    if "DeviceInfo" in df.columns:
        df["DeviceInfo"] = df["DeviceInfo"].astype(str)
        top20 = df["DeviceInfo"].value_counts().nlargest(20).index
        df["DeviceInfo"] = df["DeviceInfo"].where(
            df["DeviceInfo"].isin(top20), other="other"
        )
        le = LabelEncoder()
        df["DeviceInfo"] = le.fit_transform(df["DeviceInfo"])
        le_map["DeviceInfo"] = le

    # -- id_* categorical columns: label encode ---------------------
    id_cat_cols = [
        c for c in df.columns
        if c.startswith("id_") and df[c].dtype == "object"
    ]
    for col in id_cat_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_map[col] = le

    print(f"       Label-encoded {len(le_map)} columns")
    return df


# ===================================================================
# 4. NORMALISATION
# ===================================================================
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """StandardScaler on continuous cols; log1p + scale for TransactionAmt."""
    print("[4/5] Normalising numerical features ...")

    # -- TransactionAmt: log1p then StandardScaler ------------------
    df["TransactionAmt"] = np.log1p(df["TransactionAmt"])
    scaler_amt = StandardScaler()
    df["TransactionAmt"] = scaler_amt.fit_transform(
        df[["TransactionAmt"]]
    )

    # -- Identify columns to scale ----------------------------------
    #    Exclude: isFraud (label), TransactionID, TransactionAmt (done),
    #    and any columns that look binary (only contain 0/1).
    skip = {"isFraud", "TransactionID", "TransactionAmt"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = []
    for col in numeric_cols:
        if col in skip:
            continue
        uniq = df[col].dropna().unique()
        # skip binary flags
        if set(uniq).issubset({0, 1, 0.0, 1.0}):
            continue
        scale_cols.append(col)

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"       Scaled {len(scale_cols)} continuous columns")
    return df


# ===================================================================
# 5. DROP TransactionID & SAVE
# ===================================================================
def save_clean(df: pd.DataFrame) -> str:
    """Drop TransactionID (not a feature) and save to CSV."""
    print("[5/5] Saving cleaned dataset ...")
    df.drop(columns=["TransactionID"], inplace=True, errors="ignore")
    out_path = os.path.join(PROCESSED_DIR, "ieee_clean.csv")
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"       Saved -> {out_path}")
    print(f"       Final shape : {df.shape}")
    print(f"       NaN check   : {int(df.isna().sum().sum())}")
    print(f"       isFraud rate: {df['isFraud'].mean():.4f}")
    return out_path


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("  IEEE-CIS Fraud Detection -- Data Preprocessing (Phase 1)")
    print("=" * 60)
    df = load_and_merge()
    df = impute_missing(df)
    df = encode_categoricals(df)
    df = normalise(df)
    save_clean(df)
    print("\n[OK] Phase 1 complete.\n")


if __name__ == "__main__":
    main()
