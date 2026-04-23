"""
Phase 2 -- Non-IID Skewing Engine
==================================
Partitions the cleaned ieee_clean.csv into 4 bank profiles with
deliberately different distributions, then visualises the skew.

Usage:
    py data/src/skewing_engine.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "skew_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ===================================================================
# PARTITIONING
# ===================================================================
MIN_ROWS = 10_000   # each bank must have at least this many rows
MAX_ROWS = 50_000   # cap per bank to keep files small


def _ensure_min_rows(df: pd.DataFrame, full_df: pd.DataFrame,
                     name: str) -> pd.DataFrame:
    """If the filtered subset is too small, sample extra rows to reach MIN_ROWS."""
    if len(df) >= MIN_ROWS:
        return df
    shortfall = MIN_ROWS - len(df)
    extra_pool = full_df.drop(index=df.index, errors="ignore")
    extra = extra_pool.sample(n=min(shortfall, len(extra_pool)),
                              random_state=42)
    df = pd.concat([df, extra], ignore_index=True)
    print(f"       [{name}] padded to {len(df)} rows (needed {shortfall} extra)")
    return df


def partition_data(clean_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns {'bank_a': df_a, 'bank_b': df_b, 'bank_c': df_c, 'bank_d': df_d}

    Bank profiles (from README):
      A -- Domestic Retail:  addr2 mode (US), oversample ProductCD C/W, cap TxnAmt @75p
      B -- Int'l Corporate:  addr2 != mode, top-40% amt, oversample ProductCD H
      C -- High-Fraud Region: oversample fraud to ~15%
      D -- Card-Not-Present E-commerce: DeviceType desktop, visa+credit, high V variance
    """
    print("[1/3] Partitioning into 4 bank profiles ...")

    # ---- Bank A -- Domestic Retail --------------------------------
    # addr2 has been StandardScaled, so the literal 87 is gone.
    # We use the mode (the most frequent value, which originally was 87).
    mask_a = clean_df["addr2"] == clean_df["addr2"].mode()[0]
    bank_a = clean_df[mask_a].copy()
    # Oversample ProductCD values for consumer/wallet (C and W)
    pcd_counts = bank_a["ProductCD"].value_counts()
    if len(pcd_counts) >= 3:
        oversample_pcds = pcd_counts.index[1:3].tolist()
        oversample_rows = bank_a[bank_a["ProductCD"].isin(oversample_pcds)]
        bank_a = pd.concat([bank_a, oversample_rows], ignore_index=True)
    # Cap TransactionAmt at 75th percentile
    cap = bank_a["TransactionAmt"].quantile(0.75)
    bank_a = bank_a[bank_a["TransactionAmt"] <= cap].copy()
    bank_a = _ensure_min_rows(bank_a, clean_df, "Bank A")

    # ---- Bank B -- International Corporate -----------------------
    mask_b = clean_df["addr2"] != clean_df["addr2"].mode()[0]
    bank_b = clean_df[mask_b].copy()
    # Keep only high-value transactions (top 40%)
    amt_60p = bank_b["TransactionAmt"].quantile(0.60)
    bank_b = bank_b[bank_b["TransactionAmt"] >= amt_60p].copy()
    # Oversample highest ProductCD value (mapped from H)
    pcd_top = bank_b["ProductCD"].value_counts().index[-1]
    oversample_h = bank_b[bank_b["ProductCD"] == pcd_top]
    bank_b = pd.concat([bank_b, oversample_h], ignore_index=True)
    bank_b = _ensure_min_rows(bank_b, clean_df, "Bank B")

    # ---- Bank C -- High-Fraud Region (target 15% fraud) ----------
    fraud = clean_df[clean_df["isFraud"] == 1]
    legit = clean_df[clean_df["isFraud"] == 0]
    target_fraud_ratio = 0.15
    # Take all fraud rows, then sample enough legit to hit ratio
    n_fraud = len(fraud)
    n_legit_needed = int(n_fraud * (1 - target_fraud_ratio) / target_fraud_ratio)
    legit_sample = legit.sample(n=min(n_legit_needed, len(legit)),
                                random_state=42)
    bank_c = pd.concat([fraud, legit_sample], ignore_index=True)
    bank_c = _ensure_min_rows(bank_c, clean_df, "Bank C")

    # ---- Bank D -- Card-Not-Present E-commerce --------------------
    # DeviceType was label-encoded; the most common value (desktop) is the mode
    dt_mode = clean_df["DeviceType"].mode()[0]
    mask_d = clean_df["DeviceType"] == dt_mode
    bank_d = clean_df[mask_d].copy()
    # Oversample card4 mode (visa = most common) and card6 mode (credit = most common)
    c4_mode = bank_d["card4"].mode()[0]
    c6_mode = bank_d["card6"].mode()[0]
    visa_credit = bank_d[(bank_d["card4"] == c4_mode) & (bank_d["card6"] == c6_mode)]
    bank_d = pd.concat([bank_d, visa_credit], ignore_index=True)
    # Skew toward high V-feature variance: keep rows where sum of abs(V cols) is
    # above median (these are the "noisier" transactions).
    v_cols = [c for c in bank_d.columns if c.startswith("V") and c[1:].isdigit()]
    if v_cols:
        v_energy = bank_d[v_cols].abs().sum(axis=1)
        bank_d = bank_d[v_energy >= v_energy.median()].copy()
    bank_d = _ensure_min_rows(bank_d, clean_df, "Bank D")

    banks = {
        "bank_a": bank_a,
        "bank_b": bank_b,
        "bank_c": bank_c,
        "bank_d": bank_d,
    }

    # ---- Cap each bank at MAX_ROWS (stratified by isFraud) --------
    for name in list(banks.keys()):
        bdf = banks[name]
        if len(bdf) > MAX_ROWS:
            fraud = bdf[bdf["isFraud"] == 1]
            legit = bdf[bdf["isFraud"] == 0]
            fraud_ratio = len(fraud) / len(bdf)
            n_fraud_keep = max(1, int(MAX_ROWS * fraud_ratio))
            n_legit_keep = MAX_ROWS - n_fraud_keep
            fraud_sample = fraud.sample(n=min(n_fraud_keep, len(fraud)),
                                        random_state=42)
            legit_sample = legit.sample(n=min(n_legit_keep, len(legit)),
                                        random_state=42)
            banks[name] = pd.concat([fraud_sample, legit_sample],
                                    ignore_index=True)
            print(f"       [{name}] capped to {len(banks[name]):,} rows "
                  f"(was {len(bdf):,})")

    for name, bdf in banks.items():
        print(f"       {name}: {len(bdf):>7,} rows | "
              f"fraud rate = {bdf['isFraud'].mean():.4f}")

    return banks


# ===================================================================
# SKEW REPORT
# ===================================================================
def save_skew_report(banks: dict[str, pd.DataFrame]) -> None:
    """Save JSON report of each bank's size, fraud rate, and key stats."""
    print("[2/3] Saving skew report ...")
    report: dict = {}
    for name, bdf in banks.items():
        report[name] = {
            "num_rows": int(len(bdf)),
            "fraud_rate": round(float(bdf["isFraud"].mean()), 5),
            "txn_amt_mean": round(float(bdf["TransactionAmt"].mean()), 4),
            "txn_amt_std": round(float(bdf["TransactionAmt"].std()), 4),
        }
    path = os.path.join(PROCESSED_DIR, "skew_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"       Saved -> {path}")


# ===================================================================
# VISUALISATION
# ===================================================================
def visualize_skew(bank_dfs: dict[str, pd.DataFrame]) -> None:
    """Generate comparison plots across banks and save to skew_plots/."""
    print("[3/3] Generating skew visualisations ...")
    sns.set_theme(style="whitegrid")

    names = list(bank_dfs.keys())
    colors = sns.color_palette("viridis", len(names))

    # -- 1. Fraud-rate bar chart ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = [bank_dfs[n]["isFraud"].mean() * 100 for n in names]
    ax.bar(names, rates, color=colors)
    ax.set_ylabel("Fraud Prevalence (%)")
    ax.set_title("Fraud Rate Across Bank Partitions")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.2, f"{v:.2f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fraud_rate_comparison.png"), dpi=150)
    plt.close(fig)

    # -- 2. TransactionAmt distribution ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (name, bdf) in zip(axes.flat, bank_dfs.items()):
        ax.hist(bdf["TransactionAmt"].clip(-4, 4), bins=60,
                color=colors[names.index(name)], alpha=0.8, edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel("TransactionAmt (scaled)")
    fig.suptitle("TransactionAmt Distribution per Bank", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "txn_amt_distribution.png"), dpi=150)
    plt.close(fig)

    # -- 3. Dataset size comparison --------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = [len(bank_dfs[n]) for n in names]
    ax.barh(names, sizes, color=colors)
    ax.set_xlabel("Number of Rows")
    ax.set_title("Dataset Size per Bank Partition")
    for i, v in enumerate(sizes):
        ax.text(v + 500, i, f"{v:,}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "dataset_sizes.png"), dpi=150)
    plt.close(fig)

    print(f"       Plots saved -> {PLOTS_DIR}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("  IEEE-CIS Fraud Detection -- Non-IID Skewing (Phase 2)")
    print("=" * 60)
    clean_path = os.path.join(PROCESSED_DIR, "ieee_clean.csv")
    if not os.path.isfile(clean_path):
        sys.exit(f"ERROR: {clean_path} not found. Run preprocess.py first.")

    print(f"Loading {clean_path} ...")
    df = pd.read_csv(clean_path)
    print(f"Loaded shape: {df.shape}")

    banks = partition_data(df)
    save_skew_report(banks)
    visualize_skew(banks)

    # Save each partition as CSV for the DB provisioner
    for name, bdf in banks.items():
        out = os.path.join(PROCESSED_DIR, f"{name}.csv")
        bdf.to_csv(out, index=False, float_format="%.4f")
        print(f"  Saved {name} -> {out}")

    print("\n[OK] Phase 2 complete.\n")


if __name__ == "__main__":
    main()
