# Track 1 — Data Engineering & Simulation

**Role: The Data Architect**

You are the foundation of this entire project. The quality, realism, and correct skewing of the data you produce determines whether the federated learning experiment is scientifically valid. You do **not** build neural networks. You do **not** touch the server. Your job is to take raw transaction data, clean it, and deliberately break it into distinct, realistic "bank profiles" that simulate how real-world bank data is never uniformly distributed.

---

## Your Mission

The fundamental challenge of federated learning is **Non-IID data** (non-independent and identically distributed). If every bank had the same data distribution, simple averaging would work fine. Your job is to make sure it doesn't — by engineering datasets that are genuinely different across banks. This forces Track 3 to build smarter aggregation algorithms to handle the mismatch.

---

## File Structure

```
track1_data_engineering/
├── data/
│   ├── raw/                  # The original unmodified IEEE-CIS CSV files go here
│   ├── processed/            # Cleaned, normalized, encoded outputs go here
│   └── databases/            # Final per-bank SQLite files: bank_a.db, bank_b.db, etc.
├── src/
│   ├── preprocess.py         # Phase 1: Cleaning, normalization, encoding
│   ├── skewing_engine.py     # Phase 2: The Non-IID partitioning logic
│   ├── db_provisioner.py     # Phase 3: SQLite database creation and population
│   ├── dataloaders.py        # Phase 3: PyTorch DataLoader definitions
│   └── stress_test.py        # Phase 4: Drop-out simulation and imbalanced batch generation
├── tests/
│   ├── test_preprocess.py
│   ├── test_skewing.py
│   └── test_dataloaders.py
└── requirements.txt
```

---

## Phase 1 — Data Acquisition & Preprocessing

**File: `src/preprocess.py`**

Download the IEEE-CIS Fraud Detection dataset from Kaggle. The dataset consists of two files: `train_transaction.csv` and `train_identity.csv`. Merge them on `TransactionID`.

Your preprocessing tasks:

**Missing Values**
- `TransactionAmt`: Impute with median per `ProductCD` group.
- `card` columns (`card1`–`card6`): Impute with mode per card type.
- `addr1`, `addr2`: Impute with -1 (treat unknown as a distinct category).
- `dist1`, `dist2`: Impute with 0.
- `P_emaildomain`, `R_emaildomain`: Impute with `"unknown"`.
- `id_*` identity columns: These are sparse by design. Impute numerical ones with -1, categorical with `"unknown"`.
- `V1`–`V339` Vesta features: Impute with column median. Drop any column where >80% of values are missing.

**Normalization**
- `TransactionAmt`: Apply log1p transformation first (`log(1 + x)`), then StandardScaler. This handles the heavy right skew in transaction amounts.
- All other continuous numerical features: StandardScaler (zero mean, unit variance).
- Do **not** normalize binary flags or already-encoded categoricals.

**Categorical Encoding**
- `ProductCD`, `card4`, `card6`, `M1`–`M9`: Label encode (they are ordinal or low cardinality).
- `P_emaildomain`, `R_emaildomain`: Extract domain suffix (e.g., `gmail`, `yahoo`, `hotmail`, `other`) and label encode into ~10 buckets.
- `DeviceType`, `DeviceInfo`: Label encode after bucketing rare values into `"other"` (keep only top 20 values).

**Output:** Save the cleaned, fully numerical DataFrame to `data/processed/ieee_clean.csv`.

```python
# Expected output shape: (590,540 rows, ~200 columns)
# isFraud column must be preserved as the label
# No NaN values should remain in the output
```

---

## Phase 2 — The Non-IID Skewing Engine

**File: `src/skewing_engine.py`**

This is the most intellectually important file in your track. You will partition `ieee_clean.csv` into **4 distinct bank profiles**. The skewing must be realistic — based on actual feature distributions, not random slicing.

**Bank Profiles to Implement:**

| Bank | Profile Name | Skewing Strategy |
|------|-------------|------------------|
| A | Domestic Retail | Filter `addr2 == 87` (US domestic). Oversample `ProductCD` values C and W (consumer/wallet). Cap `TransactionAmt` at the 75th percentile. |
| B | International Corporate | Filter transactions where `addr2 != 87`. Keep only high-value transactions (top 40% by `TransactionAmt`). Oversample `ProductCD == H` (hotel/travel). |
| C | High-Fraud Region | Deliberately oversample the minority fraud class to 15% prevalence (vs. 3.5% in the original). Represents a bank in a high-risk region. |
| D | Card-Not-Present E-commerce | Filter on `DeviceType == "desktop"`. Oversample `card4 == "visa"` and `card6 == "credit"`. Skew toward high V-feature variance. |

**Implementation Requirements:**
- Each bank must receive at least 80,000 rows after skewing.
- Label distribution must be logged and saved to `data/processed/skew_report.json`.
- Overlapping rows between banks are acceptable and realistic (the same fraud pattern can appear at multiple banks).
- Provide a `visualize_skew()` function that outputs a matplotlib figure comparing feature distributions across banks — this is your proof of work.

```python
def partition_data(clean_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Returns {'bank_a': df_a, 'bank_b': df_b, 'bank_c': df_c, 'bank_d': df_d}"""
    ...

def visualize_skew(bank_dfs: dict) -> None:
    """Save distribution comparison plots to data/processed/skew_plots/"""
    ...
```

---

## Phase 3 — Database Provisioning

**File: `src/db_provisioner.py`**

Take each skewed DataFrame and write it into an isolated SQLite database.

```python
# Output: data/databases/bank_a.db, bank_b.db, bank_c.db, bank_d.db
# Each DB has a single table: `transactions`
# Schema: all feature columns + isFraud label + a primary key `txn_id`
```

- Use `sqlite3` directly or `SQLAlchemy`. Do not use Pandas `to_sql` with default settings — set `chunksize=10000` and `if_exists='replace'`.
- Add an index on `isFraud` for faster batch sampling during training.
- Verify each database after writing: row count must match the source DataFrame.

**File: `src/dataloaders.py`**

Write PyTorch DataLoader classes that Track 2 will use directly. Coordinate with Track 2 on the exact interface.

```python
class BankTransactionDataset(Dataset):
    """
    Reads directly from a SQLite .db file.
    Returns (features_tensor, label_tensor) pairs.
    """
    def __init__(self, db_path: str, split: str = "train", train_ratio: float = 0.8):
        # split: "train" or "val"
        # Must handle the train/val split deterministically (use a fixed random seed)
        ...

def get_dataloader(db_path: str, batch_size: int = 256, split: str = "train") -> DataLoader:
    """Factory function. This is the primary interface for Track 2."""
    ...
```

**Critical contract with Track 2:**
- Feature tensor dtype: `torch.float32`
- Label tensor dtype: `torch.float32` (single value, 0.0 or 1.0 — BCELoss compatible)
- Feature dimension: must be consistent across all 4 banks (same number of columns)
- Document the feature dimension in a constant: `FEATURE_DIM = N`

---

## Phase 4 — Integration & Edge Cases

**File: `src/stress_test.py`**

Work with Track 2 to make sure the data pipeline is robust under adversarial conditions. Implement the following simulation modes:

**1. Data Drop-Out Simulation**
```python
def simulate_dropout(db_path: str, drop_fraction: float = 0.3) -> DataLoader:
    """
    Returns a DataLoader where `drop_fraction` of batches are randomly empty.
    Simulates a bank node going offline mid-round.
    Track 2's training loop must handle StopIteration gracefully.
    """
```

**2. Imbalanced Batch Injection**
```python
def simulate_imbalanced_batches(db_path: str, fraud_ratio: float = 0.001) -> DataLoader:
    """
    Forces batches where fraud cases are extremely rare (1 in 1000).
    Uses WeightedRandomSampler. Tests whether Track 2's loss function
    handles severe class imbalance without gradient vanishing.
    """
```

**3. Feature Corruption**
```python
def simulate_feature_corruption(batch: tuple, corrupt_fraction: float = 0.1) -> tuple:
    """
    Randomly zeroes out corrupt_fraction of features in a batch tensor.
    Simulates sensor/ETL failures. Pass as a collate_fn to DataLoader.
    """
```

Run all three scenarios and produce a `stress_test_report.md` documenting which conditions caused training failures in Track 2, and what fixes were applied.

---

## Integration Checklist

Before handing off to Track 2, verify all of the following:

- [ ] `ieee_clean.csv` has zero NaN values (`df.isna().sum().sum() == 0`)
- [ ] All 4 `.db` files exist and are readable
- [ ] `get_dataloader()` returns correct tensor shapes — run `next(iter(loader))` and check
- [ ] `FEATURE_DIM` constant is defined and shared with Track 2
- [ ] Label tensors are float32 (not long/int) — required for BCELoss
- [ ] Train/val split is deterministic (same seed = same split every run)
- [ ] `skew_report.json` documents each bank's fraud prevalence and feature distributions
- [ ] Stress test scenarios are runnable by Track 2 with a single import

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
torch>=2.0
sqlite3  # stdlib
matplotlib>=3.7
seaborn>=0.12
kaggle  # for dataset download
```

Install: `pip install -r requirements.txt`

---

## Notes on Data Access

The IEEE-CIS dataset requires a Kaggle account. Run:
```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

Do **not** commit the raw data files. They are in `.gitignore`. Only commit the processed outputs and database files if their size is under 100MB. For larger databases, document the generation procedure so teammates can reproduce them locally.
