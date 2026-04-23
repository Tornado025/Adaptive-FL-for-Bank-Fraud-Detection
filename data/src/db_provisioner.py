"""
Phase 3 -- Database Provisioning
=================================
Reads the per-bank CSVs and writes each into an isolated SQLite database.

Usage:
    py data/src/db_provisioner.py
"""

import os
import sys
import sqlite3
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
DB_DIR = os.path.join(REPO_ROOT, "data", "databases")
os.makedirs(DB_DIR, exist_ok=True)

BANKS = ["bank_a", "bank_b", "bank_c", "bank_d"]


def provision_db(bank_name: str) -> None:
    """Write one bank CSV -> SQLite .db with an indexed transactions table."""
    csv_path = os.path.join(PROCESSED_DIR, f"{bank_name}.csv")
    db_path = os.path.join(DB_DIR, f"{bank_name}.db")

    if not os.path.isfile(csv_path):
        print(f"  [SKIP] {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Add a primary-key column
    df.insert(0, "txn_id", range(1, len(df) + 1))

    conn = sqlite3.connect(db_path)
    # Write in chunks with replace semantics
    df.to_sql("transactions", conn, if_exists="replace",
              index=False, chunksize=10_000)
    # Index on isFraud for fast batch sampling
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fraud ON transactions(isFraud)")
    conn.commit()

    # Verify row count
    cur = conn.execute("SELECT COUNT(*) FROM transactions")
    db_count = cur.fetchone()[0]
    conn.close()

    match = "[OK]" if db_count == len(df) else "[MISMATCH]"
    print(f"  {bank_name}: {db_count:>7,} rows  ->  {db_path}  {match}")


def main():
    print("=" * 60)
    print("  IEEE-CIS Fraud Detection -- DB Provisioning (Phase 3)")
    print("=" * 60)
    for bank in BANKS:
        provision_db(bank)
    print("\n[OK] Phase 3 complete.\n")


if __name__ == "__main__":
    main()
