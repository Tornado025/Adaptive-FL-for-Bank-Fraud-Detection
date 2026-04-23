"""
Master Pipeline Runner
========================
Runs Phase 1 -> Phase 2 -> Phase 3 in sequence.

Usage:
    py data/src/run_pipeline.py
"""

from preprocess import main as preprocess_main
from skewing_engine import main as skew_main
from db_provisioner import main as db_main


def main():
    print("\n" + "#" * 60)
    print("  FULL DATA PIPELINE -- Phases 1 -> 2 -> 3")
    print("#" * 60 + "\n")

    preprocess_main()
    skew_main()
    db_main()

    print("#" * 60)
    print("  ALL PHASES COMPLETE")
    print("#" * 60)
    print("\nOutputs:")
    print("  data/processed/ieee_clean.csv      -- cleaned dataset")
    print("  data/processed/bank_*.csv           -- per-bank partitions")
    print("  data/processed/skew_report.json     -- skew statistics")
    print("  data/processed/skew_plots/          -- visualisation PNGs")
    print("  data/databases/bank_*.db            -- SQLite databases")
    print()


if __name__ == "__main__":
    main()
