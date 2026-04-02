#!/usr/bin/env python3
"""
Visualize two time series datasets:
- xd from data_xd.csv
- xc from data_xc.csv

Each CSV is expected to contain columns like:
t, k, x1, x2, ..., x8

This script:
1. Loads both CSV files
2. Detects all state columns starting with 'x'
3. Creates one subplot per state variable
4. Overlays xd and xc for easy comparison
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p)
    return df


def get_state_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("x")]
    if not cols:
        raise ValueError("No state columns found. Expected columns like x1, x2, ..., x8")
    return cols


def main():
    base = Path(__file__).parent
    xd_file = base/"data_xd.csv"
    xc_file = base/"data_xc.csv"

    try:
        df_xd = load_csv(xd_file)
        df_xc = load_csv(xc_file)
    except Exception as e:
        print(f"Error loading CSV files: {e}", file=sys.stderr)
        sys.exit(1)

    for name, df in [("xd", df_xd), ("xc", df_xc)]:
        if "t" not in df.columns:
            print(f"Error: column 't' not found in {name}", file=sys.stderr)
            sys.exit(1)

    try:
        xd_states = get_state_columns(df_xd)
        xc_states = get_state_columns(df_xc)
    except Exception as e:
        print(f"Error detecting state columns: {e}", file=sys.stderr)
        sys.exit(1)

    common_states = [c for c in xd_states if c in xc_states]
    if not common_states:
        print("Error: no common x-columns found between data_xd.csv and data_xc.csv", file=sys.stderr)
        sys.exit(1)

    n = len(common_states)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)

    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, common_states):
        ax.plot(df_xd["t"], df_xd[col], label=f"xd: {col}")
        ax.plot(df_xc["t"], df_xc[col], label=f"xc: {col}")
        ax.set_title(f"{col} vs time")
        ax.set_xlabel("t")
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend()

    fig.suptitle("Comparison of xd and xc series", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()