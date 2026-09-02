"""Robust CSV reading for the LODHalal entity tables.

manufacturer.csv contains 815 rows with unescaped, unquoted commas embedded
in the free-text `nama_man` field (e.g. `356,SRIBOGA BAKERIES INTEGRA, PT`),
which breaks both pandas' C and python parser engines (verified directly).
Since the first column is always a numeric ID and the free-text column is
always last, splitting each data line on the *first* comma only resolves
this unambiguously. Applied uniformly to all two-column entity files for
consistency, even though food.csv/ingredient.csv/cert.csv currently parse
fine with pandas directly.
"""
from pathlib import Path

import pandas as pd


def read_two_col_csv_robust(path: Path, col_names: tuple[str, str]) -> pd.DataFrame:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    header, *data_lines = lines
    rows = []
    for line in data_lines:
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            raise ValueError(f"Unexpected line format in {path}: {line!r}")
        rows.append(parts)

    df = pd.DataFrame(rows, columns=list(col_names))
    df[col_names[0]] = pd.to_numeric(df[col_names[0]])
    return df
