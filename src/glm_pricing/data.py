"""Data loading utilities for freMTPL datasets."""

from __future__ import annotations

import pandas as pd


def load_freq(path: str) -> pd.DataFrame:
    """Load freMTPL2freq CSV.

    Parameters
    ----------
    path:
        Path to the freMTPL2freq CSV file.

    Returns
    -------
    pd.DataFrame
        Raw frequency table with columns IDpol, ClaimNb, Exposure, etc.
    """
    df = pd.read_csv(path)
    # Ensure IDpol is string for merging
    if "IDpol" in df.columns:
        df["IDpol"] = df["IDpol"].astype(str)
    # Ensure numeric columns are correct type
    numeric_cols = ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", "ClaimAmountTotal"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_sev(path: str) -> pd.DataFrame:
    """Load freMTPL2sev CSV.

    Parameters
    ----------
    path:
        Path to the freMTPL2sev CSV file.

    Returns
    -------
    pd.DataFrame
        Severity table with columns IDpol, ClaimAmount.
    """
    df = pd.read_csv(path)
    if "IDpol" in df.columns:
        df["IDpol"] = df["IDpol"].astype(str)
    if "ClaimAmount" in df.columns:
        df["ClaimAmount"] = pd.to_numeric(df["ClaimAmount"], errors="coerce")
    return df


def merge_freq_sev(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate severity by policy and left-merge with frequency data.

    Parameters
    ----------
    freq:
        Frequency DataFrame (one row per policy).
    sev:
        Severity DataFrame (one row per claim).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with ClaimAmountTotal column appended.
        Policies with no claims have ClaimAmountTotal = 0.
    """
    sev_agg = (
        sev.groupby("IDpol", as_index=False)["ClaimAmount"]
        .sum()
        .rename(columns={"ClaimAmount": "ClaimAmountTotal"})
    )
    combined = freq.merge(sev_agg, on="IDpol", how="left")
    combined["ClaimAmountTotal"] = combined["ClaimAmountTotal"].fillna(0.0)
    return combined
