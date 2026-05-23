"""Feature engineering and preprocessing for freMTPL data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def cap_exposure(df: pd.DataFrame, cap: float = 1.0) -> pd.DataFrame:
    """Clip Exposure to [0, cap].

    Parameters
    ----------
    df:
        Input DataFrame containing an 'Exposure' column.
    cap:
        Maximum allowed exposure value (default 1.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with Exposure clipped in-place (copy).
    """
    df = df.copy()
    df["Exposure"] = df["Exposure"].clip(lower=0.0, upper=cap)
    return df


def cap_claimnb(df: pd.DataFrame, cap: int = 4) -> pd.DataFrame:
    """Clip ClaimNb to [0, cap].

    Parameters
    ----------
    df:
        Input DataFrame containing a 'ClaimNb' column.
    cap:
        Maximum allowed claim count (default 4).

    Returns
    -------
    pd.DataFrame
        DataFrame with ClaimNb clipped (copy).
    """
    df = df.copy()
    df["ClaimNb"] = df["ClaimNb"].clip(lower=0, upper=cap)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations.

    - Bins VehAge into groups [0,1), [1,10), [10,+∞)
    - Bins DrivAge into groups [18,21), [21,26), ..., [71,+∞)
    - Clips BonusMalus to [50, 150]
    - Log-transforms Density (log1p)

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        VehAgeBin, DrivAgeBin, BonusMalusCapped, LogDensity.
    """
    df = df.copy()

    # Vehicle age bins
    veh_age_bins = [0, 1, 10, np.inf]
    veh_age_labels = ["new", "young", "old"]
    df["VehAgeBin"] = pd.cut(
        df["VehAge"],
        bins=veh_age_bins,
        labels=veh_age_labels,
        right=False,
        include_lowest=True,
    )

    # Driver age bins
    drv_age_bins = [18, 21, 26, 31, 41, 51, 71, np.inf]
    drv_age_labels = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
    df["DrivAgeBin"] = pd.cut(
        df["DrivAge"],
        bins=drv_age_bins,
        labels=drv_age_labels,
        right=False,
        include_lowest=True,
    )

    # Bonus-malus cap
    df["BonusMalusCapped"] = df["BonusMalus"].clip(lower=50, upper=150)

    # Log density
    df["LogDensity"] = np.log1p(df["Density"])

    return df


def train_test_split_policy(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and test sets by policy row.

    Parameters
    ----------
    df:
        Full dataset.
    test_size:
        Fraction for test (default 0.2).
    random_state:
        Reproducibility seed (default 42).

    Returns
    -------
    (train, test) : tuple of DataFrames
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)
