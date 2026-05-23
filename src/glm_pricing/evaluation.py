"""Model evaluation metrics for insurance GLMs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute normalized Gini coefficient.

    Measures the ranking power of predicted values relative to actual values.
    A value of 1 indicates perfect ordering; 0 indicates random ordering.

    Parameters
    ----------
    y_true:
        Observed values (e.g. claim amounts or counts).
    y_pred:
        Predicted values from the model.

    Returns
    -------
    float
        Normalized Gini coefficient in [-1, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    # Sort by predicted descending
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    lorenz = np.cumsum(y_true_sorted) / y_true.sum()
    gini_model = (lorenz.sum() - lorenz[-1] / 2 - 0.5 * (n - 1)) / (n - 1) if n > 1 else 0.0
    # Perfect Gini (sort by y_true descending)
    order_perf = np.argsort(-y_true)
    y_true_perf = y_true[order_perf]
    lorenz_perf = np.cumsum(y_true_perf) / y_true.sum()
    gini_perfect = (
        (lorenz_perf.sum() - lorenz_perf[-1] / 2 - 0.5 * (n - 1)) / (n - 1) if n > 1 else 1.0
    )
    return float(gini_model / gini_perfect) if gini_perfect != 0 else 0.0


def lift_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute a lift chart by decile.

    Policies are sorted by predicted value into *n_bins* equal-sized buckets.
    Mean predicted and mean actual values per bucket are returned.

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted values.
    n_bins:
        Number of equal-sized buckets (default 10).

    Returns
    -------
    pd.DataFrame
        Columns: bin, mean_pred, mean_actual.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["bin"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")
    result = (
        df.groupby("bin", observed=True)
        .agg(mean_pred=("y_pred", "mean"), mean_actual=("y_true", "mean"))
        .reset_index()
    )
    return result


def calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration by decile (ratio actual / predicted per bucket).

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted values.
    n_bins:
        Number of equal-sized buckets (default 10).

    Returns
    -------
    pd.DataFrame
        Columns: bin, mean_pred, mean_actual, ratio (actual/predicted).
    """
    df = lift_chart(y_true, y_pred, n_bins=n_bins)
    df["ratio"] = df["mean_actual"] / df["mean_pred"].replace(0, np.nan)
    return df


def poisson_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exposure: np.ndarray | None = None,
) -> float:
    """Compute mean Poisson deviance.

    Parameters
    ----------
    y_true:
        Observed claim counts.
    y_pred:
        Predicted claim counts (expected).
    exposure:
        Optional exposure weights.

    Returns
    -------
    float
        (Weighted) mean Poisson deviance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(min=1e-10)
    mask = y_true > 0
    dev = np.zeros_like(y_true)
    dev[mask] = 2 * (
        y_true[mask] * np.log(y_true[mask] / y_pred[mask]) - (y_true[mask] - y_pred[mask])
    )
    dev[~mask] = 2 * y_pred[~mask]
    if exposure is not None:
        w = np.asarray(exposure, dtype=float)
        return float(np.average(dev, weights=w))
    return float(dev.mean())


def gamma_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Compute mean Gamma deviance.

    Parameters
    ----------
    y_true:
        Observed severity values (must be > 0).
    y_pred:
        Predicted severity values (must be > 0).
    weights:
        Optional frequency weights.

    Returns
    -------
    float
        (Weighted) mean Gamma deviance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(min=1e-10)
    y_true = y_true.clip(min=1e-10)
    dev = 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        return float(np.average(dev, weights=w))
    return float(dev.mean())
