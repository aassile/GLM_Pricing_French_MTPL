"""GLM model fitting for frequency and severity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_frequency(
    train_df: pd.DataFrame,
    features: list[str],
) -> sm.GLMResults:
    """Fit a Poisson GLM for claim frequency.

    Uses log(Exposure) as an offset so predictions represent expected claim
    counts per unit exposure.

    Parameters
    ----------
    train_df:
        Training DataFrame. Must contain 'ClaimNb', 'Exposure', and all columns
        named in *features*.
    features:
        List of predictor column names. Categorical columns are handled
        automatically by pandas / patsy.

    Returns
    -------
    statsmodels GLMResultsWrapper
        Fitted Poisson GLM result.
    """
    formula = "ClaimNb ~ " + " + ".join(features)
    model = smf.glm(
        formula=formula,
        data=train_df,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=np.log(train_df["Exposure"].clip(lower=1e-6)),
    )
    return model.fit()


def fit_severity(
    train_df: pd.DataFrame,
    features: list[str],
) -> sm.GLMResults:
    """Fit a Gamma GLM for average claim severity.

    Only policies with ClaimNb > 0 and ClaimAmountTotal > 0 are used.
    ClaimAmountTotal is capped at its 99th percentile to reduce outlier
    influence. The target is the average claim amount (ClaimAmountTotal /
    ClaimNb), weighted by ClaimNb.

    Parameters
    ----------
    train_df:
        Training DataFrame. Must contain 'ClaimNb', 'ClaimAmountTotal', and all
        columns named in *features*.
    features:
        List of predictor column names.

    Returns
    -------
    statsmodels GLMResultsWrapper
        Fitted Gamma GLM result.
    """
    df = train_df[(train_df["ClaimNb"] > 0) & (train_df["ClaimAmountTotal"] > 0)].copy()
    cap_99 = df["ClaimAmountTotal"].quantile(0.99)
    df["ClaimAmountCapped"] = df["ClaimAmountTotal"].clip(upper=cap_99)
    df["AvgClaim"] = df["ClaimAmountCapped"] / df["ClaimNb"]
    # Remove unused categorical levels that would make patsy singular
    for col in df.columns:
        if hasattr(df[col], "cat"):
            df[col] = df[col].cat.remove_unused_categories()

    formula = "AvgClaim ~ " + " + ".join(features)
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        freq_weights=df["ClaimNb"],
    )
    return model.fit()


def predict_pure_premium(
    freq_model: sm.GLMResults,
    sev_model: sm.GLMResults,
    df: pd.DataFrame,
) -> pd.Series:
    """Compute predicted pure premium per policy.

    Pure premium = (expected claim count per exposure unit) × (predicted avg
    claim severity). The Poisson model produces expected counts with exposure
    already baked in via the offset; dividing by exposure gives a rate, and
    multiplying back by exposure gives the expected aggregate claim.

    Parameters
    ----------
    freq_model:
        Fitted Poisson GLM (from :func:`fit_frequency`).
    sev_model:
        Fitted Gamma GLM (from :func:`fit_severity`).
    df:
        DataFrame for prediction. Must contain 'Exposure' and the feature
        columns used during training.

    Returns
    -------
    pd.Series
        Predicted pure premium per policy (EUR).
    """
    freq_pred = freq_model.predict(df, offset=np.log(df["Exposure"].clip(lower=1e-6)))
    sev_pred = sev_model.predict(df)
    # freq_pred is expected claim count; pure premium = freq_pred * sev_pred
    return pd.Series(freq_pred.values * sev_pred.values, index=df.index, name="PurePremium")
