"""End-to-end GLM pricing pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from glm_pricing.data import load_freq, load_sev, merge_freq_sev
from glm_pricing.preprocessing import (
    cap_exposure,
    cap_claimnb,
    prepare_features,
    train_test_split_policy,
)
from glm_pricing.models import fit_frequency, fit_severity, predict_pure_premium
from glm_pricing.evaluation import poisson_deviance, gamma_deviance, gini

FREQ_FEATURES = ["BonusMalusCapped", "LogDensity", "VehPower", "VehAgeBin", "DrivAgeBin", "VehGas", "Area"]
SEV_FEATURES = ["BonusMalusCapped", "LogDensity", "VehPower", "VehGas", "Area"]


def run_pipeline(freq_path: str, sev_path: str) -> dict[str, Any]:
    """Load, preprocess, fit GLMs, and evaluate.

    Parameters
    ----------
    freq_path:
        Path to freMTPL2freq CSV (or sample CSV).
    sev_path:
        Path to freMTPL2sev CSV (or sample CSV). If the file is the merged
        sample CSV, pass freq_path again and sev columns will be absent —
        in that case severity fitting uses ClaimAmountTotal from freq file.

    Returns
    -------
    dict with keys:
        freq_model, sev_model, train, test, freq_pred_test, sev_pred_test,
        pure_premium_test, metrics
    """
    # --- Load ---
    freq = load_freq(freq_path)
    try:
        sev = load_sev(sev_path)
        df = merge_freq_sev(freq, sev)
    except Exception:
        # Fallback: freq_path is already merged (sample CSV)
        df = freq
        if "ClaimAmountTotal" not in df.columns:
            df["ClaimAmountTotal"] = 0.0

    # --- Preprocess ---
    df = cap_exposure(df)
    df = cap_claimnb(df)
    df = prepare_features(df)

    # --- Split ---
    train, test = train_test_split_policy(df)

    # --- Fit frequency ---
    freq_model = fit_frequency(train, FREQ_FEATURES)

    # --- Fit severity ---
    sev_model = fit_severity(train, SEV_FEATURES)

    # --- Predict ---
    freq_pred_test = freq_model.predict(test, offset=np.log(test["Exposure"].clip(lower=1e-6)))
    sev_pred_test = sev_model.predict(test)
    pure_premium_test = predict_pure_premium(freq_model, sev_model, test)

    # --- Evaluate ---
    pois_dev = poisson_deviance(test["ClaimNb"].values, freq_pred_test.values, exposure=test["Exposure"].values)
    sev_test = test[test["ClaimNb"] > 0].copy()
    sev_test["AvgClaim"] = sev_test["ClaimAmountTotal"] / sev_test["ClaimNb"].clip(lower=1)
    gam_dev = gamma_deviance(
        sev_test["AvgClaim"].values,
        sev_model.predict(sev_test).values,
        weights=sev_test["ClaimNb"].values,
    ) if len(sev_test) > 0 else float("nan")

    freq_gini = gini(test["ClaimNb"].values, freq_pred_test.values)

    metrics = {
        "poisson_deviance": pois_dev,
        "gamma_deviance": gam_dev,
        "freq_gini": freq_gini,
    }

    return {
        "freq_model": freq_model,
        "sev_model": sev_model,
        "train": train,
        "test": test,
        "freq_pred_test": freq_pred_test,
        "sev_pred_test": sev_pred_test,
        "pure_premium_test": pure_premium_test,
        "metrics": metrics,
    }
