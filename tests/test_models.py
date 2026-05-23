"""Tests for glm_pricing.models module."""

import numpy as np
import pandas as pd

from glm_pricing.models import fit_frequency, fit_severity, predict_pure_premium


def make_training_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ClaimNb": rng.integers(0, 3, size=n),
            "Exposure": rng.uniform(0.1, 1.0, size=n),
            "VehPower": rng.integers(4, 12, size=n).astype(float),
            "BonusMalusCapped": rng.uniform(50, 150, size=n),
            "LogDensity": rng.uniform(2, 8, size=n),
            "VehGas": rng.choice(["Regular", "Diesel"], size=n),
            "Area": rng.choice(["A", "B", "C"], size=n),
            "VehAgeBin": pd.Categorical(
                rng.choice(["new", "young", "old"], size=n),
                categories=["new", "young", "old"],
            ),
            "DrivAgeBin": pd.Categorical(
                rng.choice(["21-25", "26-30", "41-50"], size=n),
                categories=["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"],
            ),
        }
    )
    # Add ClaimAmountTotal for severity
    df["ClaimAmountTotal"] = np.where(
        df["ClaimNb"] > 0,
        rng.uniform(500, 5000, size=n) * df["ClaimNb"],
        0.0,
    )
    return df


FEATURES = ["BonusMalusCapped", "LogDensity", "VehPower", "VehGas", "Area"]


def test_fit_frequency():
    df = make_training_df()
    model = fit_frequency(df, FEATURES)
    assert model is not None
    preds = model.predict(df, offset=np.log(df["Exposure"].clip(lower=1e-6)))
    assert len(preds) == len(df)
    assert (preds > 0).all()


def test_fit_severity():
    df = make_training_df()
    model = fit_severity(df, FEATURES)
    assert model is not None
    preds = model.predict(df)
    assert (preds > 0).all()


def test_predict_pure_premium():
    df = make_training_df()
    freq_model = fit_frequency(df, FEATURES)
    sev_model = fit_severity(df, FEATURES)
    pp = predict_pure_premium(freq_model, sev_model, df)
    assert len(pp) == len(df)
    assert (pp >= 0).all()
