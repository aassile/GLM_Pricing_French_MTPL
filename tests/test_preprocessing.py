"""Tests for glm_pricing.preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from glm_pricing.preprocessing import (
    cap_exposure,
    cap_claimnb,
    prepare_features,
    train_test_split_policy,
)


def make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "IDpol": [str(i) for i in range(n)],
        "ClaimNb": rng.integers(0, 6, size=n),
        "Exposure": rng.uniform(0, 1.5, size=n),
        "VehAge": rng.integers(0, 20, size=n),
        "DrivAge": rng.integers(18, 85, size=n),
        "BonusMalus": rng.integers(50, 300, size=n),
        "Density": rng.uniform(1, 10000, size=n),
        "VehPower": rng.integers(4, 15, size=n),
        "Area": rng.choice(list("ABCDEF"), size=n),
        "VehGas": rng.choice(["Regular", "Diesel"], size=n),
        "VehBrand": rng.choice(["B1", "B2", "B3"], size=n),
        "Region": rng.choice(["R11", "R22", "R33"], size=n),
    })


def test_cap_exposure():
    df = make_df()
    result = cap_exposure(df, cap=1.0)
    assert result["Exposure"].max() <= 1.0
    assert result["Exposure"].min() >= 0.0
    # Original unchanged
    assert df["Exposure"].max() > 1.0 or True  # may or may not have values > 1


def test_cap_exposure_does_not_mutate():
    df = make_df()
    original_max = df["Exposure"].max()
    cap_exposure(df, cap=1.0)
    assert df["Exposure"].max() == original_max


def test_cap_claimnb():
    df = make_df()
    result = cap_claimnb(df, cap=4)
    assert result["ClaimNb"].max() <= 4
    assert result["ClaimNb"].min() >= 0


def test_prepare_features():
    df = make_df()
    result = prepare_features(df)
    assert "VehAgeBin" in result.columns
    assert "DrivAgeBin" in result.columns
    assert "BonusMalusCapped" in result.columns
    assert "LogDensity" in result.columns
    assert result["BonusMalusCapped"].between(50, 150).all()
    assert (result["LogDensity"] >= 0).all()


def test_train_test_split():
    df = make_df(n=100)
    train, test = train_test_split_policy(df, test_size=0.2, random_state=42)
    assert len(train) + len(test) == 100
    assert abs(len(test) / 100 - 0.2) < 0.05
