"""Tests for glm_pricing.evaluation module."""

import numpy as np

from glm_pricing.evaluation import (
    calibration,
    gamma_deviance,
    gini,
    lift_chart,
    poisson_deviance,
)


def test_gini_perfect():
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    # Perfect prediction → gini = 1
    g = gini(y, y)
    assert abs(g - 1.0) < 0.01


def test_gini_random():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 1, 1000)
    y_pred = rng.uniform(0, 1, 1000)
    g = gini(y_true, y_pred)
    assert -1.0 <= g <= 1.0


def test_lift_chart_shape():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 1, 100)
    y_pred = rng.uniform(0, 1, 100)
    lc = lift_chart(y_true, y_pred, n_bins=10)
    assert "mean_pred" in lc.columns
    assert "mean_actual" in lc.columns
    assert len(lc) <= 10


def test_calibration_ratio():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    cal = calibration(y, y, n_bins=5)
    # ratio should be ~1 when pred == actual
    assert (cal["ratio"].dropna() - 1.0).abs().max() < 0.1


def test_poisson_deviance_zero():
    y = np.array([1.0, 2.0, 3.0])
    # Perfect prediction → deviance = 0
    d = poisson_deviance(y, y)
    assert abs(d) < 1e-8


def test_poisson_deviance_positive():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    d = poisson_deviance(y_true, y_pred)
    assert d > 0


def test_gamma_deviance_zero():
    y = np.array([100.0, 200.0, 300.0])
    d = gamma_deviance(y, y)
    assert abs(d) < 1e-8


def test_gamma_deviance_positive():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([150.0, 250.0, 350.0])
    d = gamma_deviance(y_true, y_pred)
    assert d > 0
