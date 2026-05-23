"""GLM-based insurance pricing for French MTPL data."""

__version__ = "0.1.0"

from glm_pricing.data import load_freq, load_sev, merge_freq_sev
from glm_pricing.evaluation import (
    calibration,
    gamma_deviance,
    gini,
    lift_chart,
    poisson_deviance,
)
from glm_pricing.models import fit_frequency, fit_severity, predict_pure_premium
from glm_pricing.pipeline import run_pipeline
from glm_pricing.preprocessing import (
    cap_claimnb,
    cap_exposure,
    prepare_features,
    train_test_split_policy,
)

__all__ = [
    "load_freq",
    "load_sev",
    "merge_freq_sev",
    "cap_exposure",
    "cap_claimnb",
    "prepare_features",
    "train_test_split_policy",
    "fit_frequency",
    "fit_severity",
    "predict_pure_premium",
    "gini",
    "lift_chart",
    "calibration",
    "poisson_deviance",
    "gamma_deviance",
    "run_pipeline",
]
