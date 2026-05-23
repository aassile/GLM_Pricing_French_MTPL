# GLM Pricing – French Motor Third-Party Liability (freMTPL)

[![CI](https://github.com/aassile/GLM_Pricing_French_MTPL/actions/workflows/ci.yml/badge.svg)](https://github.com/aassile/GLM_Pricing_French_MTPL/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Objective

Build an actuarial GLM pricing model for French MTPL insurance using the
freMTPL2 dataset. The model estimates:

1. **Claim frequency** via a Poisson GLM with log link and exposure offset.
2. **Claim severity** via a Gamma GLM with log link, weighted by claim count.
3. **Pure premium** = E[Frequency] × E[Severity], the actuarially fair base rate per policy.

## Dataset

| Dataset | OpenML ID | Rows | Description |
|---------|-----------|------|-------------|
| freMTPL2freq | [41214](https://www.openml.org/d/41214) | 678,013 | One row per policy: ClaimNb, Exposure, and 10 risk features |
| freMTPL2sev | [41215](https://www.openml.org/d/41215) | 26,639 | One row per claim: IDpol, ClaimAmount |

See `data/README.md` for download instructions. A 5,000-row stratified sample
(`data/sample/freMTPL_sample.csv`) is committed for reproducible CI runs.

## Project Structure

```
GLM_Pricing_French_MTPL/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── README.md               # download instructions
│   └── sample/
│       └── freMTPL_sample.csv  # 5k-row sample (committed)
├── src/glm_pricing/
│   ├── __init__.py
│   ├── data.py                 # load_freq, load_sev, merge_freq_sev
│   ├── preprocessing.py        # feature engineering, train/test split
│   ├── models.py               # Poisson GLM, Gamma GLM, pure premium
│   ├── evaluation.py           # Gini, lift chart, calibration, deviances
│   └── pipeline.py             # end-to-end run_pipeline()
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory data analysis
│   ├── 02_frequency_glm.ipynb  # Poisson GLM for claim frequency
│   ├── 03_severity_glm.ipynb   # Gamma GLM for claim severity
│   └── 04_pure_premium.ipynb   # pure premium, lift chart, Gini
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
└── .github/workflows/ci.yml    # CI: Python 3.10 & 3.11, pytest
```

## Methodology

### 1 – Frequency Model (Poisson GLM)

Claims per policy follow a Poisson distribution. The expected count is modelled as:

```
log(E[ClaimNb]) = log(Exposure) + β₀ + β₁·BonusMalus + β₂·LogDensity + ...
```

The log-exposure offset ensures predictions are in claim-count units regardless
of policy duration.

### 2 – Severity Model (Gamma GLM)

Conditional on a claim, the average claim amount follows a Gamma distribution:

```
log(E[AvgClaim]) = β₀ + β₁·BonusMalus + β₂·LogDensity + β₃·VehPower + ...
```

Only policies with at least one claim are used. ClaimAmountTotal is capped at
its 99th percentile before fitting.

### 3 – Pure Premium

```
PurePremium = freq_pred × sev_pred
```

This is the expected loss per policy, the standard actuarial burning-cost estimate
used as a base rate before expense and profit loadings.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/aassile/GLM_Pricing_French_MTPL.git
cd GLM_Pricing_French_MTPL

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 3. (Optional) Download full data
python - <<'EOF'
from sklearn.datasets import fetch_openml
freq = fetch_openml(data_id=41214, as_frame=True, parser="auto")
sev  = fetch_openml(data_id=41215, as_frame=True, parser="auto")
freq.frame.to_csv("data/freMTPL2freq.csv", index=False)
sev.frame.to_csv("data/freMTPL2sev.csv",  index=False)
EOF

# 4. Run notebooks (sample data — no download needed)
jupyter lab notebooks/

# 5. Run pipeline programmatically
python - <<'EOF'
from glm_pricing.pipeline import run_pipeline
results = run_pipeline("data/sample/freMTPL_sample.csv",
                       "data/sample/freMTPL_sample.csv")
print(results["metrics"])
EOF
```

## Results (5k-row sample)

Metrics below are from the 80/20 train-test split on the committed sample.
Run on the full 678k dataset for production-quality estimates.

| Metric | Value |
|--------|-------|
| Poisson Deviance (frequency, test) | 0.364 |
| Gamma Deviance (severity, test) | 20.76 |
| Normalized Gini (pure premium) | 0.312 |

## Testing

```bash
pytest tests/ -v
```

19 tests covering data loading, preprocessing, model fitting, and evaluation metrics.

## CI

GitHub Actions runs on push/PR to `main`, testing Python 3.10 and 3.11.

## License

MIT – see [LICENSE](LICENSE).
