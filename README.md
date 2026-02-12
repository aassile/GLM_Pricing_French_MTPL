# GLM Insurance Pricing Project

End-to-end Generalized Linear Model (GLM) implementation for auto insurance ratemaking using the French Motor Third-Party Liability (freMTPL) dataset.

## Objective
Predict claim **frequency** (Poisson GLM) and **severity** (Gamma GLM), then compute pure premium to support fair pricing.

## Dataset
- **freMTPL2freq**: ~678k policies (exposure, claim counts, risk factors)
- **freMTPL2sev**: ~26k claims (amounts)
- Source: [Kaggle - French Motor Third-Party Liability Claims](https://www.kaggle.com/datasets/floser/french-motor-third-party-liability-claims) (or original CASdataset)

## Project Structure
- `notebooks/` → Exploratory analysis & modeling
- `src/` → Reusable code
- `data/` → Data (samples only; full data linked externally)
- `figures/` → Visualizations

## Key Technologies
- Python 3.10+
- statsmodels (GLM)
- pandas, numpy, matplotlib/seaborn
- scikit-learn (optional comparisons)

## How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/glm-insurance-pricing.git
   cd glm-insurance-pricing