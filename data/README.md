# Data

## Full Dataset

The full freMTPL2 datasets are available from [OpenML](https://www.openml.org/):

- **freMTPL2freq** (678,013 policies): https://www.openml.org/d/41214
- **freMTPL2sev** (26,639 claims): https://www.openml.org/d/41215

### Download via Python

```python
from sklearn.datasets import fetch_openml

freq = fetch_openml(data_id=41214, as_frame=True, parser="auto")
sev  = fetch_openml(data_id=41215, as_frame=True, parser="auto")

freq.frame.to_csv("data/freMTPL2freq.csv", index=False)
sev.frame.to_csv("data/freMTPL2sev.csv",  index=False)
```

Or via the `openml` package:
```python
import openml
freq_ds = openml.datasets.get_dataset(41214)
sev_ds  = openml.datasets.get_dataset(41215)
```

Full CSVs are **gitignored** (`data/*.csv`). Place them in `data/` before running notebooks 02–04.

## Sample

`data/sample/freMTPL_sample.csv` is a 5,000-row stratified sample (stratified by claim / no-claim),
merged with matching severity rows. It is committed to the repository and used by CI and
notebook quick-runs so no download is required.

## Schema

### freMTPL2freq columns
| Column | Description |
|--------|-------------|
| IDpol | Policy ID |
| ClaimNb | Number of claims |
| Exposure | Fraction of year at risk |
| Area | Area code (A–F) |
| VehPower | Vehicle power |
| VehAge | Vehicle age (years) |
| DrivAge | Driver age (years) |
| BonusMalus | Bonus-malus coefficient (50–350) |
| VehBrand | Vehicle brand |
| VehGas | Fuel type (Regular / Diesel) |
| Density | Population density at home address |
| Region | French region code |

### freMTPL2sev columns
| Column | Description |
|--------|-------------|
| IDpol | Policy ID |
| ClaimAmount | Individual claim amount (EUR) |
