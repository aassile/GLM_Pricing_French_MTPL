"""Tests for glm_pricing.data module."""

import io

import pandas as pd

from glm_pricing.data import load_freq, load_sev, merge_freq_sev

FREQ_CSV = """IDpol,ClaimNb,Exposure,Area,VehPower,VehAge,DrivAge,BonusMalus,VehBrand,VehGas,Density,Region
1,0,0.5,A,5,3,35,75,B1,Regular,100,R11
2,1,1.0,B,7,10,50,100,B2,Diesel,200,R22
3,2,0.8,C,4,1,28,110,B3,Regular,500,R33
"""

SEV_CSV = """IDpol,ClaimAmount
2,3000
3,1500
3,2500
"""


def make_freq_df():
    return pd.read_csv(io.StringIO(FREQ_CSV), dtype={"IDpol": str})


def make_sev_df():
    return pd.read_csv(io.StringIO(SEV_CSV), dtype={"IDpol": str})


def test_load_freq(tmp_path):
    p = tmp_path / "freq.csv"
    p.write_text(FREQ_CSV)
    df = load_freq(str(p))
    assert len(df) == 3
    assert "ClaimNb" in df.columns
    assert df["ClaimNb"].dtype in [float, int] or pd.api.types.is_numeric_dtype(df["ClaimNb"])


def test_load_sev(tmp_path):
    p = tmp_path / "sev.csv"
    p.write_text(SEV_CSV)
    df = load_sev(str(p))
    assert len(df) == 3
    assert "ClaimAmount" in df.columns


def test_merge_freq_sev():
    freq = make_freq_df()
    sev = make_sev_df()
    merged = merge_freq_sev(freq, sev)
    assert "ClaimAmountTotal" in merged.columns
    assert len(merged) == len(freq)
    # Policy 1 has no claims → 0
    row1 = merged[merged["IDpol"] == "1"]
    assert row1["ClaimAmountTotal"].iloc[0] == 0.0
    # Policy 2 has one claim of 3000
    row2 = merged[merged["IDpol"] == "2"]
    assert row2["ClaimAmountTotal"].iloc[0] == 3000.0
    # Policy 3 has two claims summing to 4000
    row3 = merged[merged["IDpol"] == "3"]
    assert row3["ClaimAmountTotal"].iloc[0] == 4000.0
