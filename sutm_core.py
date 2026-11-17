"""
Scoring logic for the SUTM inspection data.

Kept free of Streamlit and of file access so that it can be unit tested on its
own: the dashboard reads the Excel files and calls these functions, and the
tests call them with small hand-written frames.

Terms are the ones used in the inspection sheets (Bahasa Indonesia):
BAIK good, CUKUP fair, KURANG poor, BURUK bad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The twelve inspected components of a medium-voltage (SUTM) pole.
COND_COLS = [
    "KONDISI TIANG", "KONDISI EKSTENSI", "KONDISI TRAVERS", "KONDISI GSW",
    "KONDISI PENYANGGA TIANG", "KONDISI PENAMPANG", "KONDISI JUMPER",
    "KONDISI PENGIKAT", "KONDISI ISOLATOR TUMPU", "KONDISI ISOLATOR AFSPAN",
    "KONDISI ARRESTER", "KONDISI FCO",
]

COMP_LABELS = {
    "KONDISI TIANG": "Tiang (Pole)",
    "KONDISI EKSTENSI": "Ekstensi",
    "KONDISI TRAVERS": "Travers",
    "KONDISI GSW": "GSW",
    "KONDISI PENYANGGA TIANG": "Penyangga",
    "KONDISI PENAMPANG": "Penampang",
    "KONDISI JUMPER": "Jumper",
    "KONDISI PENGIKAT": "Pengikat",
    "KONDISI ISOLATOR TUMPU": "Isolator Tumpu",
    "KONDISI ISOLATOR AFSPAN": "Isolator Afspan",
    "KONDISI ARRESTER": "Arrester",
    "KONDISI FCO": "FCO",
}

# Health index of the thermal sheet, where the condition is already graded.
HI_MAP = {"BAIK": 3, "CUKUP": 2, "KURANG": 1, "BURUK": 0}

# Free-text findings mapped to a health index. Checked worst first.
_BAD_WORDS = ("BURUK", "PECAH", "PUTUS", "KEROPOS", "FLASH", "BOCOR", "RANTAS", "RETAK")
_POOR_WORDS = ("KURANG", "KENDOR", "LEPAS", "MIRING", "LONGGAR", "BELUM", "RAMBAT")
_FAIR_WORDS = ("CUKUP", "LUMUT", "BERKARAT", "PARALON")
_EMPTY_WORDS = ("BLANK", "TIDAK ADA", "-", "")

# Risk class of a pole, from the number of bad and poor components.
RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
_RISK_BASE = {"LOW": 15, "MEDIUM": 40, "HIGH": 70, "CRITICAL": 90}

TEMP_COLS = ["SUHU FASA R", "SUHU FASA S", "SUHU FASA T"]


def hi_from_text(value) -> int:
    """Health index 0–3 for one inspector note: 3 good, 2 fair, 1 poor, 0 bad.

    An empty or absent note means the component was found in order, which is
    how the field sheets are filled in, so it scores 3.
    """
    if pd.isna(value) or str(value).strip().upper() in _EMPTY_WORDS:
        return 3
    text = str(value).upper()
    if any(word in text for word in _BAD_WORDS):
        return 0
    if any(word in text for word in _POOR_WORDS):
        return 1
    if any(word in text for word in _FAIR_WORDS):
        return 2
    return 3


def classify_risk(n_bad: int, n_poor: int) -> str:
    """Rule used by the maintenance team: two bad components make a pole critical."""
    if n_bad >= 2:
        return "CRITICAL"
    if n_bad == 1:
        return "HIGH"
    if n_poor >= 1:
        return "MEDIUM"
    return "LOW"


def risk_score(risk_class: str, n_bad: int, n_poor: int, jitter: float = 0.0) -> float:
    """Score 0–100 for ranking poles inside a class.

    The class sets the band; the defect counts separate poles within it. The
    optional jitter only breaks ties so that identical poles do not overlap in
    the charts, and it never moves a pole into another band.
    """
    score = _RISK_BASE[risk_class] + n_bad * 3 + n_poor * 1 + jitter
    return float(np.clip(score, 0, 100).round(1))


def add_health_index(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add an HI_ column per inspected component, plus the defect counts.

    Returns the frame and the names of the HI columns that were created.
    """
    out = df.copy()
    hi_cols = []
    for col in COND_COLS:
        if col in out.columns:
            name = f"HI_{col}"
            out[name] = out[col].apply(hi_from_text)
            hi_cols.append(name)
    if hi_cols:
        out["N_BURUK"] = (out[hi_cols] == 0).sum(axis=1)
        out["N_KURANG"] = (out[hi_cols] == 1).sum(axis=1)
    else:
        out["N_BURUK"] = 0
        out["N_KURANG"] = 0
    return out, hi_cols


def pole_id(df: pd.DataFrame) -> pd.Series:
    """Unique pole key: feeder name and pole number, both trimmed."""
    return df["PENYULANG"].astype(str).str.strip() + "_" + df["NO TIANG"].astype(str).str.strip()


def delta_temperature(df: pd.DataFrame) -> pd.Series:
    """Spread between the hottest and coldest phase, in kelvin.

    A large spread between R, S and T is the usual thermographic sign of a
    loose or corroded connection. Unreadable values are dropped, not guessed.
    """
    values = df.reindex(columns=TEMP_COLS).apply(pd.to_numeric, errors="coerce")
    return (values.max(axis=1) - values.min(axis=1)).fillna(0)


def worst_case_per_pole(t1: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One row per pole, taking its worst inspection, with risk class and score.

    A pole inspected several times is as bad as its worst record, which is the
    conservative reading a maintenance plan needs.
    """
    scored, _ = add_health_index(t1)
    scored["TIANG_ID"] = pole_id(scored)
    group_cols = [c for c in ["TIANG_ID", "ULP", "PENYULANG", "NO TIANG"] if c in scored.columns]
    extra = {
        name: (name, how)
        for name, how in (("SECTION", "first"), ("ZONA", "first"), ("PETUGAS", "last"), ("TANGGAL", "max"))
        if name in scored.columns
    }
    worst = scored.groupby(group_cols, dropna=False).agg(
        N_BURUK=("N_BURUK", "max"), N_KURANG=("N_KURANG", "max"), **extra
    ).reset_index()

    worst["RISK_CLASS"] = [classify_risk(b, k) for b, k in zip(worst["N_BURUK"], worst["N_KURANG"])]
    jitter = np.random.default_rng(seed).uniform(-2, 2, len(worst))
    worst["RISK_SCORE"] = [
        risk_score(c, b, k, j)
        for c, b, k, j in zip(worst["RISK_CLASS"], worst["N_BURUK"], worst["N_KURANG"], jitter)
    ]
    return worst


def thermal_summary(t2: pd.DataFrame) -> pd.DataFrame:
    """Worst thermal reading per pole: largest phase spread, lowest health index."""
    out = t2.copy()
    out["TIANG_ID"] = pole_id(out)
    out["NUM_HI_1"] = out["HI SUTM  - 1"].map(HI_MAP).fillna(3) if "HI SUTM  - 1" in out.columns else 3
    out["NUM_HI_2"] = out["HI SUTM - 2"].map(HI_MAP).fillna(3) if "HI SUTM - 2" in out.columns else 3
    out["DELTA_SUHU"] = delta_temperature(out)
    return out.groupby("TIANG_ID").agg(
        DELTA_SUHU=("DELTA_SUHU", "max"),
        NUM_HI_1=("NUM_HI_1", "min"),
        NUM_HI_2=("NUM_HI_2", "min"),
    ).reset_index()


def merge_thermal(worst: pd.DataFrame, t2_agg: pd.DataFrame, thermal_ids) -> pd.DataFrame:
    """Attach the thermal summary to the visual inspection, keeping every pole.

    Only a part of the network is surveyed with a thermal camera, so poles
    without a reading are marked rather than dropped or assumed healthy.
    """
    merged = worst.merge(t2_agg, on="TIANG_ID", how="left")
    merged["HAS_THERMAL"] = merged["TIANG_ID"].isin(set(thermal_ids))
    merged["DELTA_SUHU"] = merged["DELTA_SUHU"].fillna(0)
    return merged


def priority_table(worst: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    """The poles to attend to first: highest risk score, worst defects first."""
    return worst.sort_values(
        ["RISK_SCORE", "N_BURUK", "N_KURANG"], ascending=[False, False, False]
    ).head(top).reset_index(drop=True)
