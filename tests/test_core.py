"""Unit tests for the SUTM scoring logic.

The inspection workbooks are confidential and are not in the repository, so
every test builds its own small frame in the shape of the real sheets.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sutm_core import (  # noqa: E402
    COND_COLS,
    add_health_index,
    classify_risk,
    delta_temperature,
    hi_from_text,
    merge_thermal,
    pole_id,
    priority_table,
    risk_score,
    thermal_summary,
    worst_case_per_pole,
)


# ─── Condition text to health index ──────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("BAIK", 3),
    ("", 3),
    ("BLANK", 3),
    ("TIDAK ADA", 3),
    ("-", 3),
    (None, 3),
    ("CUKUP", 2),
    ("BERLUMUT / LUMUT", 2),
    ("BERKARAT RINGAN", 2),
    ("KURANG", 1),
    ("PENGIKAT KENDOR", 1),
    ("ISOLATOR MIRING", 1),
    ("BURUK", 0),
    ("ISOLATOR PECAH", 0),
    ("GSW PUTUS", 0),
    ("TIANG KEROPOS", 0),
    ("ADA FLASH OVER", 0),
])
def test_hi_from_text(text, expected):
    assert hi_from_text(text) == expected


def test_worst_finding_wins_when_a_note_has_several_words():
    # "CUKUP" would score 2 on its own, but a cracked insulator is a defect.
    assert hi_from_text("CUKUP TAPI RETAK") == 0
    assert hi_from_text("BERKARAT DAN KENDOR") == 1


def test_case_and_whitespace_do_not_matter():
    assert hi_from_text("  buruk  ") == hi_from_text("BURUK") == 0


# ─── Risk classification ─────────────────────────────────────────────────────

@pytest.mark.parametrize("n_bad, n_poor, expected", [
    (0, 0, "LOW"),
    (0, 1, "MEDIUM"),
    (0, 7, "MEDIUM"),
    (1, 0, "HIGH"),
    (1, 5, "HIGH"),
    (2, 0, "CRITICAL"),
    (9, 3, "CRITICAL"),
])
def test_classify_risk(n_bad, n_poor, expected):
    assert classify_risk(n_bad, n_poor) == expected


def test_risk_score_stays_inside_its_band():
    # Jitter separates ties in a chart; it must not promote a pole.
    assert risk_score("LOW", 0, 0, jitter=1.9) < risk_score("MEDIUM", 0, 1, jitter=-1.9)
    assert risk_score("HIGH", 1, 0, jitter=1.9) < risk_score("CRITICAL", 2, 0, jitter=-1.9)


def test_risk_score_rises_with_defects_and_is_bounded():
    assert risk_score("CRITICAL", 2, 0) < risk_score("CRITICAL", 5, 0)
    assert risk_score("CRITICAL", 12, 12) <= 100
    assert risk_score("LOW", 0, 0, jitter=-50) >= 0


# ─── Frame level ─────────────────────────────────────────────────────────────

def make_t1():
    """Three inspections of two poles, in the shape of the SUTMT1 sheet."""
    blank = {c: "BAIK" for c in COND_COLS}
    rows = [
        {**blank, "ULP": "A", "PENYULANG": "FDR1", "NO TIANG": "10",
         "KONDISI TIANG": "KEROPOS", "KONDISI GSW": "PUTUS", "SECTION": "S1", "ZONA": "Z1"},
        # The same pole, seen earlier in better shape: the worst record must win.
        {**blank, "ULP": "A", "PENYULANG": "FDR1", "NO TIANG": "10",
         "KONDISI TIANG": "BAIK", "SECTION": "S1", "ZONA": "Z1"},
        {**blank, "ULP": "A", "PENYULANG": "FDR2", "NO TIANG": "7",
         "KONDISI JUMPER": "KENDOR", "SECTION": "S2", "ZONA": "Z1"},
    ]
    return pd.DataFrame(rows)


def test_add_health_index_counts_defects():
    scored, hi_cols = add_health_index(make_t1())
    assert len(hi_cols) == len(COND_COLS)
    assert list(scored["N_BURUK"]) == [2, 0, 0]
    assert list(scored["N_KURANG"]) == [0, 0, 1]


def test_pole_id_trims_and_joins():
    df = pd.DataFrame({"PENYULANG": [" FDR1 "], "NO TIANG": [" 10 "]})
    assert pole_id(df).iloc[0] == "FDR1_10"


def test_worst_case_keeps_one_row_per_pole_with_its_worst_state():
    worst = worst_case_per_pole(make_t1())
    assert len(worst) == 2
    bad = worst.set_index("TIANG_ID").loc["FDR1_10"]
    assert bad["N_BURUK"] == 2
    assert bad["RISK_CLASS"] == "CRITICAL"
    assert worst.set_index("TIANG_ID").loc["FDR2_7", "RISK_CLASS"] == "MEDIUM"


def test_worst_case_is_reproducible():
    a = worst_case_per_pole(make_t1())
    b = worst_case_per_pole(make_t1())
    pd.testing.assert_frame_equal(a, b)


def test_priority_table_puts_the_worst_pole_first():
    top = priority_table(worst_case_per_pole(make_t1()))
    assert top.iloc[0]["TIANG_ID"] == "FDR1_10"
    assert top["RISK_SCORE"].is_monotonic_decreasing


# ─── Thermal ─────────────────────────────────────────────────────────────────

def make_t2():
    return pd.DataFrame([
        {"PENYULANG": "FDR1", "NO TIANG": "10", "SUHU FASA R": 30, "SUHU FASA S": 31,
         "SUHU FASA T": 52, "HI SUTM  - 1": "KURANG", "HI SUTM - 2": "CUKUP"},
        {"PENYULANG": "FDR1", "NO TIANG": "10", "SUHU FASA R": 30, "SUHU FASA S": 30,
         "SUHU FASA T": 31, "HI SUTM  - 1": "BAIK", "HI SUTM - 2": "BAIK"},
    ])


def test_delta_temperature_is_the_phase_spread():
    assert list(delta_temperature(make_t2())) == [22, 1]


def test_unreadable_temperatures_score_zero_rather_than_guessing():
    df = pd.DataFrame([{"SUHU FASA R": "n/a", "SUHU FASA S": None, "SUHU FASA T": "-"}])
    assert delta_temperature(df).iloc[0] == 0


def test_thermal_summary_takes_the_worst_reading_per_pole():
    agg = thermal_summary(make_t2()).set_index("TIANG_ID").loc["FDR1_10"]
    assert agg["DELTA_SUHU"] == 22
    assert agg["NUM_HI_1"] == 1  # KURANG, the lower of the two visits
    assert agg["NUM_HI_2"] == 2  # CUKUP


def test_poles_without_a_thermal_survey_are_marked_not_dropped():
    merged = merge_thermal(worst_case_per_pole(make_t1()), thermal_summary(make_t2()), make_t2().pipe(pole_id))
    assert len(merged) == 2
    by_id = merged.set_index("TIANG_ID")
    assert bool(by_id.loc["FDR1_10", "HAS_THERMAL"]) is True
    assert bool(by_id.loc["FDR2_7", "HAS_THERMAL"]) is False
    assert by_id.loc["FDR2_7", "DELTA_SUHU"] == 0
