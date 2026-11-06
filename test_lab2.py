import math

import numpy as np
import pandas as pd
from scipy import stats

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput
from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic


def test_titanic_basic():
    res = analyze_titanic(TitanicInput("titanic.csv"))
    df = pd.read_csv("titanic.csv")
    assert res.row_count == 891
    assert res.missing_by_column["Age"] == 177
    assert res.missing_by_column["Cabin"] == 687
    assert res.missing_by_column["Embarked"] == 2
    assert res.missing_by_column["Survived"] == 0
    assert res.adults_over_30_count == int((df.Age > 30).sum())
    assert set(res.mean_age_by_pclass) == {1, 2, 3}
    assert all(type(k) is int for k in res.mean_age_by_pclass)
    assert all(type(k) is int for k in res.survival_rate_by_pclass)
    assert math.isclose(res.mean_age_by_pclass[1], df[df.Pclass == 1].Age.dropna().mean())
    assert math.isclose(res.survival_rate_by_pclass[3], 119 / 491)
    assert len(res.highest_fares) == 5
    assert res.highest_fares == sorted(res.highest_fares, reverse=True)
    assert res.highest_fares[0] == df.Fare.max()


def test_titanic_small_file(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        "Survived,Pclass,Age,Fare\n1,1,40,10\n0,1,,20\n1,2,20,30\n0,3,31,40\n0,3,29,50\n1,3,,60\n1,3,35,70\n",
        encoding="utf-8",
    )
    res = analyze_titanic(TitanicInput(str(path)))
    assert res.row_count == 7
    assert res.missing_by_column == {"Survived": 0, "Pclass": 0, "Age": 2, "Fare": 0}
    assert res.adults_over_30_count == 3
    assert res.mean_age_by_pclass == {1: 40.0, 2: 20.0, 3: 31.666666666666668}
    assert math.isclose(res.survival_rate_by_pclass[3], 0.5)
    assert res.highest_fares == [70.0, 60.0, 50.0, 40.0, 30.0]


def test_brain_counts_and_values():
    res = analyze_brain_correlations(BrainDataInput("brainsize.txt"))
    df = pd.read_csv("brainsize.txt", sep="\t", na_values=["NA"])
    assert res.men_count == 20 and res.women_count == 20
    assert set(res.women_mri_correlation) == {"FSIQ", "VIQ", "PIQ", "Weight", "Height"}
    for gender, got in (("Female", res.women_mri_correlation), ("Male", res.men_mri_correlation)):
        g = df[df.Gender == gender]
        for f, value in got.items():
            sub = g[[f, "MRI_Count"]].dropna()
            expected = stats.pearsonr(sub[f], sub["MRI_Count"])[0]
            assert np.isclose(value, expected)


def test_brain_strongest_feature():
    res = analyze_brain_correlations(BrainDataInput("brainsize.txt"))
    best = max(
        res.women_mri_correlation,
        key=lambda f: max(abs(res.women_mri_correlation[f]), abs(res.men_mri_correlation[f])),
    )
    assert res.strongest_mri_feature == best == "PIQ"
    assert res.strongest_mri_feature != "MRI_Count"


def test_brain_negative_correlation_counts_by_absolute_value(tmp_path):
    path = tmp_path / "b.txt"
    rows = ["Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count"]
    for i in range(6):
        rows.append(f"Female\t{100 + i % 2}\t{i}\t{(i * 7) % 5}\tNA\t{60 + (i * 3) % 4}\t{1000 - 10 * i}")
        rows.append(f"Male\t{90 + i % 3}\t{(i * 5) % 4}\t{i}\t{150 + (i * 2) % 5}\t{70 + i % 2}\t{900 + 10 * i}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = analyze_brain_correlations(BrainDataInput(str(path)))
    assert res.women_mri_correlation["VIQ"] < -0.99
    assert res.strongest_mri_feature in {"VIQ", "PIQ"}
    assert math.isnan(res.women_mri_correlation["Weight"])
