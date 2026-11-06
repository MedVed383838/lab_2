"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]
 
    def correlations(group: pd.DataFrame) -> dict[str, float]:
        return {f: float(group[f].corr(group["MRI_Count"])) for f in features}
 
    women_corr = correlations(women)
    men_corr = correlations(men)
    strongest = max(
        features,
        key=lambda f: max(abs(women_corr[f]), abs(men_corr[f])),
    )
    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest,
    )